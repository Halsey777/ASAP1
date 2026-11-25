"""Experimental discrete ASMC controller, NOT a reproduction of a proven theorem.

Only the broadcast controlled state, this controller's own submitted rate, and
the attacker-owned validation accuracy are observed. The controlled state is
normally the parameter vector; Tiny-ImageNet may append floating BN buffers. Test accuracy is never used.
All clients use delta submissions; malicious delta = dt * u.
The scalar response approximation, projection, and saturation are explicit new
engineering choices. They require validation; they do not inherit the paper's
continuous-time convergence guarantee.
"""
from dataclasses import asdict, dataclass
import math
import torch
from control.adaptive_fourier import AdaptiveFourierEstimator


@dataclass(frozen=True)
class ControllerConfig:
    k: float = 0.002
    eta: float = 0.001
    dt: float = 1.0
    harmonics: int = 10
    period: float = 100.0
    adaptation_gain: float = 0.001  # Q^{-1}=gamma I; Q=1000 I, not .001 I
    response_prior: float = 1.0
    response_lr: float = 0.05
    response_max: float = 2.0
    inverse_ridge: float = 0.01
    excitation_epsilon: float = 1e-12
    max_delta_norm: float = 10.0
    coefficient_limit: float = 1.0
    switching_boundary: float = 0.0
    switching_ramp_rounds: int = 0
    fs_enabled: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite controller setting: {key}")
        for key in ("k", "dt", "period", "response_max", "inverse_ridge",
                    "excitation_epsilon", "max_delta_norm", "coefficient_limit"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.eta < 0 or self.adaptation_gain < 0 or self.switching_boundary < 0:
            raise ValueError("eta, adaptation_gain, and switching_boundary must be nonnegative")
        if not 0 <= self.response_lr <= 1:
            raise ValueError("response_lr must be in [0, 1]")
        if not 0 < self.response_prior <= self.response_max:
            raise ValueError("response_prior must be in (0, response_max]")
        if type(self.harmonics) is not int or self.harmonics < 0:
            raise ValueError("harmonics must be a nonnegative integer")
        if type(self.switching_ramp_rounds) is not int or self.switching_ramp_rounds < 0:
            raise ValueError("switching_ramp_rounds must be a nonnegative integer")


class AdaptiveController:
    def __init__(self, reference, config: ControllerConfig):
        if reference.ndim != 1 or not torch.isfinite(reference).all():
            raise ValueError("Reference must be a finite flat control-state vector")
        self.reference = reference.detach().clone()
        self.config = config
        self.sliding = torch.zeros_like(reference)
        self.fourier = AdaptiveFourierEstimator(
            reference, order=config.harmonics, period=config.period,
            gain=config.adaptation_gain, coefficient_limit=config.coefficient_limit,
            enabled=config.fs_enabled,
        )
        self.response = config.response_prior
        self.step = 0
        self.pending = None
        self.previous_feedback = None
        self.response_updates = 0
        self.response_rejections = 0
        self.metric_feedback_scale = 1.0
        self.metric_error_pp = None

    def observe_accuracy(self, accuracy, target, *, gain, max_scale):
        values = (accuracy, target, gain, max_scale)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Accuracy feedback settings and values must be finite")
        if not 0 <= accuracy <= 100 or not 0 <= target <= 100:
            raise ValueError("Accuracy and target must be in [0, 100]")
        if gain < 0 or max_scale < 1:
            raise ValueError("Accuracy feedback gain must be nonnegative and max_scale at least 1")
        self.metric_error_pp = accuracy - target
        self.metric_feedback_scale = min(max_scale, max(0.25, 1 + gain * self.metric_error_pp))

    def basis(self, step):
        return self.fourier.basis(step * self.config.dt)

    @property
    def coefficients(self):
        return self.fourier.weights

    @torch.no_grad()
    def propose(self, current, *, active):
        if self.pending is not None:
            raise RuntimeError("Observe the previous aggregation before another proposal")
        if current.shape != self.reference.shape or not torch.isfinite(current).all():
            raise ValueError("Invalid broadcast control-state vector")
        cfg = self.config
        error = current - self.reference
        z = self.basis(self.step)
        disturbance = self.fourier.estimate(self.step * cfg.dt)
        inverse = self.response / (self.response ** 2 + cfg.inverse_ridge)
        if cfg.switching_boundary > 0:
            switching = (self.sliding / cfg.switching_boundary).clamp(-1.0, 1.0)
        else:
            switching = self.sliding.sign()
        if cfg.switching_ramp_rounds > 0:
            switching_scale = min(1.0, self.step / cfg.switching_ramp_rounds)
        else:
            switching_scale = 1.0
        rate = inverse * (-cfg.k * error - cfg.eta * switching_scale * switching - disturbance)
        if not torch.isfinite(rate).all():
            raise FloatingPointError("Nonfinite control command")
        raw_delta = cfg.dt * rate * self.metric_feedback_scale
        raw_norm = float(raw_delta.norm())
        if not math.isfinite(raw_norm):
            raise FloatingPointError("Control norm overflow")
        saturated = raw_norm > cfg.max_delta_norm
        scale = min(1.0, cfg.max_delta_norm / max(raw_norm, cfg.excitation_epsilon))
        delta = raw_delta * scale if active else torch.zeros_like(raw_delta)
        self.pending = {
            "current": current.detach().clone(), "error": error,
            "rate": delta / cfg.dt, "disturbance": disturbance,
            "basis": z, "active": bool(active), "saturated": saturated,
        }
        diagnostics = {
            "response_estimate": self.response,
            "inverse_response": inverse,
            "sliding_norm": float(self.sliding.norm()),
            "switching_norm": float(switching.norm()),
            "switching_boundary": cfg.switching_boundary,
            "switching_scale": switching_scale,
            "disturbance_norm": float(disturbance.norm()),
            "raw_delta_norm": raw_norm, "submitted_delta_norm": float(delta.norm()),
            "saturated": bool(saturated and active),
            "response_updates": self.response_updates,
            "response_rejections": self.response_rejections,
            "accuracy_error_pp": self.metric_error_pp,
            "accuracy_feedback_scale": self.metric_feedback_scale,
            "fs_enabled": cfg.fs_enabled,
            "fourier_coefficient_norm": float(self.coefficients.norm()),
        }
        return delta, diagnostics

    @torch.no_grad()
    def observe(self, next_global):
        if self.pending is None:
            raise RuntimeError("No pending control to associate with this observation")
        if next_global.shape != self.reference.shape or not torch.isfinite(next_global).all():
            raise ValueError("Invalid next broadcast control-state vector")
        cfg, p = self.config, self.pending
        change = next_global - p["current"]
        velocity = change / cfg.dt
        # Differencing removes a CONSTANT unknown drift; varying drift still
        # confounds this estimate. Use only consecutive active rounds. The
        # positive scalar model is an explicit assumption, not an AGR identity.
        if p["active"] and self.previous_feedback is not None:
            old_rate, old_velocity, old_disturbance = self.previous_feedback
            change_rate = p["rate"] - old_rate
            energy = float(change_rate.square().sum())
            if energy > cfg.excitation_epsilon:
                response_change = velocity - old_velocity - (p["disturbance"] - old_disturbance)
                estimate = float(torch.dot(change_rate, response_change)) / energy
                if math.isfinite(estimate) and 0 <= estimate <= cfg.response_max:
                    self.response = (1 - cfg.response_lr) * self.response + cfg.response_lr * estimate
                    self.response_updates += 1
                else:
                    # Reject observations inconsistent with the declared scalar
                    # model rather than treating them as evidence of zero gain.
                    self.response_rejections += 1
        self.previous_feedback = (p["rate"].clone(), velocity.clone(), p["disturbance"].clone()) if p["active"] else None
        # Explicit Euler update uses PRE-aggregation s_t, not future s_(t+1).
        # Freeze learning under command saturation and absent participation.
        if p["active"] and not p["saturated"] and cfg.adaptation_gain:
            self.fourier.adapt(self.step * cfg.dt, self.sliding, dt=cfg.dt)
        self.sliding.add_(change + cfg.dt * cfg.k * p["error"])
        if not torch.isfinite(self.sliding).all():
            raise FloatingPointError("Nonfinite controller state")
        self.step += 1
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Save controller state only between rounds")
        return {
            "config": asdict(self.config), "reference": self.reference.cpu().clone(),
            "sliding": self.sliding.cpu().clone(),
            "coefficients": self.coefficients.cpu().clone(),
            "response": self.response, "step": self.step,
            "previous_feedback": None if self.previous_feedback is None else [v.cpu().clone() for v in self.previous_feedback],
            "response_updates": self.response_updates, "response_rejections": self.response_rejections,
            "metric_feedback_scale": self.metric_feedback_scale,
            "metric_error_pp": self.metric_error_pp,
        }
