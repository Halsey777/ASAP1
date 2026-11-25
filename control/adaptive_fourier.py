"""Time-domain adaptive Fourier uncertainty estimator used by ASAP.

z(t) = [1, cos(w*t), sin(w*t), ..., cos(n*w*t), sin(n*w*t)]
Phi_hat(t) = W_hat.T @ z(t)
W_hat_next = projection(W_hat + dt * gamma * z(t) @ s(t).T)

gamma is Q^{-1} for Q=qI, hence gamma=1/q. Parameters are vector-valued;
W_hat has shape (2*order+1, parameter_count). No parameter-index waveform.
"""
import math
import torch


class AdaptiveFourierEstimator:
    def __init__(self, template, *, order=10, period=100., gain=.001, coefficient_limit=1., enabled=True):
        if template.ndim != 1 or not template.is_floating_point() or not torch.isfinite(template).all():
            raise ValueError("template must be a finite floating point parameter vector")
        if type(order) is not int or order < 0:
            raise ValueError("Fourier order must be a nonnegative integer")
        if not math.isfinite(period) or period <= 0 or not math.isfinite(gain) or gain < 0:
            raise ValueError("period must be positive; gain must be nonnegative")
        if not math.isfinite(coefficient_limit) or coefficient_limit <= 0:
            raise ValueError("coefficient_limit must be positive")
        self.order, self.period, self.gain = order, period, gain
        self.limit, self.enabled = coefficient_limit, bool(enabled)
        self.template = torch.zeros_like(template)
        # Disabled ablation does not allocate a large unused coefficient matrix.
        self.weights = template.new_zeros((2 * order + 1 if enabled else 0, template.numel()))

    def basis(self, time):
        if not math.isfinite(time):
            raise ValueError("Fourier time must be finite")
        values = [1.]
        for harmonic in range(1, self.order + 1):
            angle = 2 * math.pi * harmonic * time / self.period
            values.extend([math.cos(angle), math.sin(angle)])
        return self.template.new_tensor(values)

    @torch.no_grad()
    def estimate(self, time):
        if not self.enabled:
            return self.template.clone()
        return self.basis(time) @ self.weights

    @torch.no_grad()
    def adapt(self, time, sliding, *, dt):
        if sliding.shape != self.template.shape or not torch.isfinite(sliding).all():
            raise ValueError("Invalid sliding variable")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not self.enabled or self.gain == 0:
            return
        z = self.basis(time)
        # Row-wise in-place update avoids another (2n+1)*d outer-product tensor.
        # Each row uses the same PRE-aggregation sliding variable.
        for row, basis_value in zip(self.weights, z.tolist()):
            row.add_(sliding, alpha=dt * self.gain * basis_value)
            row.clamp_(-self.limit, self.limit)
            if not torch.isfinite(row).all():
                raise FloatingPointError("Nonfinite Fourier coefficients")
