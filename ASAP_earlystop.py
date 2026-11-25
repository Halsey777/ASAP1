"""Same-run reference capture followed by immediate next-round control.

This is a redesigned experimental implementation, not validated paper results.
Run --help for configuration; README.md specifies the protocol and assumptions.
"""
import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
# Required by deterministic CUDA matmul on supported CUDA versions. Set before
# the first CUDA operation; CPU behavior is unchanged.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import Subset

from control.asap_controller import AdaptiveController, ControllerConfig
from fl.aggregators import (CifarAlexNet, DeltaAggregator, aggregate_buffers,
                           evaluate, flatten, flatten_control_state,
                           flatten_floating_buffers, load_flat, local_delta,
                           split_control_delta)
from fl.data import (load_data, split_and_partition, split_partition_and_root,
                     split_attacker_validation)
from control.reference_protocol import ReferenceRule, ReferenceSelector, objective_stabilization_rounds


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cifar10", "mnist", "tinyimagenet", "synthetic"], default="cifar10")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
    rule = p.add_mutually_exclusive_group(required=True)
    rule.add_argument("--target-accuracy", type=float, help="Validation accuracy threshold, percentage units")
    rule.add_argument("--reference-round", type=int, help="Optional fixed-round capture protocol")
    p.add_argument("--reference-tolerance-pp", type=float, default=None,
                   help="Optional symmetric capture band around target; capture is strict when a band is set")
    p.add_argument("--allow-reference-crossing-fallback", action="store_true",
                   help="Opt in to nearest-checkpoint fallback if benign training jumps over the target band")
    p.add_argument("--capture-initial-reference", action="store_true",
                   help="Use the round-0 global model as reference while retaining the requested target for metrics")
    p.add_argument("--max-warmup-rounds", type=int, default=200)
    p.add_argument("--control-rounds", "--epochs", type=int, default=100)
    p.add_argument("--max-global-rounds", type=int, default=None,
                   help="Optional hard cap on total FL communication rounds, including warm-up and control")
    p.add_argument("--num-clients", "--num_users", type=int, default=50)
    p.add_argument("--participation", "--frac", type=float, default=1.0)
    p.add_argument("--malicious-fraction", type=float, default=0.1)
    p.add_argument("--attacker-validation-fraction", "--validation-fraction",
                   dest="attacker_validation_fraction", type=float, default=0.1,
                   help="Fraction held out from each malicious client local partition for reference selection")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--local-epochs", "--local_ep", type=int, default=5)
    p.add_argument("--batch-size", "--local_bs", type=int, default=10)
    p.add_argument("--eval-batch-size", "--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0 etc.")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--agr", choices=["fedavg", "median", "trimmed_mean", "norm_bound", "bulyan",
                                     "multi_krum", "fltrust", "centered_clip", "dnc"], default="fedavg")
    p.add_argument("--byzantine-bound", type=int,
                   help="Server-side f; required for trimmed_mean / Bulyan / Multi-Krum / DNC")
    p.add_argument("--multi-krum-m", type=int, default=None,
                   help="Number of lowest Krum-score updates averaged by Multi-Krum; default n-f")
    p.add_argument("--clip-norm", type=float, default=2.5)
    p.add_argument("--server-root-samples", type=int, default=0,
                   help="Deprecated compatibility option; Oct-2 legacy FLTrust does not use server root data")
    p.add_argument("--fltrust-root-epochs", type=int, default=1)
    p.add_argument("--fltrust-root-batch-size", type=int, default=None)
    p.add_argument("--dnc-subsample-dim", type=int, default=10000)
    p.add_argument("--dnc-iters", type=int, default=1)
    p.add_argument("--dnc-filter-factor", type=float, default=1.0)
    p.add_argument("--aggregation-chunk-size", type=int, default=200000,
                   help="Parameter chunk size for coordinate rules and pairwise distances")
    p.add_argument("--attack-mode", choices=["adaptive", "smc", "replay", "zero"], default="adaptive")
    p.add_argument("--k", type=float, default=0.002)
    p.add_argument("--eta", type=float, default=0.001)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--harmonics", type=int, default=10)
    p.add_argument("--fourier-period", type=float, default=100.0)
    gain = p.add_mutually_exclusive_group()
    gain.add_argument("--adaptation-gain", type=float, default=None, help="gamma=1/q; default .001")
    gain.add_argument("--q", type=float, default=None, help="Q=qI; equivalent to adaptation-gain=1/q")
    p.add_argument("--disable-fs", action="store_true", help="Disable both Fourier compensation and adaptation")
    p.add_argument("--response-prior", type=float, default=1.0)
    p.add_argument("--response-lr", type=float, default=0.05)
    p.add_argument("--response-max", type=float, default=2.0)
    p.add_argument("--inverse-ridge", type=float, default=0.01)
    p.add_argument("--max-delta-norm", type=float, default=10.0)
    p.add_argument("--coefficient-limit", type=float, default=1.0)
    p.add_argument("--switching-boundary", type=float, default=0.0,
                   help="Boundary-layer width phi for sat(s/phi); 0 keeps hard sign(s)")
    p.add_argument("--switching-ramp-rounds", type=int, default=0,
                   help="Ramp eta*sign(s) from zero to full strength over the first N control rounds")
    p.add_argument("--osr-window", type=int, default=10)
    p.add_argument("--metric-tolerance-pp", type=float, default=2.0)
    p.add_argument("--accuracy-feedback-gain", type=float, default=0.0,
                   help="Scale the adaptive command from attacker-validation error in percentage points")
    p.add_argument("--accuracy-feedback-max-scale", type=float, default=6.0,
                   help="Maximum multiplier for the adaptive command")
    p.add_argument("--model-file", help="Optional local Python file exporting a model factory")
    p.add_argument("--model-symbol", default="build_model", help="Factory called with num_classes=...")
    p.add_argument("--initial-weights", choices=["none", "imagenet-v1", "imagenet-v2"],
                   help="Required for built-in Tiny-ImageNet ResNet50; explicit pretraining choice")
    return p


def validate(args):
    for name in ("max_warmup_rounds", "control_rounds", "num_clients", "local_epochs",
                 "batch_size", "eval_batch_size", "cpu_threads", "osr_window",
                 "fltrust_root_epochs", "dnc_subsample_dim", "dnc_iters", "aggregation_chunk_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_global_rounds is not None and args.max_global_rounds < 1:
        raise ValueError("max_global_rounds must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.dataset == "tinyimagenet" and not args.model_file and args.initial_weights is None:
        raise ValueError("Declare --initial-weights none or imagenet-v2 for Tiny-ImageNet")
    if args.initial_weights == "imagenet-v2" and (args.dataset != "tinyimagenet" or args.model_file):
        raise ValueError("imagenet-v2 applies only to built-in Tiny-ImageNet ResNet50")
    for name in ("participation", "malicious_fraction", "attacker_validation_fraction", "lr", "momentum",
                 "metric_tolerance_pp"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"Nonfinite setting: {name}")
    if not 0 < args.participation <= 1 or not 0 < args.malicious_fraction < .5:
        raise ValueError("Require 0 < participation <= 1 and 0 < malicious_fraction < .5")
    if not 0 < args.attacker_validation_fraction < 1:
        raise ValueError("attacker_validation_fraction must be in (0,1)")
    if args.lr <= 0 or args.momentum < 0:
        raise ValueError("Invalid SGD configuration")
    if args.reference_tolerance_pp is not None and (
        not math.isfinite(args.reference_tolerance_pp) or args.reference_tolerance_pp < 0
    ):
        raise ValueError("Reference tolerance must be finite and nonnegative")
    if args.metric_tolerance_pp < 0:
        raise ValueError("Tolerances must be nonnegative")
    if (not math.isfinite(args.accuracy_feedback_gain) or args.accuracy_feedback_gain < 0
            or not math.isfinite(args.accuracy_feedback_max_scale) or args.accuracy_feedback_max_scale < 1):
        raise ValueError("Invalid accuracy feedback configuration")
    if not math.isfinite(args.switching_boundary) or args.switching_boundary < 0:
        raise ValueError("switching_boundary must be finite and nonnegative")
    if args.switching_ramp_rounds < 0:
        raise ValueError("switching_ramp_rounds must be nonnegative")
    if args.q is not None and (not math.isfinite(args.q) or args.q <= 0):
        raise ValueError("q must be finite and positive")
    malicious_count = int(args.num_clients * args.malicious_fraction)
    if malicious_count < 1:
        raise ValueError("Configured malicious fraction rounds down to zero clients")
    if args.server_root_samples < 0:
        raise ValueError("server_root_samples must be nonnegative")
    if args.fltrust_root_batch_size is not None and args.fltrust_root_batch_size < 1:
        raise ValueError("fltrust_root_batch_size must be positive")
    if not math.isfinite(args.dnc_filter_factor) or args.dnc_filter_factor <= 0:
        raise ValueError("dnc_filter_factor must be finite and positive")
    if args.agr in ("trimmed_mean", "multi_krum", "bulyan", "dnc") and args.byzantine_bound is None:
        raise ValueError("Declare --byzantine-bound for this aggregation rule")
    if args.byzantine_bound is not None and args.byzantine_bound < 0:
        raise ValueError("byzantine_bound must be nonnegative")
    if args.multi_krum_m is not None and args.multi_krum_m < 1:
        raise ValueError("multi_krum_m must be positive")
    n = max(1, int(args.num_clients * args.participation))
    f = args.byzantine_bound or 0
    if args.agr == "trimmed_mean" and n <= 2 * f:
        raise ValueError("Sampled clients must satisfy n > 2f")
    if args.agr == "multi_krum":
        if n < 2 * f + 3:
            raise ValueError("Sampled clients must satisfy n >= 2f+3")
        if args.multi_krum_m is not None and args.multi_krum_m > n - f:
            raise ValueError("Multi-Krum requires m <= n-f")
    if args.agr == "bulyan" and n < 4 * f + 3:
        raise ValueError("Bulyan requires sampled clients n >= 4f+3")
    if args.agr == "dnc" and f >= n:
        raise ValueError("DNC requires sampled clients n > f")
    return malicious_count


def make_model(args, classes):
    if args.model_file:
        path = Path(args.model_file).resolve()
        spec = importlib.util.spec_from_file_location("asap_user_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, args.model_symbol)(num_classes=classes)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model factory must return torch.nn.Module")
        return model
    if args.dataset == "synthetic":
        return torch.nn.Linear(4, classes)
    if args.dataset == "cifar10":
        return CifarAlexNet(classes)
    if args.dataset == "mnist":
        from fl.models.Nets import MLP
        return MLP(dim_in=784, dim_hidden=200, dim_out=classes)
    from torchvision.models import resnet50, ResNet50_Weights
    if args.initial_weights == "imagenet-v2":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
        return model
    if args.initial_weights == "imagenet-v1":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
        return model
    return resnet50(weights=None, num_classes=classes)


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def save_checkpoint(path, model, server, *, controller=None, **metadata):
    checkpoint = {
        "state_dict": cpu_state(model), "server_center": None if server.center is None else server.center.cpu().clone(),
        "server_config": server.config_dict(),
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "controller": None if controller is None else controller.state_dict(), **metadata,
    }
    torch.save(checkpoint, path)


def participants(args, global_round):
    rng = np.random.default_rng(args.seed + 1_000_003 * global_round)
    count = max(1, int(args.num_clients * args.participation))
    return sorted(rng.choice(args.num_clients, count, replace=False).tolist())


def train_round(model, server, dataset, partitions, chosen, malicious, delta, args, device, global_round,
                server_root_indices=None, malicious_buffer_delta=None):
    """Run one FL round using the Oct-2 legacy ASAP AGR semantics.

    The 50-client branch keeps its attacker-side validation and controller path,
    but aggregation of model parameters follows the Oct-2 implementation.
    """
    before = flatten(model)
    updates, buffers, counts = [], [], []
    streaming = server.name in ("fedavg", "centered_clip")
    accumulator = torch.zeros_like(before) if streaming else None
    old_center = (
        torch.zeros_like(before) if server.center is None else server.center
    ) if server.name == "centered_clip" else None
    max_client_norm = 0.0
    benign_norms = []
    malicious_norms = []

    for client in chosen:
        counts.append(len(partitions[client]))
        if delta is not None and client in malicious:
            import os as _osS
            if _osS.environ.get("STEALTH_MODE") == "1":
                _honest, _lb = local_delta(
                    model, dataset, partitions[client], device=device,
                    seed=(args.seed + 1_000_003 * global_round + 1009 * client) % (2**63 - 1),
                    epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
                )
                _bud = float(_osS.environ.get("STEALTH_BUDGET", "0.3"))
                _dn = float(delta.norm())
                update = _honest + (delta * (_bud / _dn) if _dn > 1e-12 else torch.zeros_like(delta))
                buffers.append(_lb)
            else:
                update = delta
                submitted_buffers = {}
                for name, value in model.named_buffers():
                    submitted = value.detach().clone()
                    if malicious_buffer_delta and name in malicious_buffer_delta:
                        if not submitted.is_floating_point():
                            raise ValueError(f"Cannot control non-floating buffer: {name}")
                        adjustment = malicious_buffer_delta[name].to(
                            device=submitted.device, dtype=submitted.dtype
                        )
                        if adjustment.shape != submitted.shape or not torch.isfinite(adjustment).all():
                            raise ValueError(f"Invalid malicious buffer delta: {name}")
                        submitted.add_(adjustment)
                        if not torch.isfinite(submitted).all():
                            raise FloatingPointError(f"Nonfinite malicious buffer submission: {name}")
                    submitted_buffers[name] = submitted
                buffers.append(submitted_buffers)
        else:
            update, local_buffers = local_delta(
                model, dataset, partitions[client], device=device,
                seed=(args.seed + 1_000_003 * global_round + 1009 * client) % (2**63 - 1),
                epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
            )
            buffers.append(local_buffers)

        update_norm = float(update.norm())
        if not math.isfinite(update_norm):
            raise FloatingPointError("Nonfinite client delta norm")
        max_client_norm = max(max_client_norm, update_norm)
        if delta is not None and client in malicious:
            malicious_norms.append(update_norm)
        else:
            benign_norms.append(update_norm)

        if server.name == "fedavg":
            # Oct-2 legacy ASAP FedAvg: unweighted client mean.
            accumulator.add_(update, alpha=1.0 / len(chosen))
        elif server.name == "centered_clip":
            residual = update - old_center
            scale = min(1.0, server.clip_norm / max(float(residual.norm()), 1e-12))
            accumulator.add_(residual, alpha=scale / len(chosen))
        else:
            updates.append(update)

    if streaming:
        server.last_selected = list(range(len(chosen)))
        aggregated = accumulator
        if server.name == "centered_clip":
            server.center = old_center + accumulator
            aggregated = server.center.clone()
    else:
        aggregated = server(
            torch.stack(updates),
            counts,
            round_seed=(args.seed + 1_000_003 * global_round) % (2**63 - 1),
            base_model=before,
        )

    import os as _os
    if _os.environ.get("FORCE_SELECT_MAL")=="1" and delta is not None and not streaming:
        _mal=[i for i,c in enumerate(chosen) if c in malicious]
        if _mal:
            _sel=sorted(set(server.last_selected)|set(_mal))
            server.last_selected=_sel
            aggregated=torch.stack(updates)[_sel].mean(dim=0)
    load_flat(model, before + aggregated)

    # Restored: aggregate local BatchNorm/statistical buffers back into the
    # global model (required for ResNet-50 Tiny-ImageNet to learn).
    selected = server.last_selected if server.name in server.FILTERING_RULES else None
    aggregate_buffers(model, buffers, counts, selected=selected)

    selected_clients = [chosen[i] for i in server.last_selected]
    selected_malicious = [client for client in selected_clients if client in malicious]
    return {
        "global_delta_norm": float(aggregated.norm()),
        "max_client_delta_norm": max_client_norm,
        "benign_delta_norm_mean": (
            sum(benign_norms) / len(benign_norms) if benign_norms else 0.0
        ),
        "benign_delta_norm_max": max(benign_norms) if benign_norms else 0.0,
        "malicious_delta_norm_mean": (
            sum(malicious_norms) / len(malicious_norms) if malicious_norms else 0.0
        ),
        "selected_clients": selected_clients,
        "selected_malicious_clients": selected_malicious,
        "selected_malicious_count": len(selected_malicious),
        "malicious_survival_rate": (
            len(selected_malicious) / len(set(chosen) & malicious)
            if set(chosen) & malicious else 0.0
        ),
    }

class MetricLog:
    columns = ["global_round", "control_round", "phase", "reference_round", "reference_captured",
               "participant_ids", "malicious_participants", "actual_malicious_fraction",
               "byzantine_bound_exceeded",
               "validation_accuracy", "validation_loss", "test_accuracy", "test_loss",
               "parameter_distance", "relative_parameter_distance", "floating_buffer_distance",
               "control_state_distance", "global_delta_norm",
               "max_client_delta_norm", "benign_delta_norm_mean", "benign_delta_norm_max",
               "malicious_delta_norm_mean", "selected_clients", "selected_malicious_clients",
               "selected_malicious_count", "malicious_survival_rate",
               "response_estimate", "inverse_response",
               "sliding_norm", "switching_norm", "switching_boundary", "switching_scale",
               "disturbance_norm", "raw_delta_norm", "submitted_delta_norm",
               "saturated", "response_updates", "response_rejections", "fs_enabled",
               "fourier_coefficient_norm", "accuracy_error_pp", "accuracy_feedback_scale",
               "elapsed_seconds"]

    def __init__(self, path):
        self.file = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                              for key, value in row.items()})
        self.file.flush()

    def close(self):
        self.file.close()


def metric_summary(trace, target, args):
    values = [row["test_accuracy"] for row in trace]
    deviation = [abs(value - target) for value in values]
    final = values[-1]
    tail = values[-min(args.osr_window, len(values)):]
    return {
        "target_accuracy": target, "post_control_rounds": len(values),
        "osr": objective_stabilization_rounds(values, target_accuracy=target,
                                              tolerance_pp=args.metric_tolerance_pp, window=args.osr_window),
        "within_band_fraction": sum(d <= args.metric_tolerance_pp for d in deviation) / len(values),
        "mean_absolute_deviation_pp": sum(deviation) / len(values),
        "maximum_absolute_deviation_pp": max(deviation),
        "final_accuracy": final, "tail_mean_accuracy": sum(tail) / len(tail),
        "final_absolute_deviation_pp": abs(final - target),
        "final_normalized_absolute_deviation_percent": 100 * abs(final - target) / target if target else None,
        "first_band_exit_round": next((i + 1 for i, d in enumerate(deviation) if d > args.metric_tolerance_pp), None),
    }


def run(args):
    malicious_count = validate(args)
    gain = 1. / args.q if args.q is not None else (.001 if args.adaptation_gain is None else args.adaptation_gain)
    cfg = ControllerConfig(
        k=args.k, eta=args.eta, dt=args.dt, harmonics=args.harmonics, period=args.fourier_period,
        adaptation_gain=gain, response_prior=args.response_prior, fs_enabled=not args.disable_fs,
        response_lr=args.response_lr, response_max=args.response_max, inverse_ridge=args.inverse_ridge,
        max_delta_norm=args.max_delta_norm, coefficient_limit=args.coefficient_limit,
        switching_boundary=args.switching_boundary,
        switching_ramp_rounds=args.switching_ramp_rounds,
    )
    if args.attack_mode == "smc":
        cfg = replace(cfg, adaptation_gain=0.0)
    rule = ReferenceRule(
        target_accuracy=args.target_accuracy,
        stop_round=args.reference_round,
        max_overshoot_pp=args.reference_tolerance_pp if args.target_accuracy is not None else None,
        allow_crossing_fallback=args.allow_reference_crossing_fallback,
        force_initial_reference=args.capture_initial_reference,
    )
    if args.reference_round and args.reference_round > args.max_warmup_rounds:
        raise ValueError("reference-round exceeds max-warmup-rounds")
    if args.reference_round and args.max_global_rounds is not None and args.reference_round > args.max_global_rounds:
        raise ValueError("reference-round exceeds max-global-rounds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script_root = Path(__file__).resolve().parent
    sources = [script_root / name for name in ("ASAP_earlystop.py", "asap_controller.py",
                                              "fl_components.py", "fl_data.py", "reference_protocol.py", "adaptive_fourier.py")]
    if args.dataset == "mnist" and not args.model_file:
        sources.append(script_root / "models" / "Nets.py")
    if args.model_file:
        sources.append(Path(args.model_file).resolve())
    # Oct-2 legacy ASAP aggregated/controlled model parameters only.
    # Local BN running buffers are deliberately not aggregated back to the global model.
    control_floating_buffers = False  # param-only control; benign BN still aggregated (avoids running_var->NaN)
    manifest = {
        "protocol": "same-run-first-crossing-then-next-round-control",
        "implementation_status": "experimental; not validated paper results",
        "args": vars(args), "effective_controller": asdict(cfg), "device": str(device),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "actual_total_malicious_fraction": malicious_count / args.num_clients,
        "submission_contract": "all clients submit model deltas; server adds aggregated delta",
        "control_state_policy": "parameters only (Oct-2 legacy ASAP)",
        "buffer_policy": (
            "Oct-2 legacy behavior: local model buffers, including BatchNorm running_mean/running_var, "
            "are not aggregated back into the global model."
        ),
        "test_role": "offline reporting only; never selects reference or enters controller",
        "reference_validation_role": "attacker-owned holdout from malicious clients only",
        "attacker_validation_fraction": args.attacker_validation_fraction,
        "reference_crossing_fallback_enabled": args.allow_reference_crossing_fallback,
        "initial_reference_forced": args.capture_initial_reference,
        "Q_diagonal": 1 / cfg.adaptation_gain if cfg.adaptation_gain else None,
        "observation_assumption": "global broadcast visible every round, including unsampled rounds",
    }
    json_write(output / "config.json", manifest)
    logger = MetricLog(output / "metrics.csv")
    start = time.perf_counter()
    try:
        dataset, test_set, labels, classes = load_data(
            args.dataset, args.data_root, download=args.download, seed=args.seed
        )

        # Fixed tail identities match the original 50-client ASAP convention.
        # With 50 clients and malicious_fraction=0.10 this is exactly 45..49.
        malicious = set(range(args.num_clients - malicious_count, args.num_clients))

        # Oct-2 legacy AGRs do not reserve trusted server data.
        partitions = split_and_partition(
            labels, num_clients=args.num_clients, seed=args.seed
        )
        server_root_ids = []

        # Reference selection uses ONLY data owned by malicious clients.
        # Each malicious client withholds a small local validation subset; the attacker
        # evaluates the broadcast global model on the union of those five subsets.
        partitions, attacker_validation_ids, attacker_validation_by_client = split_attacker_validation(
            partitions,
            malicious,
            validation_fraction=args.attacker_validation_fraction,
            seed=args.seed + 1701,
        )

        split = {
            "clients": partitions,
            "attacker_validation": attacker_validation_ids,
            "attacker_validation_by_client": {
                str(client): ids for client, ids in attacker_validation_by_client.items()
            },
            "server_root": server_root_ids,
        }
        json_write(output / "partitions.json", split)
        split_hash = hashlib.sha256((output / "partitions.json").read_bytes()).hexdigest()
        json_write(output / "malicious_clients.json", sorted(malicious))
        json_write(
            output / "attacker_validation.json",
            {
                "malicious_clients": sorted(malicious),
                "fraction_per_malicious_client": args.attacker_validation_fraction,
                "sample_count": len(attacker_validation_ids),
                "samples_by_client": {
                    str(client): ids for client, ids in attacker_validation_by_client.items()
                },
            },
        )
        attacker_validation_set = Subset(dataset, attacker_validation_ids)
        client_sizes = [len(partition) for partition in partitions]
        validation_sizes = {
            client: len(indices) for client, indices in attacker_validation_by_client.items()
        }
        print(
            "data setup: "
            f"clients={args.num_clients} malicious={len(malicious)} "
            f"malicious_ids={sorted(malicious)} "
            f"client_train_min={min(client_sizes)} client_train_max={max(client_sizes)} "
            f"attacker_validation_total={len(attacker_validation_ids)} "
            f"attacker_validation_by_client={validation_sizes} "
            f"server_root={len(server_root_ids)}",
            flush=True,
        )
        selector = ReferenceSelector(rule, selection_split_id=split_hash)
        model = make_model(args, classes).to(device)
        server = DeltaAggregator(
            args.agr,
            byzantine_bound=args.byzantine_bound or 0,
            clip_norm=args.clip_norm,
            coordinate_chunk_size=args.aggregation_chunk_size,
            multi_krum_m=args.multi_krum_m,
            dnc_subsample_dim=args.dnc_subsample_dim,
            dnc_iters=args.dnc_iters,
            dnc_filter_factor=args.dnc_filter_factor,
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        floating_buffer_count = sum(
            value.numel() for _, value in model.named_buffers() if value.is_floating_point()
        )
        control_state_count = parameter_count + (floating_buffer_count if control_floating_buffers else 0)
        json_write(output / "model.json", {
            "repr": repr(model),
            "parameter_count": parameter_count,
            "floating_buffer_count": floating_buffer_count,
            "control_state_count": control_state_count,
            "control_floating_buffers": control_floating_buffers,
            "buffers": [name for name, _ in model.named_buffers()],
            "fourier_coefficients_bytes_float32":
                (4 * (2 * cfg.harmonics + 1) * control_state_count
                 if cfg.fs_enabled and args.attack_mode not in ("replay", "zero") else 0),
        })
        save_checkpoint(output / "initial.pt", model, server, config=manifest)
        reference_round = None
        reference_test_accuracy = None
        previous_server_center = None

        # Target-based runs inspect the untrained global model before any benign
        # local update. Balanced classification tasks can already be at their
        # random-guess target at round 0 (10% for MNIST/CIFAR-10, 0.5% for
        # Tiny-ImageNet). In that case control begins immediately at round 1.
        if args.target_accuracy is not None:
            initial_validation = evaluate(model, attacker_validation_set, device, args.eval_batch_size)
            initial_test = evaluate(model, test_set, device, args.eval_batch_size)
            initial_captured = selector.observe_initial(initial_validation["accuracy"], model.state_dict())
            logger.write({"global_round": 0, "control_round": 0, "phase": "initial",
                          "reference_round": 0 if initial_captured else "",
                          "reference_captured": initial_captured,
                          "participant_ids": [], "malicious_participants": 0,
                          "actual_malicious_fraction": 0, "byzantine_bound_exceeded": False,
                          "validation_accuracy": initial_validation["accuracy"],
                          "validation_loss": initial_validation["loss"],
                          "test_accuracy": initial_test["accuracy"], "test_loss": initial_test["loss"],
                          "global_delta_norm": 0.0, "max_client_delta_norm": 0.0,
                          "selected_clients": [], "elapsed_seconds": time.perf_counter() - start})
            print(f"initial round=0 validation={initial_validation['accuracy']:.4f} "
                  f"capture={initial_captured}", flush=True)
            if initial_captured:
                checkpoint = selector.checkpoint()
                reference_round = 0
                reference_test_accuracy = initial_test["accuracy"]
                target_model = copy.deepcopy(model).eval()
                target_model.requires_grad_(False)
                torch.save(cpu_state(target_model), output / "target_model.pt")
                checkpoint["state_dict"] = {key: value.cpu() for key, value in checkpoint["state_dict"].items()}
                checkpoint.update({"reference_test_accuracy": reference_test_accuracy,
                                   "control_starts_global_round": 1,
                                   "class_count": classes, "config": manifest})
                torch.save(checkpoint, output / "reference.pt")
                save_checkpoint(output / "control_start.pt", model, server, global_round=0,
                                split_sha256=split_hash, malicious_ids=sorted(malicious), config=manifest)

        if reference_round is None:
            warmup_limit = args.max_warmup_rounds
            if args.max_global_rounds is not None:
                warmup_limit = min(warmup_limit, args.max_global_rounds)
            for round_number in range(1, warmup_limit + 1):
                chosen = participants(args, round_number)
                training = train_round(model, server, dataset, partitions, chosen, set(), None, args, device, round_number,
                                       server_root_ids)
                validation = evaluate(model, attacker_validation_set, device, args.eval_batch_size)
                current_server_center = None if server.center is None else server.center.detach().clone()
                captured = selector.observe(round_number, validation["accuracy"], model.state_dict())
                selected_checkpoint = selector.checkpoint() if captured else None
                test = evaluate(model, test_set, device, args.eval_batch_size)
                logger.write({"global_round": round_number, "control_round": 0, "phase": "benign",
                              "reference_round": selected_checkpoint["reference_round"] if captured else "",
                              "reference_captured": captured,
                              "participant_ids": chosen, "malicious_participants": 0, "actual_malicious_fraction": 0,
                              "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"],
                              "test_accuracy": test["accuracy"], "test_loss": test["loss"],
                              "elapsed_seconds": time.perf_counter() - start, **training})
                print(f"benign round={round_number} validation={validation['accuracy']:.4f} capture={captured}", flush=True)
                if captured:
                    checkpoint = selected_checkpoint
                    reference_round = checkpoint["reference_round"]

                    # A nearest-crossing fallback may select the immediately previous
                    # benign checkpoint. Restore that exact model/server state before
                    # control so the discarded overshoot round cannot leak into the run.
                    if reference_round != round_number:
                        if reference_round != round_number - 1:
                            raise RuntimeError("Fallback reference must be current or immediately previous round")
                        model.load_state_dict(checkpoint["state_dict"])
                        server.center = None if previous_server_center is None else previous_server_center.clone()

                    reference_test_accuracy = evaluate(model, test_set, device, args.eval_batch_size)["accuracy"]
                    target_model = copy.deepcopy(model).eval()
                    target_model.requires_grad_(False)
                    torch.save(cpu_state(target_model), output / "target_model.pt")
                    checkpoint["state_dict"] = {key: value.cpu() for key, value in checkpoint["state_dict"].items()}
                    checkpoint.update({"reference_test_accuracy": reference_test_accuracy,
                                       "control_starts_global_round": reference_round + 1,
                                       "class_count": classes, "config": manifest})
                    torch.save(checkpoint, output / "reference.pt")
                    save_checkpoint(output / "control_start.pt", model, server, global_round=reference_round,
                                    split_sha256=split_hash, malicious_ids=sorted(malicious), config=manifest)
                    break

                previous_server_center = current_server_center
        if reference_round is None:
            completed_warmup = args.max_warmup_rounds
            if args.max_global_rounds is not None:
                completed_warmup = min(completed_warmup, args.max_global_rounds)
            summary = {"status": "reference_not_reached", "warmup_rounds": completed_warmup,
                       "max_global_rounds": args.max_global_rounds,
                       "control_rounds": 0, "osr": None}
            json_write(output / "summary.json", summary)
            return summary

        # Both arms start from EXACTLY the just-captured global/server states.
        # Oct-2 semantics control parameters only; BN/statistical buffers stay at
        # the broadcast global values and are not part of the malicious command.
        parameter_reference = flatten(target_model).clone()
        floating_buffer_reference = flatten_floating_buffers(target_model).clone()
        reference = flatten_control_state(
            target_model, include_floating_buffers=control_floating_buffers
        ).clone()
        clean_model, clean_server = copy.deepcopy(model), copy.deepcopy(server)
        controller = AdaptiveController(reference, cfg) if args.attack_mode not in ("replay", "zero") else None
        if controller and args.target_accuracy is not None and args.accuracy_feedback_gain:
            controller.observe_accuracy(
                selector.checkpoint()["reference_validation_accuracy"], args.target_accuracy,
                gain=args.accuracy_feedback_gain, max_scale=args.accuracy_feedback_max_scale,
            )
        traces = {"control": [], "clean": []}

        effective_control_rounds = args.control_rounds
        if args.max_global_rounds is not None:
            effective_control_rounds = min(
                effective_control_rounds,
                max(0, args.max_global_rounds - reference_round),
            )

        if effective_control_rounds == 0:
            summary = {
                "status": "global_horizon_exhausted",
                "dataset": args.dataset,
                "reference_round": reference_round,
                "max_global_rounds": args.max_global_rounds,
                "reference_validation_accuracy": selector.checkpoint()["reference_validation_accuracy"],
                "reference_test_accuracy": reference_test_accuracy,
                "requested_accuracy": args.target_accuracy,
                "control_rounds": 0,
                "osr": None,
                "elapsed_seconds": time.perf_counter() - start,
            }
            json_write(output / "summary.json", summary)
            return summary

        for control_round in range(1, effective_control_rounds + 1):
            round_number = reference_round + control_round
            chosen = participants(args, round_number)
            active_count = len(set(chosen) & malicious)
            before = flatten_control_state(
                model, include_floating_buffers=control_floating_buffers
            )
            if controller:
                control_delta, diagnostics = controller.propose(before, active=active_count > 0)
            else:
                # Replay is an uncapped checkpoint-submission ablation, NOT ASMC.
                control_delta = (
                    reference - before
                    if active_count and args.attack_mode == "replay"
                    else torch.zeros_like(reference)
                )
                diagnostics = {"submitted_delta_norm": float(control_delta.norm()), "saturated": False}

            parameter_delta, floating_buffer_delta = split_control_delta(
                model, control_delta, include_floating_buffers=control_floating_buffers
            )
            for phase, arm_model, arm_server in (("control", model, server), ("clean", clean_model, clean_server)):
                training = train_round(
                    arm_model, arm_server, dataset, partitions, chosen,
                    malicious if phase == "control" else set(),
                    parameter_delta if phase == "control" else None,
                    args, device, round_number, server_root_ids,
                    malicious_buffer_delta=(floating_buffer_delta if phase == "control" else None),
                )
                if phase == "control" and controller:
                    controller.observe(flatten_control_state(
                        arm_model, include_floating_buffers=control_floating_buffers
                    ))
                validation = evaluate(arm_model, attacker_validation_set, device, args.eval_batch_size)
                if phase == "control" and controller and args.target_accuracy is not None and args.accuracy_feedback_gain:
                    controller.observe_accuracy(
                        validation["accuracy"], args.target_accuracy,
                        gain=args.accuracy_feedback_gain, max_scale=args.accuracy_feedback_max_scale,
                    )
                test = evaluate(arm_model, test_set, device, args.eval_batch_size)
                parameter_distance = float((flatten(arm_model) - parameter_reference).norm())
                current_floating_buffers = flatten_floating_buffers(arm_model)
                floating_buffer_distance = (
                    float((current_floating_buffers - floating_buffer_reference).norm())
                    if current_floating_buffers.numel() else 0.0
                )
                control_state_distance = float((
                    flatten_control_state(
                        arm_model, include_floating_buffers=control_floating_buffers
                    ) - reference
                ).norm())
                row = {"global_round": round_number, "control_round": control_round, "phase": phase,
                       "reference_round": reference_round, "reference_captured": False,
                       "participant_ids": chosen, "malicious_participants": active_count if phase == "control" else 0,
                       "actual_malicious_fraction": active_count / len(chosen) if phase == "control" else 0,
                       "byzantine_bound_exceeded": bool(phase == "control" and args.agr in ("trimmed_mean", "multi_krum", "bulyan", "dnc")
                                                          and active_count > args.byzantine_bound),
                       "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"],
                       "test_accuracy": test["accuracy"], "test_loss": test["loss"],
                       "parameter_distance": parameter_distance,
                       "relative_parameter_distance": (
                           parameter_distance / max(float(parameter_reference.norm()), 1e-12)
                       ),
                       "floating_buffer_distance": floating_buffer_distance,
                       "control_state_distance": control_state_distance,
                       "elapsed_seconds": time.perf_counter() - start, **training,
                       **(diagnostics if phase == "control" else {})}
                logger.write(row)
                traces[phase].append(row)
            control_row = traces["control"][-1]
            print(
                f"control round={control_round} global={round_number} "
                f"malicious={active_count}/{len(chosen)} "
                f"test={control_row['test_accuracy']:.4f} "
                f"clean={traces['clean'][-1]['test_accuracy']:.4f} "
                f"distance={control_row['parameter_distance']:.6f} "
                f"submitted={control_row.get('submitted_delta_norm', 0.0):.6f} "
                f"benign_norm={control_row.get('benign_delta_norm_mean', 0.0):.4f}/"
                f"{control_row.get('benign_delta_norm_max', 0.0):.4f} "
                f"mal_norm={control_row.get('malicious_delta_norm_mean', 0.0):.4f} "
                f"survive={control_row.get('selected_malicious_count', 0)}/{active_count} "
                f"survival={control_row.get('malicious_survival_rate', 0.0):.2f} "
                f"response={control_row.get('response_estimate', 0.0):.6f} "
                f"disturbance={control_row.get('disturbance_norm', 0.0):.6f} "
                f"saturated={control_row.get('saturated', False)}",
                flush=True,
            )

        summary = {
            "status": "completed", "dataset": args.dataset,
            "synthetic_only": args.dataset == "synthetic", "reference_round": reference_round,
            "control_starts_global_round": reference_round + 1,
            "reference_validation_accuracy": selector.checkpoint()["reference_validation_accuracy"],
            "reference_selection_rule": selector.checkpoint().get("selection_rule"),
            "reference_fallback": selector.checkpoint().get("reference_fallback", False),
            "reference_absolute_deviation_pp": selector.checkpoint().get("reference_absolute_deviation_pp"),
            "reference_test_accuracy": reference_test_accuracy,
            "requested_accuracy": args.target_accuracy,
            "control_rounds": effective_control_rounds,
            "configured_control_rounds": args.control_rounds,
            "max_global_rounds": args.max_global_rounds,
            "active_control_rounds": sum(row["malicious_participants"] > 0 for row in traces["control"]),
            "saturated_rounds": sum(row.get("saturated", False) for row in traces["control"]),
            "byzantine_bound_exceeded_rounds": sum(row["byzantine_bound_exceeded"] for row in traces["control"]),
            "reference_maintenance": {phase: metric_summary(trace, reference_test_accuracy, args)
                                      for phase, trace in traces.items()},
            "requested_target": None if args.target_accuracy is None else {
                phase: metric_summary(trace, args.target_accuracy, args) for phase, trace in traces.items()},
            "clean_minus_control_test_accuracy_pp": [clean["test_accuracy"] - control["test_accuracy"]
                                                      for clean, control in zip(traces["clean"], traces["control"])],
            "elapsed_seconds": time.perf_counter() - start,
            "interpretation": (
                "maintenance from zero initial controlled-state error; Tiny-ImageNet controls "
                "parameters plus floating model buffers; OSR is not degradation speed"
                if control_floating_buffers else
                "maintenance from zero initial parameter error; OSR is not degradation speed"
            ),
        }
        json_write(output / "summary.json", summary)
        final_global_round = reference_round + effective_control_rounds
        save_checkpoint(output / "control_final.pt", model, server, controller=controller,
                        global_round=final_global_round, config=manifest)
        save_checkpoint(output / "clean_final.pt", clean_model, clean_server,
                        global_round=final_global_round, config=manifest)
        return summary
    except Exception as error:
        json_write(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                             "elapsed_seconds": time.perf_counter() - start})
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = run(arguments)
    print(json.dumps({"status": result["status"], "output": str(Path(arguments.output).resolve())}), flush=True)
