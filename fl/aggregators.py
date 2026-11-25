"""Training and server components with an explicit delta-submission contract."""
import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


class CifarAlexNet(nn.Module):
    """Same layer geometry/state_dict keys as models/cifar/alexnet.py, logits only."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def flatten(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def floating_buffer_items(model):
    """Floating model buffers that participate in inference (e.g. BN statistics)."""
    return [(name, value) for name, value in model.named_buffers() if value.is_floating_point()]


def flatten_floating_buffers(model):
    items = floating_buffer_items(model)
    if not items:
        return flatten(model).new_zeros(0)
    return torch.cat([value.detach().reshape(-1) for _, value in items])


def flatten_control_state(model, *, include_floating_buffers=False):
    """Controller state: parameters plus optional floating inference buffers."""
    parameters = flatten(model)
    if not include_floating_buffers:
        return parameters
    buffers = flatten_floating_buffers(model)
    return torch.cat([parameters, buffers]) if buffers.numel() else parameters


def split_control_delta(model, vector, *, include_floating_buffers=False):
    """Split a controller delta into parameter and floating-buffer components."""
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("Control delta must be a finite flat vector")
    parameter_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(value.numel() for _, value in floating_buffer_items(model)) if include_floating_buffers else 0
    if vector.numel() != parameter_count + buffer_count:
        raise ValueError("Control delta length mismatch")

    parameter_delta = vector[:parameter_count]
    buffer_delta = {}
    offset = parameter_count
    if include_floating_buffers:
        for name, value in floating_buffer_items(model):
            length = value.numel()
            buffer_delta[name] = vector[offset:offset + length].view_as(value)
            offset += length
    return parameter_delta, buffer_delta


@torch.no_grad()
def load_flat(model, vector):
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Nonfinite aggregated model")
    if vector.ndim != 1 or vector.numel() != sum(p.numel() for p in model.parameters()):
        raise ValueError("Parameter vector length mismatch")
    offset = 0
    for param in model.parameters():
        param.copy_(vector[offset:offset + param.numel()].view_as(param))
        offset += param.numel()


def logits(model, x):
    output = model(x)
    return output[-1] if isinstance(output, (tuple, list)) else output


@torch.no_grad()
def evaluate(model, dataset, device, batch_size):
    was_training = model.training
    model.eval()
    total, correct, loss = 0, 0, 0.0
    try:
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            x, y = x.to(device), y.to(device)
            output = logits(model, x)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Nonfinite evaluation output")
            loss += float(nn.functional.cross_entropy(output, y, reduction="sum"))
            correct += int((output.argmax(1) == y).sum())
            total += y.numel()
    finally:
        model.train(was_training)
    if total == 0:
        raise ValueError("Empty evaluation dataset")
    return {"accuracy": 100.0 * correct / total, "loss": loss / total}


def local_delta(global_model, dataset, indices, *, device, seed, epochs, batch_size, lr, momentum):
    if not len(indices):
        raise ValueError("Empty client dataset")
    # Resetting per (global round, client) pairs minibatches/dropout across arms.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    local = copy.deepcopy(global_model).to(device)
    local.train()
    optimizer = torch.optim.SGD(local.parameters(), lr=lr, momentum=momentum)
    loader = DataLoader(Subset(dataset, list(map(int, indices))), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(logits(local, x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local training loss")
            loss.backward()
            optimizer.step()
    delta = flatten(local) - flatten(global_model)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Nonfinite local update")
    buffers = {name: value.detach().clone() for name, value in local.named_buffers()}
    return delta, buffers


def _pairwise_squared_distances(updates, chunk_size):
    """Compute pairwise squared distances without materializing NxNxD tensors."""
    n, d = updates.shape
    distances = updates.new_zeros((n, n))
    for start in range(0, d, chunk_size):
        block = updates[:, start:start + chunk_size]
        norms = block.square().sum(1)
        distances.add_(norms[:, None] + norms[None, :] - 2 * (block @ block.T))
    return distances.clamp_min_(0)


def _krum_scores(updates, f, chunk_size):
    n = len(updates)
    if n < 2 * f + 3:
        raise ValueError("Krum family requires n >= 2f+3")
    distances = _pairwise_squared_distances(updates, chunk_size)
    distances.fill_diagonal_(float("inf"))
    neighbor_count = n - f - 2
    return distances.topk(neighbor_count, largest=False, dim=1).values.sum(1)


def _coordinate_median_block(block):
    """Deterministic real-valued coordinate median.

    For an even number of clients this averages the two middle values, matching
    the usual statistical median instead of torch.median's lower-middle choice.
    Sorting keeps the CUDA path compatible with deterministic execution.
    """
    ordered = block.sort(dim=0).values
    lower = (block.shape[0] - 1) // 2
    upper = block.shape[0] // 2
    return 0.5 * (ordered[lower] + ordered[upper])


def _coordinate_median(updates, chunk_size):
    out = torch.empty_like(updates[0])
    for start in range(0, updates.shape[1], chunk_size):
        block = updates[:, start:start + chunk_size]
        out[start:start + block.shape[1]] = _coordinate_median_block(block)
    return out


def _trimmed_mean(updates, f, chunk_size):
    n = len(updates)
    if n <= 2 * f:
        raise ValueError("Trimmed mean requires n > 2f")
    out = torch.empty_like(updates[0])
    for start in range(0, updates.shape[1], chunk_size):
        block = updates[:, start:start + chunk_size].sort(0).values
        out[start:start + block.shape[1]] = block[f:n-f].mean(0)
    return out


class DeltaAggregator:
    """Legacy-ASAP aggregation behavior under the new delta-submission interface.

    The original ASAP experiments aggregated flattened local model parameters for
    FedAvg/Median/TrimmedMean/NormBounding/Krum/Bulyan/FLTrust, while CC and DNC
    operated on model updates. Translation-invariant rules can operate directly
    on deltas. NormBounding and legacy FLTrust explicitly reconstruct each local
    model from base_model + delta before applying the original rule.
    """

    SUPPORTED = (
        "fedavg", "median", "trimmed_mean", "norm_bound", "bulyan",
        "multi_krum", "fltrust", "centered_clip", "dnc",
    )
    FILTERING_RULES = {"multi_krum", "bulyan", "dnc"}

    def __init__(
        self, name="fedavg", *, byzantine_bound=0, clip_norm=2.5,
        coordinate_chunk_size=200_000, multi_krum_m=None,
        dnc_subsample_dim=10_000, dnc_iters=1, dnc_filter_factor=1.0,
    ):
        if name not in self.SUPPORTED:
            raise ValueError(f"Unsupported aggregation rule: {name}")
        if type(byzantine_bound) is not int or byzantine_bound < 0:
            raise ValueError("byzantine_bound must be a nonnegative integer")
        if not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if multi_krum_m is not None and (type(multi_krum_m) is not int or multi_krum_m < 1):
            raise ValueError("multi_krum_m must be a positive integer or None")
        if type(dnc_subsample_dim) is not int or dnc_subsample_dim < 1:
            raise ValueError("dnc_subsample_dim must be a positive integer")
        if type(dnc_iters) is not int or dnc_iters < 1:
            raise ValueError("dnc_iters must be a positive integer")
        if type(coordinate_chunk_size) is not int or coordinate_chunk_size < 1:
            raise ValueError("coordinate_chunk_size must be positive")
        if not math.isfinite(dnc_filter_factor) or dnc_filter_factor <= 0:
            raise ValueError("dnc_filter_factor must be positive")
        self.name = name
        self.f = byzantine_bound
        self.clip_norm = clip_norm
        self.coordinate_chunk_size = coordinate_chunk_size
        self.multi_krum_m = multi_krum_m
        self.dnc_subsample_dim = dnc_subsample_dim
        self.dnc_iters = dnc_iters
        self.dnc_filter_factor = dnc_filter_factor
        self.center = None
        self.last_selected = []

    def config_dict(self):
        return {
            "name": self.name,
            "byzantine_bound": self.f,
            "clip_norm": self.clip_norm,
            "coordinate_chunk_size": self.coordinate_chunk_size,
            "multi_krum_m": self.multi_krum_m,
            "dnc_subsample_dim": self.dnc_subsample_dim,
            "dnc_iters": self.dnc_iters,
            "dnc_filter_factor": self.dnc_filter_factor,
            "semantics": "standardized_robust_baselines_2026_10_06",
        }

    @staticmethod
    def _krum_choice(remaining, f, chunk_size):
        """Return the lowest standard Krum-score candidate.

        The self-distance is excluded and each score sums distances to the
        m-f-2 closest *other* candidates in the current set.
        """
        m = remaining.shape[0]
        neighbor_count = m - f - 2
        if neighbor_count <= 0:
            raise ValueError("Not enough remaining clients for Krum score")
        distances = _pairwise_squared_distances(remaining, chunk_size)
        distances.fill_diagonal_(float("inf"))
        scores = distances.topk(neighbor_count, largest=False, dim=1).values.sum(1)
        return torch.argsort(scores, stable=True)[0]

    @torch.no_grad()
    def __call__(self, updates, sample_counts, *, root_update=None, round_seed=0, rng_seed=None, base_model=None):
        if rng_seed is not None:
            round_seed = rng_seed
        if updates.ndim != 2 or updates.shape[0] == 0 or not torch.isfinite(updates).all():
            raise ValueError("Expected a finite nonempty [participants, parameters] matrix")
        n, parameter_count = updates.shape
        counts = updates.new_tensor(sample_counts)
        if counts.shape != (n,) or not torch.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("One positive sample count required per actual participant")
        self.last_selected = list(range(n))

        # Original ASAP FedAvg was an unweighted mean over clients.
        if self.name == "fedavg":
            return updates.mean(0)

        if self.name == "median":
            # Preserve the exact legacy ASAP/PyTorch semantics used by the
            # original experiments: for an even number of clients,
            # torch.median(..., dim=0)[0] returns the lower middle value.
            return torch.median(updates, dim=0)[0]

        if self.name == "trimmed_mean":
            if n <= 2 * self.f:
                raise ValueError("Trimmed mean requires n > 2f")
            ordered = updates.sort(0).values
            return ordered[self.f:n-self.f].mean(0)

        if self.name == "norm_bound":
            if base_model is None:
                raise ValueError("Legacy norm bounding requires the pre-round global model")
            base = base_model.to(device=updates.device, dtype=updates.dtype).reshape(-1)
            if base.numel() != parameter_count:
                raise ValueError("base_model length mismatch")
            local_models = updates + base[None, :]
            per_norm = local_models.norm(dim=1)
            avg_norm = per_norm.mean()
            scales = (avg_norm / per_norm.clamp_min(1e-12)).clamp(max=1.0)
            aggregated_model = (local_models * scales[:, None]).mean(0)
            return aggregated_model - base

        if self.name == "multi_krum":
            if n < 2 * self.f + 3:
                raise ValueError("Krum-family aggregation requires n >= 2f+3")
            # Standard Multi-Krum: compute one Krum score per submitted update,
            # select the m lowest-scoring updates, then average them uniformly.
            # The original Blanchard et al. experiment commonly used m=n-f.
            m_select = n - self.f if self.multi_krum_m is None else self.multi_krum_m
            if not 1 <= m_select <= n - self.f:
                raise ValueError("Multi-Krum requires 1 <= m <= n-f")
            scores = _krum_scores(updates, self.f, self.coordinate_chunk_size)
            selected = torch.argsort(scores, stable=True)[:m_select]
            self.last_selected = [int(i) for i in selected]
            return updates[selected].mean(0)

        if self.name == "bulyan":
            if n < 4 * self.f + 3:
                raise ValueError("Bulyan requires n >= 4f+3")
            selection_size = n - 2 * self.f
            beta = n - 4 * self.f
            remaining = updates.clone()
            original_indices = torch.arange(n, device=updates.device)
            candidates = []
            selected_indices = []
            while len(candidates) < selection_size:
                local_idx = self._krum_choice(remaining, self.f, self.coordinate_chunk_size)
                candidates.append(remaining[local_idx].clone())
                selected_indices.append(int(original_indices[local_idx]))
                mask = torch.ones(remaining.shape[0], dtype=torch.bool, device=updates.device)
                mask[local_idx] = False
                remaining = remaining[mask]
                original_indices = original_indices[mask]
            candidates = torch.stack(candidates)
            self.last_selected = selected_indices
            median = _coordinate_median(candidates, self.coordinate_chunk_size)
            closest = torch.argsort(torch.abs(candidates - median), dim=0)
            column_index = torch.arange(candidates.shape[1], device=candidates.device)[None, :]
            sorted_values = candidates[closest, column_index]
            return torch.mean(sorted_values[:beta], dim=0)

        if self.name == "fltrust":
            # IMPORTANT: this intentionally reproduces the original ASAP code,
            # which used w_locals[0] as the "central" model. This is NOT the
            # canonical FLTrust trusted-root-data protocol.
            if base_model is None:
                raise ValueError("Legacy ASAP FLTrust requires the pre-round global model")
            base = base_model.to(device=updates.device, dtype=updates.dtype).reshape(-1)
            if base.numel() != parameter_count:
                raise ValueError("base_model length mismatch")
            local_models = updates + base[None, :]
            central = local_models[0]
            central_norm = central.norm().clamp_min(1e-12)
            client_norms = local_models.norm(dim=1).clamp_min(1e-12)
            trust = torch.nn.functional.cosine_similarity(
                local_models, central[None, :].expand_as(local_models), dim=1
            ).clamp_min(0)
            norm_scale = central_norm / client_norms
            trust_sum = trust.sum().clamp_min(1e-12)
            aggregated_model = (local_models * (trust * norm_scale)[:, None]).sum(0) / trust_sum
            return aggregated_model - base

        if self.name == "dnc":
            if n <= self.f:
                raise ValueError("DNC requires n > f")
            keep_count = n - int(self.dnc_filter_factor * self.f)
            if keep_count <= 0:
                raise ValueError("DNC filtering removes all participants")
            sample_count = min(self.dnc_subsample_dim, parameter_count)

            # DnC Algorithm 2: repeat random-coordinate spectral filtering and
            # aggregate only clients retained by every repetition (set intersection).
            # The supplied round seed makes control/clean arms use identical
            # random subspaces in the same global round.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(round_seed) % (2**63 - 1))
            benign_sets = []
            for _ in range(self.dnc_iters):
                indices = torch.randperm(parameter_count, generator=generator)[:sample_count]
                sampled = updates[:, indices.to(updates.device)]
                mu = sampled.mean(dim=0)
                centered = sampled - mu
                v = torch.linalg.svd(centered, full_matrices=False)[2][0, :]
                scores = torch.stack([
                    torch.dot(update - mu, v).square()
                    for update in sampled
                ]).detach().cpu().numpy()
                good = scores.argsort()[:keep_count]
                benign_sets.append(set(good.tolist()))

            selected_list = sorted(set.intersection(*benign_sets))
            if not selected_list:
                raise RuntimeError("DNC intersection selected no clients")
            self.last_selected = selected_list
            selected = torch.tensor(selected_list, device=updates.device, dtype=torch.long)
            return updates[selected].mean(0)

        # Original Centered Clipping operated on model deltas and kept its
        # momentum/center across rounds.
        if self.center is None:
            self.center = torch.zeros_like(updates[0])
        residual = updates - self.center
        scales = (self.clip_norm / residual.norm(dim=1).clamp_min(1e-12)).clamp(max=1)
        self.center = self.center + (residual * scales[:, None]).mean(0)
        return self.center.clone()


@torch.no_grad()
def aggregate_buffers(model, client_buffers, sample_counts, *, selected=None):
    """Explicit BN policy: weighted float buffers; maximum integer counters.

    Multi-Krum uses only selected clients. Other defenses apply to parameters;
    their buffers use all participants. Under Tiny-ImageNet joint control,
    malicious clients may submit controlled floating buffers while integer
    counters remain ordinary model state. This is an experimental policy, not
    a claim of robust BN aggregation.
    """
    selected = list(range(len(client_buffers))) if selected is None else selected
    weights = [float(sample_counts[i]) for i in selected]
    total = sum(weights)
    if not selected or total <= 0:
        raise ValueError("No buffer contributors")
    for name, target in model.named_buffers():
        values = [client_buffers[i][name].to(target.device) for i in selected]
        if target.is_floating_point():
            out = sum(v * (w / total) for v, w in zip(values, weights))
            if not torch.isfinite(out).all():
                raise FloatingPointError(f"Nonfinite model buffer: {name}")
            target.copy_(out)
        else:
            target.copy_(torch.stack(values).max(0).values)
