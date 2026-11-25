# ASAP: in-training target capture and adaptive control against robust FL aggregation

Normal federated training runs until the global model first reaches a set accuracy; that
global model is captured as the **reference/target**, and from the next round the attacker
applies adaptive control to hold the global model at the reference — against robust Byzantine
aggregators. Malicious clients only observe the broadcast global model and their own
held-out validation accuracy; the test set is never used for capture or control.

## Structure

```
ASAP_earlystop.py          single entry: warmup -> capture -> control, paired clean baseline,
                           checkpoints, metrics, CLI (AGR switches, datasets, attack modes)
control/
  asap_controller.py       sliding-mode state + Fourier compensation + scalar response estimate
  adaptive_fourier.py      online-learned Fourier series over communication time
  reference_protocol.py    reference capture / selection
fl/
  aggregators.py           all robust AGRs (DeltaAggregator) + Krum scoring
  data.py                  IID partitioning + attacker-owned validation split
  models/Nets.py           MNIST MLP (784 -> 200 -> 10)
```

`ASAP_earlystop.py` is directly runnable (`python ASAP_earlystop.py ...`) and covers every
dataset and AGR through CLI flags.

## Install

Python 3.10+. Install a PyTorch/torchvision build matching your CUDA, then:

```bash
python -m pip install -r requirements.txt
```

Datasets download automatically through torchvision on first run.

## Run

```bash
# CIFAR-10 (AlexNet)
python ASAP_earlystop.py --dataset cifar10 --data-root data/cifar --download \
  --target-accuracy 55 --output runs/cifar55 --num-clients 50 --malicious-fraction 0.1 \
  --local-epochs 2 --batch-size 32 --lr 0.003 --agr multi_krum --device cuda:0

# MNIST (MLP)
python ASAP_earlystop.py --dataset mnist --data-root data \
  --target-accuracy 85 --output runs/mnist85 --num-clients 50 --malicious-fraction 0.2 \
  --local-epochs 1 --batch-size 128 --lr 0.0005 --agr median --device cuda:0

# Tiny-ImageNet (ImageNet-pretrained ResNet50)
python ASAP_earlystop.py --dataset tinyimagenet --data-root data/tiny-imagenet-200 \
  --initial-weights imagenet-v2 --target-accuracy 54 --output runs/tiny54 \
  --num-clients 50 --malicious-fraction 0.2 --local-epochs 1 --batch-size 32 --lr 0.0005 \
  --agr multi_krum --byzantine-bound 10 --multi-krum-m 40 --device cuda:0
```

- `--target-accuracy` is a percentage (55 means 55%, not 0.55).
- `--initial-weights` is one of `none`, `imagenet-v1`, `imagenet-v2` and is required for
  Tiny-ImageNet. `v1`/`v2` load the ImageNet-pretrained ResNet50 (initial pretraining, distinct
  from the in-run captured target).
- `--reference-round N` captures at a fixed round instead of on an accuracy trigger (mutually
  exclusive with `--target-accuracy`).

## Protocol

- Every client submits a parameter delta relative to the current global model; the server adds
  the aggregated delta.
- Reference validation comes only from malicious clients' own local partitions: each holds out
  `--attacker-validation-fraction` (default 0.1), and the attacker evaluates the broadcast global
  on the union of those samples.
- Before capture, all clients (including future attackers) train normally. After capture the
  reference is frozen and the server global keeps updating; control starts the following round.
- 50-client default: the last 5 ids (`45,46,47,48,49`) are malicious. Partial participation
  aggregates only the actually selected clients.

## Aggregation rules (`fl/aggregators.py`)

Supported: `fedavg`, `median`, `trimmed_mean`, `norm_bound`, `multi_krum`, `bulyan`, `fltrust`,
`centered_clip`, `dnc`. Selection/filtering rules are `multi_krum`, `bulyan`, `dnc`
(`FILTERING_RULES`).

- **Median**: coordinate-wise median; with n=50 even, the mean of the 25th and 26th sorted values.
- **Trimmed Mean**: drop the smallest f and largest f per coordinate, average the remaining n-2f.
- **Multi-Krum**: Krum score over the nearest `n-f-2` other updates; average the lowest-score `m`
  (`--multi-krum-m`, default `n-f`).
- **Bulyan**: iterative Krum picks `n-2f` candidates, then a coordinate-wise trimmed mean over the
  nearest `n-4f`.
- **DnC**: random-subspace SVD outlier filtering (`--dnc-subsample-dim`, `--dnc-iters`,
  `--dnc-filter-factor`); the per-round random subspace is seeded by the global round.

`centered_clip` is a streaming residual-clipping special case handled in `ASAP_earlystop.py`.
AGR switches and hyperparameters (`--agr`, `--byzantine-bound`, `--multi-krum-m`, `--clip-norm`,
`--dnc-*`, `--fltrust-root-epochs`, ...) live in the `parser()` of `ASAP_earlystop.py`.

## Controller (`control/`)

Control uses `t = control_step * dt` with an online-learned Fourier series:

```
z(t)       = [1, cos(wt), sin(wt), ..., cos(n*wt), sin(n*wt)]
Phi_hat(t) = W_hat.T @ z(t)
u(t)       = J_hat_inv * (-k*error - eta*sign(sliding) - Phi_hat(t))
W_hat     += dt * gamma * z(t) * sliding.T        (clipped)
```

Initial error, sliding variable and coefficients are zero, so the first malicious delta is zero.
The control delta has a norm cap (`--max-delta-norm`) and per-coefficient bounds; coefficient
learning pauses on saturation or rounds with no malicious submission. `--attack-mode` selects
`adaptive` (default), `smc`, `zero`, or `replay` (submit the frozen reference delta each round).

## Outputs

Each run writes to `--output` (which must not already exist), one CSV row per round:

- `target_model.pt`, `reference.pt` — captured global state plus capture round, validation/test
  accuracy and config.
- `control_start.pt`, `initial.pt`, `control_final.pt`, `clean_final.pt` — phase snapshots.
- `metrics.csv` — per-round accuracy, selected/malicious client ids, parameter distances, control
  and Fourier norms, saturation and response-rejection diagnostics.
- `config.json`, `partitions.json`, `malicious_clients.json`, `attacker_validation.json`.
- `summary.json` — status and the normalized deviation from the requested target (null until OSR
  is reached); `failure.json` on error.
