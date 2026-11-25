# Validation record — 2026-09-25

Changes are based on upstream commit `300b88d` (`ASAP-old version`). No manuscript files were edited.

## Automated checks

29 tests pass with Python 3.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, on Windows CPU:

```bash
python -m unittest discover -s tests -v
```

Coverage includes real AlexNet forward/backward/SGD; snapshot isolation and a plain loadable target state dict; first-crossing and next-round activation; test/validation isolation; no padding of sampled clients; aggregation and buffer policies; actual time-based Fourier harmonics, adaptive coefficient updates, compensation affecting submitted controls, and the FS-disabled ablation. Streaming aggregators are checked against their batch equivalents.

An additional known-plant regression compares FS enabled/disabled under a sinusoidal disturbance. With the small time step and gains specified in `tests/test_fourier.py`, tail tracking RMSE was approximately 0.00024 with FS and 0.0040 without it. This is a numerical implementation check on a known system, not evidence of FL attack performance. Large Euler steps and unsuitable gains can fail; no universal stability claim is made.

Running unrestricted unittest discovery from the root also attempts to import legacy aggregator packages whose `global_utils` dependency was not uploaded. These packages are outside the new entry point. The scoped command above is the intended test command, also used by CI.

## Real-data MNIST smoke run

Used the MNIST data already present in the repository; no dataset download was required:

```bash
python ASAP.py --dataset mnist --data-root data --output runs/mnist_capture_fs_smoke --target-accuracy 50 --max-warmup-rounds 5 --control-rounds 3 --num-clients 10 --local-epochs 1 --batch-size 64 --lr .01 --device cpu
```

- First threshold crossing: global round 1, validation accuracy 89.276184%, test accuracy 90.18%.
- The requested threshold was 50%; the checkpoint is correctly labelled with its actual accuracy, not labelled a 50% model. The default rule is first `>=` crossing, without an overshoot rejection limit.
- `target_model.pt` is saved at round 1; control starts at global round 2.
- Exactly 1 of 10 clients is malicious during the controlled continuation.
- Initial controlled submission norm is zero, consistent with zero initial parameter error and zero adaptive state.
- At control round 3, Fourier coefficient norm is 0.0027972180, Fourier estimate norm is 0.0086521646, and submitted delta norm is 0.4086678922. The estimate is used in the control formula.
- Final controlled test accuracy: 93.47%; matched clean continuation: 93.75%.
- Relative to the actual 90.18% reference test accuracy, final control deviation is 3.29 percentage points. This run does **not** demonstrate successful maintenance within a 2-point band.
- Only 3 control rounds were run; the default 10-round OSR window is incomplete and is correctly reported as null.
- Elapsed CPU time: approximately 136 seconds.

The small original k/eta settings and development FS gain are not calibrated full-experiment settings. The run verifies capture, freezing, continuation, and FS execution; it is not a positive attack-efficacy result. The corresponding unrounded metrics and summary are in `validation/mnist_smoke/`.

## Remaining empirical validation

No full CIFAR-10/Tiny-ImageNet multi-seed benchmark, CUDA validation, large-model memory benchmark, or universal AGR convergence test was performed. The scalar effective-response approximation, Fourier gains/period, discretization step and clipping settings require calibration on development configurations before formal evaluation.
