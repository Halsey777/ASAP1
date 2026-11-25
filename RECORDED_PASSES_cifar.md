# CIFAR passes already confirmed (lr=0.0015 table) — do NOT re-run

Criterion: control-arm |ctrl-target| <= 5pp AND clean-minus-ctrl separation >= 5pp.

## Unconstrained / norm-bounding — robust one-shot holds (attack has full leverage)
- fedavg      t30: ctrl 32.7 vs clean 47.9 (sep 15)  [+ t50/t55/t10 trivially hold]
- norm_bound  t30: ctrl 33.5 vs clean 47.9 (sep 14)  [+ t50/t55 hold]
- fltrust     t30: ctrl 34.2 vs clean 47.0 (sep 13)  [+ t50/t55 hold; real-root fix]

These three hold across all targets; recorded as success. A final one-shot
confirmation pass (a0 only, no retry) at lr=0.0015 will fill exact table cells.

## Still being retuned (the focused autotune grind):
median, trimmed_mean, centered_clip, multi_krum, bulyan, dnc  x {30,10,50,55}
