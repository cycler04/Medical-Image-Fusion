# Training Safety & Stability Guidelines

- **Explosion Detection**: Always run training with `--stop_on_explosion` enabled when available.
- **Immediate Halt on NaNs**: If loss or gradient norms diverge or become non-finite (`NaN`/`Inf`), terminate the run immediately and adjust loss weights or learning rates.
- **Color Mode Flag**: In INet exploration runs, verify the `--color_mode` parameter (`none`, `conditional`, `full`). Using `none` zeroes out chromatic loss (`ks=0.0`).
