# Benchmark and Reporting Guidelines

- **Baseline Comparison**: When compiling benchmark tables in `docs/reports/`, always include the original published paper results (e.g., original INet paper baseline) alongside the current experiment runs for direct comparison.
- **Run Selection**: Verify which run configurations are active before compiling summaries; omit deprecated exploratory configurations (such as unnormalized 2000-epoch runs) unless explicitly requested.
- **Output Format**: Format metric tables with clear modality breakdowns (MRI-PET, MRI-SPECT, MRI-CT) and highlight top-performing checkpoints across SSIM, CC, SCD, MS-SSIM, and Nabf.
