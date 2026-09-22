# Medical-Image-Fusion (MMIF) Unified Framework

A comprehensive framework and benchmark suite for Multimodal Medical Image Fusion (MMIF) and Infrared-Visible Image Fusion (IVIF), integrating deep learning models, classical transform/decomposition methods, unified Python & MATLAB evaluation metrics, and literature analysis tools.

**Google Drive Backup**: [MMIF Drive Resources](https://drive.google.com/drive/folders/1SovvwL8Db34qWVcCdQCUtvadULhgAEiu?usp=sharing)

---

## 📁 Workspace Architecture

```
Medical-Image-Fusion/
│
├── data/                                      # Benchmark datasets and image assets
│   ├── raw/                                   # Source datasets
│   │   ├── Harvard-AANLIB/                    # Harvard Whole Brain Atlas (CT-MRI, PET-MRI, SPECT-MRI)
│   │   ├── ADNI-Clinical/                     # ADNI clinical dataset & paired metadata
│   │   ├── MSRS-MultiSpectral/                # Multi-Spectral Road Scenario dataset (IR-VIS)
│   │   ├── LLVIP-Infrared-Visible/            # Low-Light Visible-Infrared dataset
│   │   └── TNO-NightVision/                   # TNO Night Vision dataset
│   └── samples/                               # Sample image pairs and color-space test assets
│
├── outputs/                                   # Consolidated benchmark outputs and results
│   ├── fused/                                 # Fused output images across 37 DL & classical models
│   ├── extracted_info/                        # Extracted JSON schemas, verified metadata & mappings
│   └── figures/                               # Extracted method diagrams & visual figures
├── models/                                    # Fusion algorithms & benchmark baselines
│   ├── runnable/                              # 22 actively verified runnable deep learning models
│   │   ├── DRMF/, ATDFusion/, CENet/, SwinFusion/, LKC-FUNet/, UUD-Fusion/,
│   │   ├── DeDNet/, Diff-IF/, IGNet/, MMIF-CDDFuse/, CM-CSAMFNet/, SD-Fuse/,
│   │   └── MMIF-INet/, TSFI-Fusion/, DATFuse/, CMMDL/, APGFusion/, PSLPT/, ...
│   ├── classical/                             # Classical decomposition & transform-based methods (MATLAB)
│   │   ├── decomposition/                     # ShearLab3Dv11, NSCT, Wavelet, LP, PCNN toolboxes
│   │   └── methods/                           # CoF-MSMG-PCNN, LP-PCNN, PADCDTNP, FDFusion, MDHU, FRAC-HCM, ...
│   └── reference/                             # External reference implementations & comparative baselines
│       ├── ASFE-Fusion/, MMIF-DDFM/, SeAFusion/, U2Fusion/, EMOST/, FusionGAN/, ...
│
├── metrics/                                   # Unified evaluation suite (Python & MATLAB)
│   ├── python/                                # Python metric calculation modules
│   │   ├── info_metrics.py                    # EN, MI, FMI, SCD, NCIE, TE
│   │   ├── img_metrics.py                     # SF, AG, SD, EI, CC, Qabf, NABF, rSFe
│   │   ├── structure_metrics.py               # SSIM, MS-SSIM, MEF-SSIM
│   │   ├── visual_metrics.py                  # VIF, VIFF, QCV, QCB
│   │   ├── quality_metrics.py                 # MSE, PSNR, Mean Intensity
│   │   └── evaluate_all.py                    # Batch evaluation runner
│   └── matlab/                                # Classical MATLAB metric functions & toolboxes
│       ├── Qabf.m, VIFF_Public.m, ssim_index.m, metricMI.m, etc.
│       ├── VIF/                               # VIF toolbox
│       └── objective_evaluation/              # General evaluation metric toolset
│
├── src/                                       # Core pipeline scripts & tooling
│   ├── data_processing/                       # ADNI inspect/filter, normalization, alignment
│   ├── evaluation/                            # Metric benchmarking execution & visualization
└── requirements.txt                           # Consolidated Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running Evaluation Metrics (Python)
```bash
python metrics/python/evaluate_all.py
```
Or run the metric test suite:
```bash
python metrics/python/AA_test.py
```

### 3. Evaluating with MATLAB
Run `metrics/matlab/AArun_batch_metrics.m` in MATLAB or via command line.
