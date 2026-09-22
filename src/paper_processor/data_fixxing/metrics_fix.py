import json
import re
from ftfy import fix_text

METRIC_ALIASES = {
    "SSIM": [
        "SSIM",
        "Structural Similarity Index Measure",
        "Structural Similarity Index",
        "Structural Similarity",
        "QSSIM",
        "Structural Similarity-Based Metric (SSIM)",
        "Structure Similarity Index Measure (SSIM)"
    ],

    "MS-SSIM": [
        "MS-SSIM",
        "MS_SSIM",
        "Multi-Scale Structural Similarity",
        "Multi-Scale Structural Similarity Index",
        "Multi-Scale SSIM",
    ],

    "VIF": [
        "VIF",
        "Visual Information Fidelity",
        "Visual Fidelity",
        "VIFP",
        "QVIF",
        "Q_VIF",
    ],

    "VIFF": [
        "VIFF",
        "Visual Information Fidelity for Fusion",
    ],

    "MI": [
        "MI",
        "Mutual Information",
        "QMI",
        "Q_MI",
        "Quantitative Mutual Information",
    ],

    "EN": [
        "EN",
        "Entropy",
        "Information Entropy",
        "IE",
        "H",
        "Q_EN",
        "Entropy Metric (EN)"
    ],

    "SD": [
        "SD",
        "Standard Deviation",
        "STD",
        "Q_SD",
    ],

    "SF": [
        "SF",
        "Spatial Frequency",
        "Q_SF",
    ],

    "AG": [
        "AG",
        "Average Gradient",
        "AVG",
    ],

    "PSNR": [
        "PSNR",
        "Peak Signal-to-Noise Ratio",
    ],

    "SCD": [
        "SCD",
        "Sum of Correlation Differences",
        "structural content difference",
        "structural content dissimilarity",
        "Q_SCD",
        "Sum of Correlations of Differences (SCD)",
        "Sum of Correlation of Differences (SCD)",
        "Sum of the Correlations of Differences (SCD)"
    ],

    "QAB/F": [
        "QAB/F",
        "Qabf",
        "Qab/f",
        "Q_AB/F",
        "Q_ABF",
        "edge-based fusion quality metric",
        "edge information measurement",
        "Quality Assessment of Blended Images for Fusion (QAB/F)",
        "Quality Assessment-Based Metric (Qabf)",
        "Edge-based quality metric QAB/F",
        "gradient-based fusion performance (Qabf)",
        "gradient-based similarity measurement (Qabf)",
        "quality of fusion Qabf",
        "Edge Preservation Values (QAB/F)",
        "HVSQAB/F"
    ],

    "NAB/F": [
        "NAB/F",
        "Nabf"
    ],

    "IoU": [
        "IoU",
        "Intersection over Union",
        "intersection-over-union",
    ],

    "mIoU": [
        "mIoU",
        "mean Intersection over Union",
        "Mean Intersection-over-Union",
    ],

    "AP": [
        "AP",
        "Average Precision",
    ],

    "mAP": [
        "mAP",
        "mean average precision",
        "mAP50",
        "mAP@0.5",
        "mAP@.5",
        "mAP@0.5:0.95",
    ],

    "RMSE": [
        "RMSE",
        "Root Mean Square Error",
    ],

    "MSE": [
        "MSE",
        "Mean Squared Error",
    ],

    "MAE": [
        "MAE",
        "Mean Absolute Error",
    ],
    
    "CC": [
        "CC",
        "Correlation Coefficient",
        "Correlation Coefficient (CC)",
        "CORR",
        "Pearson Correlation Coefficient",
    ],
    
    "NCC": [
        "NCC",
        "normalized cross-correlation",
    ],

    "Precision": [
        "Precision",
        "precision",
        "PPV",
        "mean Precision",
        "mPrecision",
    ],

    "Recall": [
        "Recall",
        "recall",
        "Sensitivity",
        "sensitivity",
        "SEN",
        "mean Recall",
        "mRecall",
    ],

    "Specificity": [
        "Specificity",
        "specificity",
    ],

    "Accuracy": [
        "Accuracy",
        "accuracy",
        "overall accuracy",
    ],

    "Balanced Accuracy": [
        "Balanced Accuracy",
        "balanced accuracy",
    ],

    "Pixel Accuracy": [
        "Pixel Accuracy",
        "PA",
        "mean Pixel Accuracy",
        "Mean Pixel Accuracy",
        "mPA",
        "mAcc",
    ],

    "F1-score": [
        "F1-score",
        "F1 score",
    ],

    "MCC": [
        "MCC",
        "Matthews Correlation Coefficient",
        "Matthews Correlation Coefficient (MCC)",
        "Matthews correlation coefficient",
        "Mathews correlation coefficient",
    ],

    "AUC": [
        "AUC",
        "AUC-ROC",
        "Area Under the Curve",
        "Area Under the Curve (AUC)",
        "Area Under the ROC Curve",
        "Area Under the ROC Curve (AUC)",
    ],

    "QCV": [
        "QCV",
        "Qcv",
        "Q_CV",
        "Chen–Varshney metric",
        "Chen-Varshney metric",
        "Chen-Varsheny Metric",
    ],

    "QCB": [
        "QCB",
        "Qcb",
        "Q_CB",
        "Chen-Blum Metric",
    ],

    "QG": [
        "QG",
        "Q_G",
        "Gradient-Based Metric",
        "gradient-based metric",
    ],

    "QS": [
        "QS",
        "Qs",
        "Q_S",
        "Piella's Metric",
    ],

    "QP": [
        "QP",
        "Qp",
        "Phase Congruency Metric",
        "image feature-based metric using phase consistency",
    ],

    "QM": [
        "QM",
        "Q_M",
        "Multiscale Scheme Metric",
    ],

    "QY": [
        "QY",
        "Qy",
        "Improved Structural Similarity Index",
        "improved structural similarity index",
    ],

    "QNCIE": [
        "QNCIE",
        "NCIE",
        "nonlinear correlation information entropy",
        "Non-linear Correlation Information",
    ],

    "QNICE": [
        "QNICE",
        "Non-linear Correlation Metric",
    ],

    "FSIM": [
        "FSIM",
        "Feature Similarity Index",
    ],

    "FMI": [
        "FMI",
        "Feature Mutual Information",
        "FMI_pixel",
        "FMI_dct",
        "FMI_w",
    ],

    "EI": [
        "EI",
        "Edge Intensity",
        "edge intensity",
    ],

    "IFC": [
        "IFC",
    ],

    "NMI": [
        "NMI",
        "Normalized Mutual Information",
        "normalized mutual information",
        "Normalized Mutual Metric",
    ],

    "MAPE": [
        "MAPE",
    ],

    "NME": [
        "NME",
        "Normalized Mean Error",
    ],

    "NMB": [
        "NMB",
        "Normalized Mean Bias",
    ],

    "R2": [
        "R2",
        "Coefficient of Determination",
        "Coefficient of determination R",
        "R-squared",
        "Adjusted R-squared",
    ],

    "Kappa": [
        "Kappa",
        "Cohen's Kappa",
        "Cohen’s Kappa",
    ],

    "Dice": [
        "Dice",
        "Dice coefficient",
        "Dice score",
        "Dice Similarity Coefficient",
        "Dice Similarity Coefficient (DSC)",
        "DSC",
        "mean Dice coefficient",
        "mDice",
    ],

    "clDice": [
        "clDice",
    ],

    "HD95": [
        "HD95",
        "95% Hausdorff distance",
    ],

    "LPIPS": [
        "LPIPS",
    ],

    "DISTS": [
        "DISTS",
    ],

    "FID": [
        "FID",
        "Fréchet Inception Distance",
    ],

    "NIQE": [
        "NIQE",
        "natural image quality evaluator",
    ],

    "MEF-SSIM": [
        "MEF-SSIM",
        "multi-exposure fusion structural similarity index",
    ],

    "SAM": [
        "SAM",
    ],

    "ERGAS": [
        "ERGAS",
    ],

    "SCC": [
        "SCC",
    ],

    "QNR": [
        "QNR",
    ],

    "EPE": [
        "EPE",
    ],

    "TRE": [
        "TRE",
    ],

    "Da": [
        "Da",
    ],

    "Ds": [
        "Ds",
    ],

    "AP@0.5": [
        "AP50",
        "AP@.5",
        "AP@0.5",
    ],

    "AP@0.7": [
        "AP@0.7",
    ],

    "AP@0.9": [
        "AP@0.9",
    ],

    "mAP@0.75": [
        "mAP@0.75",
    ],

    "mAP50-95": [
        "mAP50-95",
        "mAP@0.5:0.95",
    ],

    "Boundary IoU": [
        "Boundary IoU",
    ],

    "Cosine Similarity": [
        "Cosine Similarity",
    ],

    "EPI": [
        "EPI",
        "edge preservation index",
    ],

    "QTE": [
        "QTE",
        "Tsallis Entropy",
        "Tsallis entropy",
    ],

    "RL": [
        "RL",
        "Ranking Loss",
        "ranking loss",
    ],

    "COV": [
        "Coverage",
        "coverage",
        "COV",
    ],

    "NPV": [
        "NPV",
    ],

    "OR": [
        "Odds Ratio",
        "OR",
    ],

    "CI": [
        "95% Confidence Interval",
        "CI",
    ],

    "P-value": [
        "P-value",
        "p-values",
    ],

    "F-statistic": [
        "F-statistic",
    ],

    "Latency": [
        "Latency",
        "execution latency",
    ],

    "FLOPs": [
        "FLOPs",
        "floating-point operations",
    ],

    "Params": [
        "Params",
        "model parameters",
    ],

    "FPS": [
        "FPS",
    ],

    "Time": [
        "Time",
        "Time (ms)",
    ],

    "LMI": [
        "localized mutual information",
        "LMI",
    ],
    
    "Avg.Rank": [
        "Avg.Rank",
        "Average ranking",
    ],
    
    "Mortality": [
        "Mortality",
        "Mortality rate"
    ],
    
    "TDH": [
        "TDH [%]",
        "TDh [%]"
    ]
}
# ---------------------------------------------------
# Build reverse lookup table
# ---------------------------------------------------

def normalize_text(text):
    text = fix_text(text)

    text = text.lower()

    # remove (...) acronym parts
    text = re.sub(r"\([^)]*\)", "", text)

    # remove separators
    text = re.sub(r"[_\-/–^]", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove spaces
    text = re.sub(r"\s+", "", text)

    return text.strip()


LOOKUP = {}

for canonical, aliases in METRIC_ALIASES.items():
    for alias in aliases:
        LOOKUP[normalize_text(alias)] = canonical


# ---------------------------------------------------
# Normalize one metric
# ---------------------------------------------------

def normalize_metric(metric_name):
    normalized = normalize_text(metric_name)

    return LOOKUP.get(normalized, metric_name)


# ---------------------------------------------------
# Fix metrics inside your JSON
# ---------------------------------------------------

def fix_metrics(data):
    metrics = data["metadata"]["evaluation_metrics"]

    fixed_metrics = []

    for metric in metrics:
        fixed = normalize_metric(metric)

        if fixed not in fixed_metrics:
            fixed_metrics.append(fixed)

    data["metadata"]["evaluation_metrics"] = fixed_metrics

    return data

def fix_all_metrics_in_json(json_data):
    for entry in json_data:
        fix_metrics(entry)
    return json_data

# ---------------------------------------------------
# Example
# ---------------------------------------------------
if __name__ == "__main__":
    with open("docs/extracted_info/all_papers_structured_raw.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    fixed_data = fix_all_metrics_in_json(data)
    with open("docs/extracted_info/all_papers_structured.json", "w", encoding="utf-8") as f:
        json.dump(fixed_data, f)
