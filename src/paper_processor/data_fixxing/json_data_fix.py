import json


# ---------------------------------------------------
# Fusion-related metrics
# ---------------------------------------------------

IMAGE_FUSION_METRICS = {
    "EN",
    "MI",
    "NMI",
    "FMI",
    "SSIM",
    "MS-SSIM",
    "VIF",
    "VIFF",
    "PSNR",
    "SF",
    "SD",
    "AG",
    "EI",
    "SCD",
    "QAB/F",
    "QCV",
    "QCB",
    "QG",
    "QS",
    "QP",
    "QM",
    "QY",
    "QNCIE",
}


def change_data(old):
    # ---------------------------------------------------
    # Add ID + is_image_fusion
    # ---------------------------------------------------

    new = []

    for i, sample in enumerate(old):

        metrics = sample.get("metadata", {}).get("evaluation_metrics", [])

        # count how many fusion metrics appear
        fusion_metric_count = sum(
            metric in IMAGE_FUSION_METRICS
            for metric in metrics
        )

        # set flag
        sample["metadata"]["is_image_fusion"] = fusion_metric_count >= 3

        new.append(sample)
    
    return new


