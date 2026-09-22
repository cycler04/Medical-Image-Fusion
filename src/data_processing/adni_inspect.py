import pandas as pd
import matplotlib.pyplot as plt


mri_data = pd.read_csv("data/raw/ADNI-Clinical/All_Subjects_Key_MRI_27Mar2026.csv")
pet_data = pd.read_csv("data/raw/ADNI-Clinical/All_Subjects_Key_PET_27Mar2026.csv")
# print(f"MRI columns: {mri_data.columns.tolist()}")
# print(f"MRI example: {mri_data.head()}")
# print(f"PET columns: {pet_data.columns.tolist()}")
# print(f"PET example: {pet_data.head()}")

# Merge on both columns to find matching rows
# merged = pd.merge(
#     mri_data,
#     pet_data,
#     on=["subject_id", "image_visit"],
#     suffixes=("_mri", "_pet")
# )
# merged = pd.merge(
#     mri_data,
#     pet_data,
#     on=["subject_id", "image_visit"],
#     suffixes=("_mri", "_pet")
# )

# # Select key columns to inspect, with image_date from both tables side by side
# inspect_cols = [
#     "subject_id",
#     "image_visit",
#     "image_date_mri",
#     "image_date_pet",
# ]

# # # Add any other columns you want to compare (adjust to your actual column names)
# optional_cols = [col for col in ["series_type", "tau_pet", "amyloid_pet", "radiopharmaceutical"] if col in merged.columns]

# merged = merged[inspect_cols + optional_cols]

# print(f"Total MRI records: {len(mri_data)}")
# print(f"Total PET records: {len(pet_data)}")
# print(f"Rows with matching subject_id AND image_visit: {len(merged)}")


# merged.to_csv("merged_mri_pet.csv", index=False)


# Ensure date columns are datetime
mri_data["image_date"] = pd.to_datetime(mri_data["image_date"])
pet_data["image_date"] = pd.to_datetime(pet_data["image_date"])

# Step 1: merge on subject_id only (cartesian product per subject)
merged = pd.merge(
    mri_data,
    pet_data,
    on="subject_id",
    suffixes=("_mri", "_pet")
)

# Step 2: filter where date difference is within 30 days
merged["date_diff_days"] = (merged["image_date_mri"] - merged["image_date_pet"]).dt.days.abs()
merged = merged[merged["date_diff_days"] <= 7]

merged["visit_match"] = merged["image_visit_mri"] == merged["image_visit_pet"]

merged = merged[merged["visit_match"] == True]

# Step 3: inspect key columns
inspect_cols = [
    "subject_id",
    "image_id_mri",
    "image_id_pet",
    "image_visit_mri",
    "image_visit_pet",
    "visit_match",
    "image_date_mri",
    "image_date_pet",
    "date_diff_days",
]

optional_cols = [col for col in ["acquisition_plane", "series_description", "tau_pet", "amyloid_pet", "radiopharmaceutical", "pet_description"] if col in merged.columns]

series_description_counts = merged["series_description"].value_counts()

print(f"Matched rows: {len(merged)}")
# print(f"Acquisition plane distribution:\n{merged['acquisition_plane'].value_counts()}")
# print(f"PET description distribution:\n{merged['pet_description'].value_counts()}")
series_description_counts.to_json("series_description_counts.json", orient="index")
merged = merged[inspect_cols + optional_cols]

merged.to_csv("merged_mri_pet_by_date.csv", index=False)