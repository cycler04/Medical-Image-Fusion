from pathlib import Path
import random
from PIL import Image

root_folder = "data/raw/Harvard-AANLIB/PET-MRI"
root_path = Path(root_folder)
    
# Get subfolders (e.g. MRI, PET)
original_folders = [f for f in root_path.iterdir() if f.is_dir() and 'eval' not in f.name.lower() and f.name.lower() != 'mydatasets']

# Discover fused model folders for PET-MRI
fused_root_folder = Path("outputs/fused")
fused_folders = [f / "PET-MRI" for f in fused_root_folder.iterdir() if (f / "PET-MRI").is_dir()]

print(f"Fused folders: {[f.name for f in fused_folders]}")
print(f"Original folders: {[f.name for f in original_folders]}")

import matplotlib.pyplot as plt

# Number of random images to display
num_images = 2

# Get random fused images from first fused folder
fused_images = list(fused_folders[0].glob('*'))
random_fused_images = random.sample(fused_images, min(num_images, len(fused_images)))

# Display each fused image with its corresponding original images and all fused results
fig, axes = plt.subplots(num_images, len(original_folders) + len(fused_folders), figsize=(12, 3 * num_images))

for row, random_fused in enumerate(random_fused_images):
    # Get corresponding images from original folders
    for col, orig_folder in enumerate(original_folders):
        img_path = orig_folder / random_fused.name
        if img_path.exists():
            img = Image.open(img_path)
            axes[row, col].imshow(img, cmap='gray')
            if row == 0:  # Only set title on first row
                axes[row, col].set_title(orig_folder.name, fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
    
    # Add all fused images
    for col, fused_folder in enumerate(fused_folders):
        fused_img_path = fused_folder / random_fused.name
        if fused_img_path.exists():
            fused_img = Image.open(fused_img_path)
            axes[row, len(original_folders) + col].imshow(fused_img, cmap='gray')
            if row == 0:  # Only set title on first row
                axes[row, len(original_folders) + col].set_title(fused_folder.name, fontsize=12, fontweight='bold')
            axes[row, len(original_folders) + col].axis('off')

plt.tight_layout()
plt.show()

