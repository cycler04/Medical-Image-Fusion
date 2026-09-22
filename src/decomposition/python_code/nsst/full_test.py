import numpy as np
from decomposition.python_code.nsst.nsst_dec import nsst_dec
from decomposition.python_code.nsst.nsst_rec import nsst_rec1
import cv2
import matplotlib.pyplot as plt
from decomposition.python_code.nsst.processing_ultis import stats
from time import time



img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
params = {'dcomp': [3, 2, 1], 'dsize': [16, 16, 16]}
stats(img, "original")

lpfilt = 'maxflat'   # or gaussian if error

# print("Decomposing the image using NSST...")

start = time()
dst, shear_f = nsst_dec(img, params, lpfilt=lpfilt)
# print("Reconstructing the image from NSST coefficients...")
# stats(dst[0], "dst[0] - lowpass")
# count = 1
# for band in dst[1:]:
#     for d in range(band.shape[2]):
#         stats(band[:, :, d], f"dst {count} dir {d+1}")
#     count += 1

end = time()
print(f"NSST decomposition completed in {end - start:.2f} seconds.")

start = time()

rec = nsst_rec1(dst, lpfilt)
stats(rec, "reconstructed")
end = time()
print(f"NSST reconstruction completed in {end - start:.2f} seconds.")
# print("Complete.") 
error = np.mean((img - rec) ** 2)
print(f"Mean Squared Error between original and reconstructed image: {error:.2f}")

psnr = 10 * np.log10(255**2 / np.mean((img - rec) ** 2))
print(f"PSNR between original and reconstructed image: {psnr:.2f} dB")


# visualize the first level of shearlet coefficients
fig, axes = plt.subplots(4, 4, figsize=(12, 4))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[0, 1].imshow(dst[0], cmap='gray')
axes[0, 1].set_title('Lowpass Subband')
axes[0, 1].axis('off')
axes[0, 2].imshow(rec, cmap='gray')
axes[0, 2].set_title('Reconstructed Image')
axes[0, 2].axis('off')

for i in range(4):
    axes[1, i].imshow(shear_f[1][:, :, i], cmap='gray')
    axes[1, i].set_title(f'Shearlet Direction {i+1}')
    axes[1, i].axis('off')

for i in range(4):
    axes[2, i].imshow(dst[1][:, :, i], cmap='gray')
    axes[2, i].set_title(f'Shearlet level 1 Coefficients {i+1}')
    axes[2, i].axis('off')
    
for i in range(4):
    axes[3, i].imshow(dst[2][:, :, i], cmap='gray')
    axes[3, i].set_title(f'Shearlet level 2 Coefficients {i+1}')
    axes[3, i].axis('off')
# Turn off unused axes
# Metrics display on axes[0, 3]
axes[0, 3].axis('off')

metrics_text = (
    f"MSE: {error:.4f}\n"
    f"PSNR: {psnr:.2f} dB"
)

axes[0, 3].text(
    0.05, 0.5, metrics_text,
    fontsize=12,
    verticalalignment='center',
    transform=axes[0, 3].transAxes
)

plt.tight_layout()
plt.show()
