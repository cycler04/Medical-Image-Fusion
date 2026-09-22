"""
This is a reference implementation of Bilateral/Guided filter. 
Since the code use tools and short, it just a notes for future used, not a module
"""
from turtle import lt

import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_detail(detail):
    vis = detail + 128  # center around gray
    return np.clip(vis, 0, 255).astype(np.uint8)

# img = cv2.imread('data/raw/Harvard-AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png', cv2.IMREAD_GRAYSCALE)

img = np.array([[10, 30, 200, 200],
                [10, 30, 200, 200],
                [10, 100, 200, 200],
                [10, 100, 150, 100]], dtype=np.float32)


# img2 = img/2 + np.random.normal(0, 5, img.shape).astype(np.float32)
img2 = img/2

print("Input Image:\n", img2)
# img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

# d = kernel size
# sigmaColor = intensity similarity (σ_r)
# sigmaSpace = spatial distance (σ_s)
bilat_filtered = cv2.bilateralFilter(img, d=3, sigmaColor=100, sigmaSpace=1, borderType=cv2.BORDER_REPLICATE)

# guide image = same as input (common case)
guided_filtered = guidedFilter(
    guide=img2,
    src=img2,
    radius=3,
    eps=0.01
)

img_f = img.astype(np.float32)

bilat_base = bilat_filtered.astype(np.float32)
guided_base = guided_filtered.astype(np.float32)

bilat_detail = img_f - bilat_base
guided_detail = img_f - guided_base

print("Guided Base:\n", guided_base)

def visualize_edge_map(img, ax, title):
    edge_map = np.abs(img)
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

    h, w = edge_map.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    ax.plot_surface(X, Y, edge_map)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Edge")
    
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

visualize_edge_map(img, ax1, "Original")
visualize_edge_map(bilat_base, ax2, "Bilateral Base")

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

visualize_edge_map(img2, ax1, "Original")
visualize_edge_map(img, ax2, "Guided Image")
visualize_edge_map(guided_base, ax3, "Guided Base")

plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original')
# axes[0].axis('off')

# axes[1].imshow(bilat_base, cmap='gray')
# axes[1].set_title('Bilateral Base')
# axes[1].axis('off')

# axes[2].imshow(visualize_detail(bilat_detail), cmap='gray')
# axes[2].set_title('Bilateral Detail')
# axes[2].axis('off')

# axes[3].imshow(guided_base, cmap='gray')
# axes[3].set_title('Guided Base')
# axes[3].axis('off')

# axes[4].imshow(visualize_detail(guided_detail), cmap='gray')
# axes[4].set_title('Guided Detail')
# axes[4].axis('off')

# plt.tight_layout()
# plt.show()