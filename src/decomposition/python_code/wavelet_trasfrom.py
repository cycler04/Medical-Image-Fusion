import numpy as np
from scipy.signal import convolve2d
import pywt

from PIL import Image
# image1 = "Harvard/CT-MRI/CT/2004.png"  # 2D float64 array, dims divisible by 2^3 = 8
# image2 = "Harvard/CT-MRI/MRI/2004.png"

# img1 = Image.open(image1).convert('L')
# img2 = Image.open(image2).convert('L')

A = np.array([
    [80, 20, 85, 90],
    [75, 25, 78, 82],
    [80, 22, 88, 92],
    [78, 24, 80, 85]
], dtype=np.float64)
   

cA1, (cH1, cV1, cD1) = pywt.swt2(A, 'db1', 1)[0]
print(cA1, cA1.shape)

cA2, (cH2, cV2, cD2) = pywt.dwt2(A, 'db1')
print(cA2, cA2.shape)
