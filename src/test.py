from scipy.io import loadmat
import numpy as np

data = loadmat("C:\\Program Files\\MATLAB\\R2026a\\toolbox\\wavelet\\core\\wavelet\\dmey.mat")

print(data.keys())
W = data['dmey']
Lo_D = (np.sqrt(2) * W / np.sum(W))
Lo_D[0] = Lo_D[0][::-1]
print(Lo_D)

