import numpy as np

def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, data_range=1.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / m))
