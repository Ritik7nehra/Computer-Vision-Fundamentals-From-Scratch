import numpy as np
from .convolution import correlate2d, extract_sliding_windows

def my_gaussian_kernel(k, s):
    if k % 2 == 0:
        raise ValueError("kernel size k must be odd, got %d" % k)
    half = k // 2
    kernel = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            x, y = i - half, j - half
            kernel[i, j] = np.exp(-(x * x + y * y) / (2.0 * s * s))
    kernel /= np.sum(kernel)
    return kernel

def myGaussianSmoothing(I, k, s):
    return correlate2d(I, my_gaussian_kernel(k, s), boundary="edge")

def myMedianFilter(I, k):
    if k % 2 == 0:
        raise ValueError("kernel size k must be odd, got %d" % k)
    stack = extract_sliding_windows(I, k, boundary="edge")
    values = stack.copy()
    n = k * k
    for i in range(1, n):
        key = values[i].copy()
        j = i - 1
        while j >= 0:
            mask = values[j] > key
            if not np.any(mask):
                break
            values[j + 1][mask] = values[j][mask]
            values[j][mask] = key[mask]
            j -= 1
    return values[n // 2]
