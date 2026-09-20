import numpy as np
from .convolution import correlate2d

_SOBEL_X = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
_SOBEL_Y = np.array([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])

def mySobelFilter(I):
    Gx = correlate2d(I, _SOBEL_X, boundary="edge")
    Gy = correlate2d(I, _SOBEL_Y, boundary="edge")
    mag = np.sqrt(Gx * Gx + Gy * Gy)
    max_mag = np.max(mag)
    mag_norm = mag / max_mag if max_mag > 0 else mag
    ori = np.arctan2(Gy, Gx)
    return mag_norm, ori
