import numpy as np

def hsv_to_rgb(h, s, v):
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r, g, b = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
    m0, m1, m2 = i == 0, i == 1, i == 2
    m3, m4, m5 = i == 3, i == 4, i == 5
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    return r, g, b
