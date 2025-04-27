import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_frames(frame1, frame2, num_interpolations):
    x_orig = [0, 1]
    x_interpolated = np.linspace(0, 1, num_interpolations + 2)[1:-1]
    interpolated = []

    for t in x_interpolated:
        frame = np.zeros_like(frame1)
        for channel in range(frame1.shape[2]):
            spline = CubicSpline(x_orig, [frame1[:, :, channel], frame2[:, :, channel]], axis=0)
            frame[:, :, channel] = np.clip(spline(t), 0, 255)
        interpolated.append(frame.astype(np.uint8))
    return interpolated
