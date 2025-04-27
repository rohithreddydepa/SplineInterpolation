import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_frames(frame1, frame2, num_interpolations):
    x_points = np.array([0, 1])
    times_to_interpolate = np.linspace(0, 1, num_interpolations + 2)[1:-1]
    interpolated_frames = []

    height, width, channels = frame1.shape

    # Stack frames along a new axis for per-pixel time interpolation
    stacked_frames = np.stack((frame1, frame2), axis=0)  # shape (2, height, width, channels)

    for t in times_to_interpolate:
        interpolated_frame = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            # For each pixel (i, j) interpolate across time dimension
            pixel_values = stacked_frames[:, :, :, c]  # shape (2, height, width)

            # Apply cubic spline along time axis (axis=0)
            cs = CubicSpline(x_points, pixel_values, axis=0)
            interpolated_values = cs(t)

            interpolated_frame[:, :, c] = np.clip(interpolated_values, 0, 255)

        interpolated_frames.append(interpolated_frame.astype(np.uint8))

    return interpolated_frames
