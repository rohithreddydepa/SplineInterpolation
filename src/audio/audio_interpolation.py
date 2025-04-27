import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_audio(audio, factor):
    # Normalize audio slightly
    audio = audio / np.max(np.abs(audio))

    original_indices = np.arange(len(audio))
    interpolated_indices = np.linspace(0, len(audio) - 1, num=len(audio) * factor)

    # Use 'natural' boundary condition for smoothness
    spline = CubicSpline(original_indices, audio, bc_type='natural')

    interpolated_audio = spline(interpolated_indices)

    # Clip values between -1 and 1 to avoid any numerical overshoots
    interpolated_audio = np.clip(interpolated_audio, -1.0, 1.0)

    return interpolated_audio
