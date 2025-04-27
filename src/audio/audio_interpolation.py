import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_audio(audio, factor):
    original_indices = np.arange(len(audio))
    interpolated_indices = np.linspace(0, len(audio) - 1, num=len(audio) * factor)
    spline = CubicSpline(original_indices, audio)
    interpolated_audio = spline(interpolated_indices)
    return interpolated_audio
