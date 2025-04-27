import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_audio(original, enhanced):
    min_len = min(len(original), len(enhanced))
    mse = mean_squared_error(original[:min_len], enhanced[:min_len])
    snr = 10 * np.log10(np.sum(original[:min_len]**2) / np.sum((original[:min_len] - enhanced[:min_len])**2))
    return mse, snr
