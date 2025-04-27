import librosa
import soundfile as sf

def load_audio(path):
    audio, sr = librosa.load(path, sr=None)
    return audio, sr

def save_audio(path, audio, sr):
    sf.write(path, audio, sr)
