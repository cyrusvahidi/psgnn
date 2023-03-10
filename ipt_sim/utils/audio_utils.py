from typing import Union, Callable

import numpy as np
from scipy.io import wavfile


def load_audio_file(audio_file: str, preprocessors: Union[Callable] = []):
    """Load audio, applying a series of preprocessing transforms
    Args:
        audio_file: audio file path
        preprocessors: iterable of transforms to be composed
    """
    sr, audio = wavfile.read(audio_file)

    for processor in preprocessors:
        audio = processor(audio)

    return audio, sr


def load_numpy(file_path: str, preprocessors: Union[Callable] = [], **kwargs):
    npy = np.load(file_path)
    for processor in preprocessors:
        npy = processor(npy)
    return npy


def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val
