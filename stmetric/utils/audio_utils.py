from typing import Union, Callable

import librosa
import soundfile as sf
import numpy as np
import torch
import gin
import resampy
from scipy.io import wavfile
import librosa.effects


@gin.configurable
def load_audio_file(audio_file: str, preprocessors: Union[Callable] = []):
    ''' Load audio, applying a series of preprocessing transforms
    Args:
        audio_file: audio file path
        preprocessors: iterable of transforms to be composed
    '''
    sr, audio = wavfile.read(audio_file)

    for processor in preprocessors:
        audio = processor(audio)
    
    return audio, sr


@gin.configurable
def load_numpy(file_path: str, preprocessors: Union[Callable] = [], **kwargs):
    npy = np.load(file_path)
    for processor in preprocessors:
        npy = processor(npy)
    return npy


@gin.configurable
def resample_audio(audio: np.ndarray, original_sr: float, target_sr: float):
    return resampy.resample(audio, original_sr, target_sr)


@gin.configurable
def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val

@gin.configurable
def write_audio(out_path: str, audio: np.ndarray, sr: float = 16000):
    sf.write(out_path, audio, sr, 'PCM_24')