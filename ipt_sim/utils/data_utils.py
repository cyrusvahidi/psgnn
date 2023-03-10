import os, functools
import pathlib as pl
import math
from typing import Union

import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
from librosa import note_to_midi as note_to_midi


from .audio_utils import normalize_audio


# SOL 0.9 HQ utils
def get_sol_filepaths(sol_dir, filelist=None):
    """Retrieve SOL files from SOL directory
    optionally contained in file list of <instrument>-<mute>-<tech>-<pitch>-<intensity>
    """
    filepaths = [
        os.path.relpath(f, sol_dir)
        for f in glob.glob(f"{sol_dir}/**/*.wav", recursive=True)
        if not filelist or os.path.splitext(os.path.basename(f))[0] in filelist
    ]
    return filepaths


def get_sol_filepath(sol_dir, fname, ext=".wav"):
    filepath = glob.glob(f"{sol_dir}/**/{fname}{ext}", recursive=True)[0]
    return os.path.relpath(filepath, sol_dir)


def get_sol_instrument(f):
    return f.split("-")[0].split("+")[0]


def get_unique_instruments(filelist):
    """returns list of unique instruments (without mutes) from a filename list"""
    instruments = set([get_sol_instrument(f) for f in filelist])
    return instruments


def get_sol_instrument_mute(filename):
    return filename.split("-")[0]


def get_sol_technique(filename):
    tokens = filename.split("-")[1:]

    idx = -1
    for i, t in enumerate(tokens):
        if t.isupper() or t == "unp":
            idx = i
            if t == "unp":
                idx = i + 1
            break

    return "-".join(tokens[:idx])


def get_sol_pitch_dynamics(filename):
    filename = os.path.splitext(os.path.basename(filename))[0]
    tokens = filename.split("-")[1:]

    idx = -1
    for i, t in enumerate(tokens):
        if t.isupper() or t == "unp":
            pitch = t if "+" not in t else t[:-1]
            pitch = pitch if "unp" not in pitch else False
            if len(tokens) > i + 1:
                if tokens[i + 1].isupper():  # next token is another pitch
                    return pitch, tokens[i + 2]
                else:
                    return pitch, tokens[i + 1]
            else:
                return pitch, None
    return False, False


def get_sol_IMT(filename):
    im = get_sol_instrument_mute(filename)
    t = get_sol_technique(filename)
    return "-".join([im, t])


def get_sol_files(df, data_dir):
    """Get SOL files from dataframe contained in directory of numpy segments
    Args:
        df: orchidea data frame
        data_dir: directory of segmented numpy files
    """
    # get the audio file names for this split without ext
    audio_fnames = list(os.path.basename(os.path.splitext(f)[0]) for f in df.Path)
    # get the segmented numpy file names from the data dir
    np_file_list = os.listdir(data_dir)
    # get all the numpy file paths for this data split
    files = [
        os.path.join(data_dir, np_f)
        for audio_f in audio_fnames
        for np_f in np_file_list
        if np_f.startswith(audio_f)
    ]
    return files


def create_numpy_files(
    arrays: Union[np.ndarray],
    fnames: Union[str],
    output_dir: str,
    segment_length: float = None,
):
    """Save numpy arrays to files
    if segment_length specified, each array is sliced
    segment_length: number of samples
    """
    for arr, f in tqdm(zip(arrays, fnames)):
        f = os.path.splitext(f)[0]  # no extension
        if not segment_length:
            np.save(os.path.join(output_dir, f + ".npy"), arr)
        else:
            n_segments = math.ceil(len(arr) / segment_length)
            arr = pad_or_trim_along_axis(arr, n_segments * segment_length)
            for i in range(n_segments):
                segment = arr[i * segment_length : (i + 1) * segment_length]
                np.save(os.path.join(output_dir, f + f"_{i}" + ".npy"), segment)


def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)


def pad_or_trim_along_axis(arr: np.ndarray, output_length: int, axis=-1):
    if arr.shape[axis] < output_length:
        n_pad = output_length - arr.shape[axis]

        n_dims_end = len(arr.shape) - axis - 1 if axis >= 0 else 0
        n_dims_end *= n_dims_end > 0

        padding = [(0, 0)] * axis + [(0, n_pad)] + [(0, 0)] * n_dims_end

        return np.pad(arr, padding)
    else:
        return np.take_along_axis(arr, np.arange(0, output_length, 1), axis)


def prep_audio(audio, input_sr=44100, target_sr=16000, in_shape=16000):
    audio = normalize_audio(audio)
    if in_shape:
        audio = pad_or_trim_along_axis(audio, in_shape)
    return audio


def get_fname(fpath):
    return os.path.splitext(os.path.basename(fpath))[0]


def replace_ext(fpath, new_ext=".npy"):
    pre, _ = os.path.splitext(fpath)
    return pre + new_ext


def load_and_save_dissim_mat(
    sim_csv="jasmp/sim.csv",
    extended_csv: str = "jasmp/extended_filelist.csv",
    out_path: str = "jasmp",
):
    ext_files = pd.read_csv(extended_csv, index_col=0).to_dict(orient="records")
    seed_dissim = pd.read_csv(sim_csv, header=None).to_numpy(dtype=np.float32)

    n_seed, n_ext = len(seed_dissim), len(ext_files)
    ndim = n_seed + n_ext
    dissim_mat = np.zeros((ndim, ndim), dtype=np.float32)
    for i in range(ndim):
        for j in range(ndim - 1, i, -1):
            seed_id_i = i if i < n_seed else ext_files[i - n_seed]["seed_id"]
            seed_id_j = j if j < n_seed else ext_files[j - n_seed]["seed_id"]
            dissim_mat[i, j] = seed_dissim[seed_id_i, seed_id_j]
    dissim_mat = dissim_mat + dissim_mat.T
    np.save(os.path.join(out_path, "dissim_mat"), dissim_mat)


def df_attach_imt_pitch_dynamics(
    meta_path="/homes/cv300/Documents/timbre-metric/jasmp/",
    seed_filelist="seed_filelist.csv",
):
    SEED_FILELIST = os.path.join(meta_path, "seed_filelist.csv")
    df = pd.read_csv(SEED_FILELIST, index_col=0)
    df["IMT"] = [get_sol_IMT(get_fname(f)) for f in list(df["fpath"])]
    df["pitch"] = [get_sol_pitch_dynamics(get_fname(f))[0] for f in list(df["fpath"])]
    df["dynamics"] = [
        get_sol_pitch_dynamics(get_fname(f))[1] for f in list(df["fpath"])
    ]
    return df


def get_extended_files(
    SOL_PATH="/import/c4dm-datasets/SOL_0.9_HQ/",
    meta_path="/homes/cv300/Documents/timbre-metric/jasmp/",
    seed_filelist="seed_filelist.csv",
    filter_fn=lambda _, x: x,
    semitones=None,
):
    SEED_FILELIST = os.path.join(meta_path, seed_filelist)
    INST_TO_FAMILY = os.path.join(meta_path, "instrument_to_family.csv")
    inst_to_fam = (
        pd.read_csv(INST_TO_FAMILY, index_col=0)
        .set_index("instrument")
        .to_dict()["family"]
    )
    df = pd.read_csv(SEED_FILELIST, index_col=0)

    seed_imts = {get_sol_IMT(f): i for i, f in enumerate(list(df["fname"]))}
    ext_files = [
        (seed_imts[get_sol_IMT(os.path.basename(f))], f)
        for f in get_sol_filepaths(SOL_PATH)
        if get_sol_IMT(os.path.basename(f)) in seed_imts.keys()
    ]
    ext_files = list(
        filter(functools.partial(filter_fn, df, semitones=semitones), ext_files)
    )

    df_ext_tmp = pd.DataFrame(ext_files, columns=["seed_id", "filepath"])
    df_ext_tmp = df_ext_tmp[df_ext_tmp["seed_id"].isin(df.index)]
    df_ext_tmp["instrument"] = [
        get_sol_instrument(get_fname(f)) for f in list(df_ext_tmp["filepath"])
    ]
    df_ext_tmp["IMT"] = [
        get_sol_IMT(get_fname(f)) for f in list(df_ext_tmp["filepath"])
    ]
    df_ext_tmp["pitch"] = [
        get_sol_pitch_dynamics(get_fname(f))[0] for f in list(df_ext_tmp["filepath"])
    ]
    df_ext_tmp["dynamics"] = [
        get_sol_pitch_dynamics(get_fname(f))[1] for f in list(df_ext_tmp["filepath"])
    ]
    df_ext_tmp["instrument family"] = [
        inst_to_fam[pl.Path(f).parts[0]] for f in list(df_ext_tmp["filepath"])
    ]
    return df_ext_tmp


def df_attach_meta(df, meta_path="/homes/cv300/Documents/timbre-metric/jasmp/"):
    INST_TO_FAMILY = os.path.join(meta_path, "instrument_to_family.csv")
    inst_to_fam = (
        pd.read_csv(INST_TO_FAMILY, index_col=0)
        .set_index("instrument")
        .to_dict()["family"]
    )
    df["instrument"] = [get_sol_instrument(get_fname(f)) for f in list(df["filepath"])]
    df["IMT"] = [get_sol_IMT(get_fname(f)) for f in list(df["filepath"])]
    df["pitch"] = [
        get_sol_pitch_dynamics(get_fname(f))[0] for f in list(df["filepath"])
    ]
    df["dynamics"] = [
        get_sol_pitch_dynamics(get_fname(f))[1] for f in list(df["filepath"])
    ]
    df["instrument family"] = [
        inst_to_fam[pl.Path(f).parts[0]] for f in list(df["filepath"])
    ]
    return df


def filter_dynamics(df, f, **kwargs):
    seed_id, fname = f
    seed_pitch = get_sol_pitch_dynamics(df["fname"][seed_id])[0]
    ext_pitch, _ = get_sol_pitch_dynamics(fname)
    return seed_pitch == ext_pitch


def filter_pitch(df, f, semitones=2):
    seed_id, fname = f
    seed_pitch = get_sol_pitch_dynamics(df["fname"][seed_id])[0]
    ext_pitch = get_sol_pitch_dynamics(fname)[0]
    if not ext_pitch:
        return True
    ext_midi = note_to_midi(ext_pitch)
    if not seed_pitch:
        return True
    seed_midi = note_to_midi(seed_pitch)
    return ext_midi <= seed_midi + semitones and ext_midi >= seed_midi - semitones
