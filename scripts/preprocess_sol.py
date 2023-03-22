import os

import fire, numpy as np, torch, pandas as pd, librosa.feature
from kymatio.torch import TimeFrequencyScattering, Scattering1D
from tqdm import tqdm
import openl3

from ipt_sim.utils import (
    load_audio_file,
    make_directory,
    pad_or_trim_along_axis,
    torch_float32,
    normalize_audio,
    torch_float32,
    get_sol_filepaths,
    get_fname,
)


class SOLExtractor:
    def __init__(
        self,
        sol_dir,
        output_dir,
        in_shape,
        input_sr=44100,
        target_sr=44100,
        seed_csv: str = "jasmp/seed_filelist.csv",
        ext_csv: str = "jasmp/extended_pitch_F4.csv",
    ):
        self.sol_dir = sol_dir
        self.output_dir = output_dir
        self.in_shape = in_shape
        self.input_sr = input_sr
        self.target_sr = target_sr

        make_directory(self.output_dir)

        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.load_files()

    def get_audio(self, filepath):
        audio, sr = load_audio_file(filepath, preprocessors=[normalize_audio])
        audio = pad_or_trim_along_axis(normalize_audio(audio), self.in_shape)
        audio = torch_float32(audio)
        return audio

    def load_files(self):
        df = pd.read_csv(self.seed_csv, index_col=0)

        seed_files = list(df["fname"])
        self.seed_files = get_sol_filepaths(self.sol_dir, seed_files)
        self.ext_files = list(pd.read_csv(self.ext_csv, index_col=0)["filepath"])
        self.all_files = self.seed_files + self.ext_files

    def save_stats(self):
        stats_path = os.path.join(self.output_dir, "stats")
        make_directory(stats_path)
        np.save(os.path.join(stats_path, "mu"), self.mu.cpu().numpy())
        np.save(os.path.join(stats_path, "var"), self.var.cpu().numpy())

        if hasattr(self, "mean"):
            np.save(os.path.join(stats_path, "mean"), self.mean.cpu().numpy())


class JTFSExtractorSOL(SOLExtractor):
    def __init__(
        self,
        sol_dir,
        jtfs_kwargs={
            "shape": 2**16,
            "Q": (12, 1),
            "J": 8,
            "J_fr": 6,
            "Q_fr": 1,
            "T": 2**16,
        },
        F_octaves=2,
        input_sr=44100,
        target_sr=44100,
        c=1e-3,
    ):
        super().__init__(
            sol_dir,
            os.path.join(sol_dir, "jtfs"),
            jtfs_kwargs["shape"],
            input_sr,
            target_sr,
        )

        # jtfs_kwargs["T"] = 2 ** int(np.log2(jtfs_kwargs["shape"]))
        self.jtfs = TimeFrequencyScattering(
            **jtfs_kwargs, F=jtfs_kwargs["Q"][0] * F_octaves, format="time"
        ).cuda()

        self.samples = []
        self.fnames = []
        self.c = c

    def save_features(self):
        self.mean = torch.stack(self.samples).mean(dim=0)
        # apply log compression and median normalization
        for idx, s in enumerate(self.samples):
            self.samples[idx] = s
            # self.samples[idx] = torch.log1p(s / (self.c * self.mean))
            np.save(self.fnames[idx], s.cpu().numpy())

        samples = torch.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        self.mu = samples.mean(dim=0)
        self.var = samples.var(dim=0)

    def extract(self):
        print("Extracting TRAIN set JTFS coefficients and stats ...")
        for idx, filepath in enumerate(tqdm(self.all_files)):
            dirname, fname = os.path.dirname(filepath), get_fname(filepath)
            fulldir = os.path.join(self.output_dir, dirname)
            make_directory(fulldir)
            audio = self.get_audio(os.path.join(self.sol_dir, filepath))
            coefs = self.jtfs(audio.cuda())[1:].mean(dim=-1)
            self.samples.append(coefs)
            self.fnames.append(os.path.join(fulldir, fname))


class Scat1DExtractorSOL(SOLExtractor):
    def __init__(
        self,
        sol_dir,
        scat1d_kwargs={
            "shape": 44100,
            "Q": (12, 1),
            "J": 8,
            "T": 44100,
        },
        input_sr=44100,
        target_sr=44100,
        c=1e-3,
        max_order=2
    ):
        super().__init__(
            sol_dir,
            os.path.join(sol_dir, f"scat1d_{max_order}"),
            scat1d_kwargs["shape"],
            input_sr,
            target_sr,
        )

        scat1d_kwargs["T"] = 2 ** int(np.log2(scat1d_kwargs["shape"]))
        self.scat1d = Scattering1D(**scat1d_kwargs, max_order=max_order).cuda()
        meta = self.scat1d.meta()
        if max_order == 2:
            order1 = np.where(meta["order"] == 1)
            order2 = np.where(meta["order"] == 2)
            self.idxs = np.concatenate([order1[0], order2[0]])
        else:
            self.idxs = np.where(meta["order"] == 1)
        # self.idxs = order1[0]

        self.samples = []
        self.fnames = []
        self.c = c

    def save_features(self):
        self.mean = torch.stack(self.samples).mean(dim=0)
        # apply log compression and median normalization
        for idx, s in enumerate(self.samples):
            self.samples[idx] = s
            # self.samples[idx] = torch.log1p(s / (self.c * self.mean))
            np.save(self.fnames[idx], s.cpu().numpy())

        samples = torch.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        self.mu = samples.mean(dim=0)
        self.var = samples.var(dim=0)

    def extract(self):
        print("Extracting TRAIN set SCAT1D coefficients and stats ...")
        for idx, filepath in enumerate(tqdm(self.all_files)):
            dirname, fname = os.path.dirname(filepath), get_fname(filepath)
            fulldir = os.path.join(self.output_dir, dirname)
            make_directory(fulldir)
            audio = self.get_audio(os.path.join(self.sol_dir, filepath))
            coefs = self.scat1d(audio.cuda())[self.idxs, 0]
            self.samples.append(coefs)
            self.fnames.append(os.path.join(fulldir, fname))


class OpenL3SOLExtractor(SOLExtractor):
    def __init__(self, sol_dir, input_sr=44100, target_sr=44100):

        super().__init__(
            sol_dir, os.path.join(sol_dir, "openl3"), 44100, input_sr, target_sr
        )

        self.samples = []

    def extract(self):
        for idx, filepath in enumerate(tqdm(self.all_files)):
            dirname, fname = os.path.dirname(filepath), get_fname(filepath)
            fulldir = os.path.join(self.output_dir, dirname)
            make_directory(fulldir)
            if not os.path.exists(os.path.join(fulldir, fname) + ".npy"):
                audio = self.get_audio(os.path.join(self.sol_dir, filepath)).numpy()
                embeddings, timestamps = openl3.get_audio_embedding(
                    audio, 44100, embedding_size=512, frontend='kapre'
                )

                embeddings_mean = embeddings.mean(axis=0)
                self.samples.append(embeddings_mean)
                np.save(os.path.join(fulldir, fname), embeddings_mean)

    def save_stats(self):
        # reshape to (n_examples x timesteps, embedding_size)
        samples = np.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        mu = samples.mean(axis=0)
        var = samples.var(axis=0)
        stats_path = os.path.join(self.output_dir, "stats")
        make_directory(stats_path)
        np.save(os.path.join(stats_path, "mu"), mu)
        np.save(os.path.join(stats_path, "var"), var)


class MFCCSOLExtractor(SOLExtractor):
    def __init__(self, sol_dir, input_sr=44100, target_sr=44100):

        super().__init__(
            sol_dir, os.path.join(sol_dir, "mfcc"), 44100, input_sr, target_sr
        )

        self.samples = []

    def extract(self):
        for idx, filepath in enumerate(tqdm(self.all_files)):
            dirname, fname = os.path.dirname(filepath), get_fname(filepath)
            fulldir = os.path.join(self.output_dir, dirname)
            make_directory(fulldir)
            audio = self.get_audio(os.path.join(self.sol_dir, filepath)).numpy()

            S = librosa.feature.melspectrogram(y=audio, sr=self.input_sr, n_mels=128)
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)

            mfccs = mfccs.mean(axis=-1)
            self.samples.append(mfccs)
            np.save(os.path.join(fulldir, fname), mfccs)

    def save_stats(self):
        # reshape to (n_examples x timesteps, embedding_size)
        samples = np.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        mu = samples.mean(axis=0)
        var = samples.var(axis=0)
        stats_path = os.path.join(self.output_dir, "stats")
        make_directory(stats_path)
        np.save(os.path.join(stats_path, "mu"), mu)
        np.save(os.path.join(stats_path, "var"), var)



import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


class YAMNETSOLExtractor(SOLExtractor):
    def __init__(self, sol_dir, input_sr=44100, target_sr=16000):

        super().__init__(
            sol_dir, os.path.join(sol_dir, "yamnet"), 44100, input_sr, target_sr
        )

        self.samples = []

        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(yamnet_model_handle)


    def extract(self):
        for idx, filepath in enumerate(tqdm(self.all_files)):
            dirname, fname = os.path.dirname(filepath), get_fname(filepath)
            fulldir = os.path.join(self.output_dir, dirname)
            make_directory(fulldir)
            if not os.path.exists(os.path.join(fulldir, fname) + ".npy"):
                audio = load_wav_16k_mono(os.path.join(self.sol_dir, filepath))
                _, embeddings, _ = self.yamnet_model(audio)
                import pdb; pdb.set_trace()
                embeddings_mean = embeddings.mean(axis=0)
                self.samples.append(embeddings_mean)
                np.save(os.path.join(fulldir, fname), embeddings_mean)

    def save_stats(self):
        # reshape to (n_examples x timesteps, embedding_size)
        samples = np.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        mu = samples.mean(axis=0)
        var = samples.var(axis=0)
        stats_path = os.path.join(self.output_dir, "stats")
        make_directory(stats_path)
        np.save(os.path.join(stats_path, "mu"), mu)
        np.save(os.path.join(stats_path, "var"), var)


def extract_jtfs_stats(
    sol_dir="/import/c4dm-datasets/SOL_0.9_HQ/",
    jtfs_kwargs={
        "shape": 2**16,
        "Q": (8, 1),
        "Q_fr": 1,
        "J": 12,
        "J_fr": 4,
    },
    F_octaves=1,
    feature="yamnet"
):
    """Extract training set statistics from Joint Time-Frequency Scattering
    Coefficients.
    Args:
    data_dir   -- path to segmented numpy audio files
    output_dir -- output path for the statistics
    gin_config -- feature extraction configuration
    """
    if feature == "jtfs":
        extractor = JTFSExtractorSOL(sol_dir, jtfs_kwargs, F_octaves=F_octaves)
    elif feature == "openl3":
        extractor = OpenL3SOLExtractor(sol_dir)
    elif feature == "yamnet":
        extractor = YAMNETSOLExtractor(sol_dir)
    elif "scat1d" in feature:
        extractor = Scat1DExtractorSOL(sol_dir, max_order=int(feature.split("_")[-1]))
    elif feature == "mfcc":
        extractor = MFCCSOLExtractor(sol_dir)
    extractor.extract()
    extractor.save_features()
    extractor.save_stats()


def main():
    fire.Fire(extract_jtfs_stats)


if __name__ == "__main__":
    main()