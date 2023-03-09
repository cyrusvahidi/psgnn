import math, os

import gin, fire, numpy as np, torch, kymatio
from kymatio.torch import Scattering1D
from tqdm import tqdm

import timbremetrics 

from stmetric.utils import (
    gin_register_and_parse, 
    make_directory, 
    pad_or_trim_along_axis, 
    normalize_audio, 
    resample_audio,
    torch_float32)

class ScatteringExtractor():

    def __init__(self,
                 output_dir,
                 scat_kwargs= {
                     'shape': 2**14,
                     'Q': (12, 1),
                     'J': 8, 
                     'average_fr': True,
                     'T': 2**14},
                 input_sr = 44100,
                 target_sr = 44100):
        super().__init__()

        self.output_dir = output_dir
        self.input_sr = input_sr 
        self.target_sr = target_sr
        self.scat_kwargs = scat_kwargs

        self.scat = Scattering1D(**scat_kwargs).cuda()
        meta = self.scat.meta()
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        self.idxs = np.concatenate([order1[0], order2[0]])

        self.samples = []
        self.fnames = []

        self.datasets = timbremetrics.list_datasets()

    def extract_jtfs(self):
        print('Extracting TRAIN set JTFS coefficients and stats ...')

        for dataset in tqdm(self.datasets):
            dataset_dir = os.path.join(self.output_dir, dataset)
            make_directory(dataset_dir)
            audio_files = timbremetrics.get_audio(dataset)
            for idx, item in enumerate(tqdm(audio_files)): 
                audio = self.prep_audio(item['audio'])
                coefs = self.scat(audio.cuda())[self.idxs].mean(dim=-1)
                self.samples.append(coefs)
                self.fnames.append(os.path.join(dataset_dir, str(idx)))

    def prep_audio(self, audio):
        audio = normalize_audio(audio)
        audio = resample_audio(audio, self.input_sr, self.target_sr)
        audio = torch_float32(pad_or_trim_along_axis(audio, self.scat_kwargs['shape']))
        return audio

    def compute_stats(self):
        self.median = torch.stack(self.samples).median(dim=0)[0]
        # apply log compression and median normalization
        for idx, s in enumerate(self.samples):
            self.samples[idx] = torch.log1p(s / (1e-3 * self.median))
            np.save(self.fnames[idx], self.samples[idx].cpu().numpy())

        samples = torch.stack(self.samples).reshape(-1, self.samples[0].shape[0])
        self.mu = samples.mean(dim=0)
        self.var = samples.var(dim=0)

    def save(self):
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu'), self.mu.cpu().numpy())
        np.save(os.path.join(stats_path, 'var'), self.var.cpu().numpy())
        np.save(os.path.join(stats_path, 'median'), self.median.cpu().numpy())


@gin.configurable
def extract_jtfs_stats(output_dir='datasets/ts_Q3_44100', 
                       scat_kwargs={
                            'shape': 2**15,
                            'Q': (3, 1),
                            'J': int(np.log2(2 ** 15) - 1), 
                            'average': True,
                            'T': 2**15}):
    ''' Extract training set statistics from Time Scattering Coefficients. 
        Args:
        data_dir   -- path to segmented numpy audio files
        output_dir -- output path for the statistics
        gin_config -- feature extraction configuration
    '''
    output_dir = os.path.join(os.getcwd(), output_dir)
    make_directory(output_dir)

    extractor = ScatteringExtractor(output_dir, scat_kwargs)
    extractor.extract_jtfs()
    extractor.compute_stats()
    extractor.save()


def main():
  fire.Fire(extract_jtfs_stats)

if __name__ == "__main__":
    main()