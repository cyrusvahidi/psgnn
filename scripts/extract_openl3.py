import os

import timbremetrics
import torch
import numpy as np
import fire 
import openl3
from tqdm import tqdm

from stmetric.utils import make_directory


def extract_openl3(output_dir = 'openl3-6144'):
    out_dir = os.path.join(os.getcwd(), output_dir)
    make_directory(out_dir)

    datasets = timbremetrics.list_datasets()

    samples = []

    for dataset in tqdm(datasets):
        dataset_dir = os.path.join(out_dir, dataset)
        make_directory(dataset_dir)
        audio = timbremetrics.get_audio(dataset)
        for idx, entry in enumerate(tqdm(audio)): 
            embeddings, timestamps = openl3.get_audio_embedding(
                entry['audio'], 
                entry['sample_rate'], 
                embedding_size = 6144)
            embeddings_mean = embeddings.mean(axis=0)
            samples.append(embeddings_mean)

            np.save(os.path.join(dataset_dir, str(idx)), embeddings_mean)

    # reshape to (n_examples x timesteps, embedding_size)
    samples = np.stack(samples).reshape(-1, samples[0].shape[0])
    mu = samples.mean(axis=0)
    var = samples.var(axis=0)
    stats_path = os.path.join(output_dir, 'stats')
    make_directory(stats_path)
    np.save(os.path.join(stats_path, 'mu'), mu)
    np.save(os.path.join(stats_path, 'var'), var)

def main():
  fire.Fire(extract_openl3)

if __name__ == "__main__":
    main()