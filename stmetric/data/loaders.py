import os
from typing import Union

import gin, torch, numpy as np, pandas as pd, scipy.io as sio
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch.nn.functional as F

from stmetric.utils import (
    replace_ext
)
from stmetric.utils import (get_sol_filepaths, 
                            get_sol_filepath,
                            get_sol_instrument, 
                            get_fname)


@gin.configurable 
class TripletRatioDataset(Dataset):
    def __init__(self, k = 5, not_k = None): 
        super().__init__()
        self.k = k
        self.not_k = not_k if not_k else k

    def get_k_nn(self, dissim_mat, anchor_idx, filter_idxs):
        sorted_idxs = dissim_mat[anchor_idx].argsort()
        sorted_idxs = sorted_idxs[sorted_idxs != anchor_idx]
        sorted_idxs = sorted_idxs[np.isin(sorted_idxs, filter_idxs)]
        k = self.k if self.k > 0 and self.k < len(dissim_mat) else len(dissim_mat) - 1
        k_nn = sorted_idxs[:k]
        return k_nn

    def get_not_nn(self, dissim_mat, anchor_idx, filter_idxs, n=10):
        sorted_idxs = self.get_sorted_idxs(dissim_mat, anchor_idx, filter_idxs)
        not_nn = sorted_idxs[self.k + 1:]
        not_nn_shuffled = torch.randperm(len(not_nn))
        not_nn = not_nn[not_nn_shuffled][:n]
        return not_nn

    def get_sorted_idxs(self, dissim_mat, anchor_idx, filter_idxs):
        sorted_idxs = dissim_mat[anchor_idx].argsort()
        sorted_idxs = sorted_idxs[sorted_idxs != anchor_idx]
        sorted_idxs = sorted_idxs[np.isin(sorted_idxs, filter_idxs)]
        return sorted_idxs



@gin.configurable
class SOLTripletRatioDataset(TripletRatioDataset):
    def __init__(
        self, 
        sim_mat: str = '/homes/cv300/Documents/timbre-metric/jasmp/sim.npy',
        seed_csv: str = 'jasmp/seed_filelist_pitched.csv',
        ext_csv: str = 'jasmp/extended_filelist_pitched.csv', 
        sol_dir: str = '/import/c4dm-datasets/SOL_0.9_HQ/',
        feature: str = 'jtfs',
        filter_idxs = None,  # seed item indices in the dissim mat to include
        k = None,
        not_k = None,
    ):
        self.seed_dissim = 1.0 - np.load(sim_mat)
        k = (len(self.seed_dissim) - 1) if not k else k
        super().__init__(k, not_k)

        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.sol_dir = sol_dir 
        self.filter_idxs = filter_idxs if filter_idxs is not None else [i for i in range(len(pd.read_csv(seed_csv, index_col=0)))]
        self.filter_idxs = [f for f in self.filter_idxs if f != 55] # 'Tubas/Bass-Tuba/exploding-slap-pitched/
        self.feature = feature
        self.do_pca = 'pca' in self.feature
        if 'pca' in feature:
            self.feature = self.feature.split('-')[0]
        self.feature_path = os.path.join(sol_dir, self.feature)

        self.load_seed()
        self.load_dissim_matrix()
        self.load_triplets()
        self.load_extended_files()

        self.seed_features = torch.cat([f['features'] for f in self.seed_filelist])
        self.extended_features = {k: torch.cat([f['features'] for f in v]) for k, v in self.anchor_to_extended.items()}
        self.all_features = torch.cat([f['features'] for f in self.filelist])
        self.normalize_features()

    def load_seed(self):
        df = pd.read_csv(self.seed_csv, index_col=0)

        self.seed_files = list(df['fname'])
        self.seed_filelist = []
        for i, f in enumerate(self.seed_files):
            if i in self.filter_idxs:
                fpath = replace_ext(get_sol_filepath(self.sol_dir, f))
                seed_id = i 
                file_info = {'fpath': fpath, 
                             'seed_id': seed_id,
                             'pitch': df.iloc[i]['pitch'], 
                             'dynamics': df.iloc[i]['dynamics'],
                             'family': df.iloc[i]['instrument family']}
                features = self.load_features([file_info])
                file_info['features'] = features
                self.seed_filelist.append(file_info)

    def load_dissim_matrix(self):
        self.dissim_mat = torch.tensor(self.seed_dissim)

    def load_stats(self):
        stats_dir = os.path.join(self.feature_path, 'stats')
        self.mu = np.load(os.path.join(stats_dir, 'mu.npy'))
        self.std = np.sqrt(np.load(os.path.join(stats_dir, 'var.npy')))
        try:
            self.mean = np.load(os.path.join(stats_dir, 'mean.npy'))
        except FileNotFoundError:
            print('mean file not found')

    def load_features(self, filelist):

        if 'rand' not in self.feature:
            self.load_stats() 

        features = []
        for f in filelist:
            fpath = f['fpath']
            if 'rand' not in self.feature:
                Sx = np.load(os.path.join(self.feature_path, fpath))
                if 'scat1d' in self.feature or 'jtfs' in self.feature:
                    Sx = np.log1p(Sx / (1e-3 * self.mean))
                features.append(Sx)
            else:
                dim = int(self.feature.split('-')[1])
                features.append(torch.randn(dim))
        features = torch.tensor(np.stack(features))
        # if self.do_pca:
        #     print('computing PCA of features')
        #     features = self.pca_features(features)
        return features

    def normalize_features(self):
        if 'rand' not in self.feature: 
            mu = torch.mean(self.all_features, dim=0)
            std = torch.std(self.all_features, dim=0)
            self.all_features = (self.all_features - mu) / std
            self.seed_features = (self.seed_features - mu) / std
            self.extended_features = {k: (v - mu) / std for k, v in self.extended_features.items()}

    def load_labels(self):
        self.dynamics_to_idx = {k: i for i, k in enumerate(set(self.df_ext['dynamics']))}
        self.dynamics_labels = torch.stack([torch.tensor(self.dynamics_to_idx[f['dynamics']]) 
                                            for f in self.filelist])

    def load_triplets(self):
        triplet_idxs = []
        anchors = self.filter_idxs
        for anchor_idx in anchors:
            sorted_idxs = self.dissim_mat[anchor_idx].argsort()
            sorted_idxs = sorted_idxs[sorted_idxs != anchor_idx]
            sorted_idxs = sorted_idxs[np.isin(sorted_idxs, anchors)]
            comb = torch.combinations(sorted_idxs)
            valid = (self.dissim_mat[anchor_idx, comb[:, 0]] < self.dissim_mat[anchor_idx, comb[:, 1]])
            i_j_idxs = comb[valid.nonzero()[:, 0]]
            a_idxs = torch.tensor([anchor_idx]).repeat(len(i_j_idxs))[:, None]
            triplet_idxs.append(torch.cat([a_idxs, i_j_idxs], dim=-1))
        
        self.triplet_idxs = torch.cat(triplet_idxs)

    def load_extended_files(self):
        df_ext = pd.read_csv(self.ext_csv, index_col=0)
        self.ext_files = df_ext.to_dict(orient='records')
        self.anchor_to_extended = {i: self.get_extended_files(anchor_idx) 
                                   for i, anchor_idx in enumerate(self.filter_idxs)}
        # self.extended_features = {k: self.load_features(v) for k, v in self.anchor_to_extended.items()}

        self.filelist = self.seed_filelist + [item for l in self.anchor_to_extended.values() for item in l]
        # self.all_features = torch.cat([self.seed_features] + [x for x in self.extended_features.values()]) 

        m = min([len(x) for _, x in self.anchor_to_extended.items()])
        self.num_ext_per_sample = 2**int(np.log2(m))
        

    def get_extended_files(self, anchor_idx):
        ext_files = [{'fpath': replace_ext(f['filepath']), 'seed_id': f['seed_id'], 'pitch': f['pitch'], 
                      'dynamics': f['dynamics'], 'family': f['instrument family']} 
                     for f in self.ext_files 
                     if f['seed_id'] == anchor_idx] 
        for f in ext_files:
            f.update({"features": self.load_features([f])})
        return ext_files 

    def pca_features(self, features, n_components=40):
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(features.numpy())
        features = torch.tensor(X) 
        return features
    
    def __getitem__(self, idx):
        triplet_idxs = self.triplet_idxs[idx] # idxs of triplet items (into dissim mat)
        feature_idxs = torch.tensor([list(self.filter_idxs).index(x) for x in triplet_idxs]) # idxs of triplet in feature tensor
        triplets = self.seed_features[feature_idxs] 
        distances = self.dissim_mat[triplet_idxs[0], triplet_idxs[[1, 2]]]

        anchor_idx = int(feature_idxs[0]) 
        anchor_ext = self.extended_features[anchor_idx]
        ext_idxs = torch.randperm(len(anchor_ext))[:self.num_ext_per_sample]
        anchor_ext = anchor_ext[ext_idxs]
        
        return triplets, distances, anchor_ext, anchor_idx

    def __len__(self):
        return len(self.triplet_idxs)


class SplitGenerator():

    def __init__(self, seed_csv):
        self.df = pd.read_csv(seed_csv, index_col=0)

    def get_split(self):
        return self.train_idxs, self.val_idxs, self.test_idxs

    def get_split_id(self):
            return self.split_id

    
class InstrumentSplitGenerator(SplitGenerator):
    
    def __init__(self, seed_csv: str, overfit: bool = False):
        super().__init__(seed_csv)

        self.instruments = sorted(list(set(self.df['instrument family'])))
        self.overfit = overfit
        self.idx = -1

    def split(self, instrument=None):
        self.idx = self.idx + 1 if not instrument else self.instruments.index(instrument)
        self.split_id = self.instruments[self.idx]
        print(f"Testing {self.split_id}")
        train_instrs = self.instruments.copy()
        train_instrs.remove(self.split_id)

        self.train_idxs = [i 
                           for i, f in enumerate(list(self.df['instrument family']))
                           if f in train_instrs]
        self.test_idxs = [i 
                          for i, f in enumerate(list(self.df['instrument family']))
                          if f not in train_instrs]
        if self.overfit:
            self.train_idxs.append(self.test_idxs)
        self.val_idxs = self.test_idxs.copy()


class KFoldsSplitGenerator(SplitGenerator):
    
    def __init__(self, seed_csv, n_splits=4, overfit: bool = False, fast_dev_run=False):
        super().__init__(seed_csv)
        self.idxs = [i for i in range(len(self.df))]
        self.labels = [x for x in self.df['instrument family']]
        self.overfit = overfit

        self.fast_dev_run = fast_dev_run

        self.skf = StratifiedKFold3().split(np.array(self.idxs), np.array(self.labels), n_splits=n_splits)

        self.split_id = -1

    def split(self, x=None):
        if not self.fast_dev_run:
            self.train_idxs, self.val_idxs, self.test_idxs = next(self.skf)
            if self.overfit:
                self.train_idxs.append(self.test_idxs)
            self.split_id += 1
        else:
            self.train_idxs, self.val_idxs, self.test_idxs = np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), np.array([1,2,3,4,5])


class StratifiedKFold3():

    def split(self, X, y, n_splits=None):
        s = StratifiedKFold(n_splits).split(X, y)
        for train_idxs, test_idxs in s:
            y_train = y[train_idxs]
            train_idxs, val_idxs = train_test_split(train_idxs, stratify=y_train, test_size=(1 / (n_splits)))
            yield train_idxs, val_idxs, test_idxs




@gin.configurable
class LegacySOLTripletRatioDataset(TripletRatioDataset):
    def __init__(
        self, 
        sim_mat: str = '/homes/cv300/Documents/timbre-metric/jasmp/sim.npy',
        seed_csv: str = 'jasmp/seed_filelist_pitched.csv',
        ext_csv: str = 'jasmp/extended_filelist_pitched.csv', 
        sol_dir: str = '/import/c4dm-datasets/SOL_0.9_HQ/',
        feature: str = 'jtfs',
        filter_idxs = None,
        k = None,
        not_k = None,
        overfit = False,
        batch_size=256
    ):
        self.seed_dissim = 1.0 - np.load(sim_mat)
        k = (len(self.seed_dissim) - 1) if not k else k
        super().__init__(k, not_k)

        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.sol_dir = sol_dir 
        self.filter_idxs = filter_idxs if filter_idxs is not None else [i for i in range(len(pd.read_csv(seed_csv, index_col=0)))]
        self.feature = feature
        self.do_pca = 'pca' in self.feature
        if 'pca' in feature:
            self.feature = self.feature.split('-')[0]
        self.feature_path = os.path.join(sol_dir, self.feature)
        self.batch_size = batch_size

        self.load_seed_and_extended_files()
        self.load_features()
        self.load_dissim_matrix()
        self.load_labels()

    def load_seed_and_extended_files(self):
        df = pd.read_csv(self.seed_csv, index_col=0)

        self.seed_files = list(df['fname'])
        self.filelist = []
        for i, f in enumerate(self.seed_files):
            if i in self.filter_idxs:
                fpath = replace_ext(get_sol_filepath(self.sol_dir, f))
                seed_id = i 
                self.filelist.append({'fpath': fpath, 
                                      'seed_id': seed_id,
                                      'pitch': df.iloc[i]['pitch'], 
                                      'dynamics': df.iloc[i]['dynamics'],
                                      'family': df.iloc[i]['instrument family']})

        self.seed_to_ext = {d['seed_id']: [i] for i, d in enumerate(self.filelist)}

        self.df_ext = pd.read_csv(self.ext_csv, index_col=0)
        ext_files = self.df_ext.to_dict(orient='records')

        for i, f in enumerate(ext_files):
            seed_id = f['seed_id'] 
            if seed_id in self.filter_idxs:
                fpath = replace_ext(f['filepath'])
                self.filelist.append({'fpath': fpath, 
                                      'seed_id': seed_id,
                                      'pitch': f['pitch'], 
                                      'dynamics': f['dynamics'],
                                      'family': f['instrument family']})
                self.seed_to_ext[seed_id].append(len(self.filelist) - 1)

    def load_dissim_matrix(self):
        self.dissim_mat = torch.tensor(self.seed_dissim)

        # self.NN_query = torch.zeros(len(self.filelist), 50)
        # for i, x1 in enumerate(self.filelist):
        #     self.NN_query[i] = torch.sort(self.dissim_mat[x1['seed_id']])[0][:50]

    def load_stats(self):
        stats_dir = os.path.join(self.feature_path, 'stats')
        self.mu = np.load(os.path.join(stats_dir, 'mu.npy'))
        self.std = np.sqrt(np.load(os.path.join(stats_dir, 'var.npy')))
        try:
            self.mean = np.load(os.path.join(stats_dir, 'mean.npy'))
        except FileNotFoundError:
            print('mean file not found')

    def load_features(self):
        if 'rand' not in self.feature:
            self.load_stats() 

        features = []
        for f in self.filelist:
            fpath = f['fpath']
            if 'rand' not in self.feature:
                Sx = np.load(os.path.join(self.feature_path, fpath))
                if 'scat1d' in self.feature or 'jtfs' in self.feature:
                    Sx = np.log1p(Sx / (1e-3 * self.mean))
                features.append(Sx)
            else:
                dim = int(self.feature.split('-')[1])
                features.append(torch.randn(dim))
        self.features = torch.tensor(np.stack(features))
        if 'rand' not in self.feature: 
            mu = torch.mean(self.features, dim=0)
            std = torch.std(self.features, dim=0)
            self.features = (self.features - mu) / std
        if self.do_pca:
            print('computing PCA of features')
            self.pca_features()

    def load_labels(self):
        self.dynamics_to_idx = {k: i for i, k in enumerate(set(self.df_ext['dynamics']))}
        self.dynamics_labels = torch.stack([torch.tensor(self.dynamics_to_idx[f['dynamics']]) 
                                            for f in self.filelist])

    def get_anchor_triplets(self, anchor_idx):
        seed_id = self.filelist[anchor_idx]['seed_id']
        # get valid triplets in the anchor's k-nearest neighborhood and not K-nn
        k_nn = self.get_k_nn(self.dissim_mat, seed_id, filter_idxs=self.filter_idxs)
        not_nn = self.get_not_nn(self.dissim_mat, seed_id, n=self.batch_size - self.k, filter_idxs=self.filter_idxs)

        idxs = torch.cat([k_nn, not_nn])
        comb = torch.combinations(idxs)
        valid = (self.dissim_mat[seed_id, comb[:, 0]] < self.dissim_mat[seed_id, comb[:, 1]])
        i_j_idxs = comb[valid.nonzero()[:, 0]]

        # get the i and j idxs for (a, i, j) where d(a, i) < d(a, j)
        i_idxs = i_j_idxs[:, 0]
        j_idxs = i_j_idxs[:, 1]
        distance_a_i = self.dissim_mat[seed_id, i_idxs]
        distance_a_j = self.dissim_mat[seed_id, j_idxs]
        # stack as (N, 2) for N distances D(a, i) and D(a, j)
        distances = torch.stack([distance_a_i, distance_a_j], dim=-1)    

        for i in range(len(i_idxs)):
            # pick a random one from the extended set
            i_idxs[i] = np.random.choice(self.seed_to_ext[int(i_idxs[i])])
            j_idxs[i] = np.random.choice(self.seed_to_ext[int(j_idxs[i])])
        a_idxs = torch.tensor([anchor_idx]).repeat(len(i_idxs))

        # stack as (N, feature_dim, 3) for N (a, i, j)
        feature_idxs = torch.stack([a_idxs, i_idxs, j_idxs], dim=-1)

        return feature_idxs, distances
    

    def pca_features(self, n_components=40):
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(self.features.numpy())
        self.features = torch.tensor(X) 

    def ndcg(self, net, out_dim, k=10):
        batch_sz = 64
        features = torch.zeros(self.features.shape[0], out_dim)
        for i in range(self.features.shape[0] // batch_sz + 1):
            features[i*batch_sz:(i+1)*batch_sz] = net(features[i*batch_sz:(i+1)*batch_sz])
        
        discounts = 1 / torch.log2(1 + torch.arange(1, k + 1))
        features = features.detach().cpu()
        pdist = torch.zeros((features.shape[0], k))
        for i in range(pdist.shape[0]):
            pdist[i] = torch.sort(F.pairwise_distance(features[i], features))[0][:k]
        qdist = self.NN_query[:, :k]

        ideal = (2**(-np.log2(qdist + 1)) * discounts).sum(dim=1)
        embed = (2**(-np.log2(pdist + 1)) * discounts).sum(dim=1)
        ndcg = (embed / ideal).mean()
        return ndcg

    def precision_at_k(self, net=None, out_dim=None, k=5, device=None):
        def _p_at_k(pdists, idx, labels):
            # get top k queries
            sorted_idxs = pdists[idx].argsort()
            top_k = sorted_idxs[sorted_idxs != idx][:k]
            # convert all items to their seed idx

            top_k = np.array([self.filelist[int(i)]['seed_id'] for i in top_k]) 
            top_k_labels = labels[top_k] 
            anchor_label = labels[self.filelist[idx]['seed_id']]
            
            n_correct = (anchor_label == top_k_labels).sum()
            return n_correct

        sim = "/homes/cv300/Documents/timbre-metric/notebooks/lostanlen2020jasmp/experiments/similarity/ticelJudgments.mat"
        mat = sio.loadmat(sim)
        labels = mat['ensemble'][0]

        if net:
            batch_sz = 64
            features = torch.zeros(self.features.shape[0], out_dim)
            for i in range(self.features.shape[0] // batch_sz + 1):
                features[i*batch_sz:(i+1)*batch_sz] = net(self.features[i*batch_sz:(i+1)*batch_sz].to(device)) 
            features = features.detach().cpu()
        else:
            features = self.features
        pdists = torch.cdist(features, features)
        n_correct = 0
        n_total = 0
        for i in range(len(pdists)):
            _n_correct = _p_at_k(pdists, i, labels)
            n_correct += _n_correct
            n_total += 5
        return n_correct / n_total
    
    def __getitem__(self, idx):
        feature_idxs, distances = self.get_anchor_triplets(idx)

        features = self.features[feature_idxs] 
        labels = self.dynamics_labels[feature_idxs].reshape(-1)

        return features, distances, labels

    def __len__(self):
        return len(self.filelist)