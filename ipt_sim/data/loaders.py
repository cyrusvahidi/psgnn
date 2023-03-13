import os, torch, numpy as np, pandas as pd, scipy.io as sio, tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from ipt_sim.utils import get_sol_filepath, replace_ext


class IptSimDataset(Dataset):
    def __init__(
        self,
        f_seed_labels: str = "./jasmp/judgements.mat",
        seed_csv: str = "./jasmp/seed_filelist.csv",
        ext_csv: str = "jasmp/extended_pitch_F4.csv",
        sol_dir: str = "/import/c4dm-datasets/SOL_0.9_HQ/",
        feature: str = "jtfs",
        seed_idxs=None,
    ):
        """SOL Instrumental Playing Technique Dataset
        Args:
            f_seed_labels: str
                file path to human judgement labels for each of the 78 seed audio samples
            seed_csv: str
                file path to csv containing metadata for the 78 seed audio samples
            ext_csv: str
                file path to csv containing metadata for the extended set of sounds
                if None, only seed files are loaded
            sol_dir: str
                path to directory containing SOL_0.9_HQ dataset
            feature: str
                identifier for feature to load: ['jtfs', 'mfcc', 'scat1d_o1', 'scat1d_o2']
        """
        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.sol_dir = sol_dir
        self.feature = feature
        self.seed_labels = sio.loadmat(f_seed_labels)["ensemble"][0]
        self.seed_idxs = (
            seed_idxs if seed_idxs is not None else list(range(len(self.seed_labels)))
        )
        self.feature_path = os.path.join(sol_dir, feature)
        self.load_seed_files()
        if ext_csv:
            self.load_extended_files()
        else:
            self.filelist = self.seed_filelist

        self.get_std_stats()

    def load_seed_files(self):
        df_seed = pd.read_csv(self.seed_csv, index_col=0)

        self.seed_files = df_seed.to_dict(orient="records")
        # seed_filelist = []
        print("Loading seed files ...")
        
        self.seed_filelist = [
            {
                "fpath": replace_ext(f["fpath"]),
                "seed_id": i,
                "pitch": f["pitch"],
                "dynamics": f["dynamics"],
                "family": f["instrument family"],
                "label": self.seed_labels[i],
                "features": self.load_features([replace_ext(f["fpath"])])[0],
            }
            for i, f in tqdm.tqdm(enumerate(self.seed_files))
        ]
        # for i, f in tqdm.tqdm(enumerate(self.seed_files)):
        #     if i in self.seed_idxs:
        #         fpath = replace_ext(get_sol_filepath(self.sol_dir, f))
        #         seed_id = i
        #         label = self.seed_labels[i]
        #         file_info = {
        #             "fpath": fpath,
        #             "seed_id": seed_id,
        #             "pitch": df.iloc[i]["pitch"],
        #             "dynamics": df.iloc[i]["dynamics"],
        #             "family": df.iloc[i]["instrument family"],
        #             "label": label,
        #         }
        #         features = self.load_features([fpath])[0]
        #         file_info["features"] = features
        #         seed_filelist.append(file_info)
        # self.seed_filelist = seed_filelist

    def load_extended_files(self):
        df_ext = pd.read_csv(self.ext_csv, index_col=0)
        self.ext_files = df_ext.to_dict(orient="records")

        print("Loading extended files ...")
        ext_filelist = [
            {
                "fpath": replace_ext(f["filepath"]),
                "seed_id": f["seed_id"],
                "pitch": f["pitch"],
                "dynamics": f["dynamics"],
                "family": f["instrument family"],
                "label": self.seed_labels[f["seed_id"]],
                "features": self.load_features([replace_ext(f["filepath"])])[0],
            }
            for f in tqdm.tqdm(self.ext_files)
        ]

        self.filelist = self.seed_filelist + ext_filelist

    def load_features(self, filelist):
        self.load_feature_stats()

        features = []
        for f in filelist:
            fpath = f
            Sx = np.load(os.path.join(self.feature_path, fpath))
            if "scat1d" in self.feature or "jtfs" in self.feature:
                Sx = np.log1p(Sx / (1e-3 * self.mean))
            features.append(Sx)
        features = torch.tensor(np.stack(features))
        return features

    def load_feature_stats(self):
        stats_dir = os.path.join(self.feature_path, "stats")
        # self.mu = np.load(os.path.join(stats_dir, "mu.npy"))
        # self.std = np.sqrt(np.load(os.path.join(stats_dir, "var.npy")))
        try:
            self.mean = np.load(os.path.join(stats_dir, "mean.npy"))
        except FileNotFoundError:
            print("mean file not found")

    def get_std_stats(self):
        self.mu = self.features.mean(dim=0)
        self.std = self.features.std(dim=0)

    @property
    def features(self):
        features = torch.stack([f["features"] for f in self.filelist])
        return features

    def __getitem__(self, idx):
        item = self.filelist[idx]
        features = (item["features"] - self.mean) / self.std
        y = item["label"]
        return features, y

    def __len__(self):
        return len(self.filelist)


class SplitGenerator:
    def __init__(self, seed_csv):
        self.df = pd.read_csv(seed_csv, index_col=0)

    def get_split(self):
        return self.train_idxs, self.val_idxs, self.test_idxs

    def get_split_id(self):
        return self.split_id


class InstrumentSplitGenerator(SplitGenerator):
    def __init__(
        self, seed_csv: str = "./jasmp/seed_filelist.csv", overfit: bool = False
    ):
        super().__init__(seed_csv)

        self.instruments = sorted(list(set(self.df["instrument family"])))
        self.overfit = overfit
        self.idx = -1

    def split(self, instrument=None):
        self.idx = (
            self.idx + 1 if not instrument else self.instruments.index(instrument)
        )
        self.split_id = self.instruments[self.idx]
        print(f"Testing {self.split_id}")
        train_instrs = self.instruments.copy()
        train_instrs.remove(self.split_id)

        self.train_idxs = [
            i
            for i, f in enumerate(list(self.df["instrument family"]))
            if f in train_instrs
        ]
        self.test_idxs = [
            i
            for i, f in enumerate(list(self.df["instrument family"]))
            if f not in train_instrs
        ]
        if self.overfit:
            self.train_idxs.append(self.test_idxs)
        self.val_idxs = self.test_idxs.copy()


class KFoldsSplitGenerator(SplitGenerator):
    def __init__(
        self,
        seed_csv: str = "./jasmp/seed_filelist.csv",
        n_splits=4,
        overfit: bool = False,
    ):
        super().__init__(seed_csv)
        self.idxs = [i for i in range(len(self.df))]
        self.labels = [x for x in self.df["instrument family"]]
        self.overfit = overfit

        self.skf = StratifiedKFold3().split(
            np.array(self.idxs), np.array(self.labels), n_splits=n_splits
        )

        self.split_id = -1

    def split(self, x=None):
        self.train_idxs, self.val_idxs, self.test_idxs = next(self.skf)
        if self.overfit:
            self.train_idxs.append(self.test_idxs)
        self.split_id += 1


class StratifiedKFold3:
    def split(self, X, y, n_splits=None):
        s = StratifiedKFold(n_splits).split(X, y)
        for train_idxs, test_idxs in s:
            y_train = y[train_idxs]
            train_idxs, val_idxs = train_test_split(
                train_idxs, stratify=y_train, test_size=(1 / (n_splits))
            )
            yield train_idxs, val_idxs, test_idxs
