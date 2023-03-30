import abc
import pytorch_lightning as pl
from torch.utils.data import DataLoader


import torch_geometric.data as geom_data

from ipt_sim.data.loaders import (
    IptSimDataset,
    IptSimGraphDataset,
    InstrumentSplitGenerator,
)


class BaseDataModule(pl.LightningDataModule):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, batch_size: float = 1, num_workers: float = 4):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abc.abstractmethod
    def setup(self, stage=None):
        return

    @abc.abstractmethod
    def train_dataloader(self):
        return

    @abc.abstractmethod
    def val_dataloader(self):
        return

    @abc.abstractmethod
    def test_dataloader(self):
        return


class SolIPTSimDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: float = 1,
        num_workers: float = 4,
        seed_csv: str = "./jasmp/seed_filelist.csv",
        split_idxs=(None, None, None),
        ext_csv=None,
        feature="jtfs",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.split_gen = InstrumentSplitGenerator()
        self.split_gen.split(self.split_gen.instruments[1])
        self.train_idxs = None # self.split_gen.train_idxs
        self.val_idxs, self.test_idxs = None, None
        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.feature = feature

    def setup(self, stage=None):
        self.train_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.train_idxs
        )
        self.val_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.val_idxs
        )
        self.test_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, test_idxs=self.test_idxs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class SolIPTSimGraphDataModule(SolIPTSimDataModule):
    def __init__(
        self,
        batch_size: float = 1,
        num_workers: float = 4,
        seed_csv: str = "./jasmp/seed_filelist.csv",
        split_idxs=(None, None, None),
        ext_csv=None,
        feature="jtfs",
    ):
        super().__init__(
            batch_size, num_workers, seed_csv, split_idxs, ext_csv, feature
        )
        # create Dataset from our torch Dataset
        # create DataLoader
        self.train_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.train_idxs
        )
        self.val_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.val_idxs
        )
        self.test_ds = IptSimDataset(
            ext_csv=self.ext_csv, feature=self.feature, test_idxs=self.test_idxs
        )

    def setup(self, stage=None):
        self.train_ds = geom_data.DataLoader(
            geom_data.Dataset(
                x=self.train_ds.features,
                edge_index=self.train_ds.edge_index,
                y=self.train_ds.labels,
            ),
            batch_size=self.batch_size,
        )
        self.val_ds = geom_data.DataLoader(
            geom_data.Dataset(
                x=self.val_ds.features,
                edge_index=self.val_ds.edge_index,
                y=self.val_ds.labels,
            ),
            batch_size=self.batch_size,
        )
        self.test_ds = geom_data.DataLoader(
            geom_data.Dataset(
                x=self.test_ds.features,
                edge_index=self.test_ds.edge_index,
                y=self.test_ds.labels,
            ),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
