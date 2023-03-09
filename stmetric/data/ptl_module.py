import abc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import gin

from stmetric.data import (SOLTripletRatioDataset)

@gin.configurable
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
        

@gin.configurable
class DissimilarityDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        batch_size: float = 1, 
        num_workers: float = 4,
        learn = False,
        split_idxs = None,
        overfit: bool = False,
        seed_csv = None,
        ext_csv = None
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.learn = learn
        self.train_idxs, self.val_idxs, self.test_idxs = split_idxs
        self.overfit = overfit
        self.seed_csv = seed_csv 
        self.ext_csv = ext_csv

    def setup(self, stage=None):
        self.train_ds = SOLTripletRatioDataset(filter_idxs=self.train_idxs, 
                                               seed_csv=self.seed_csv,
                                               ext_csv=self.ext_csv)
        import numpy as np
        self.test_ds = SOLTripletRatioDataset(filter_idxs=self.test_idxs, 
                                              seed_csv=self.seed_csv,
                                              ext_csv=self.ext_csv)
        self.val_ds = SOLTripletRatioDataset(filter_idxs=self.val_idxs, 
                                             seed_csv=self.seed_csv,
                                             ext_csv=self.ext_csv)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=True,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )