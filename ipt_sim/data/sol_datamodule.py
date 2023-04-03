import abc
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ipt_sim.data.loaders import IptSimDataset, InstrumentSplitGenerator, KFoldsSplitGenerator


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
        feature="jtfs"
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # self.split_gen = InstrumentSplitGenerator()
        # self.split_gen.split(self.split_gen.instruments[1])
        self.split_gen = KFoldsSplitGenerator(n_splits=4, val_set=False)
        self.split_gen.split()
        self.train_idxs = None #self.split_gen.train_idxs 
        self.val_idxs, self.test_idxs = None, None # None, self.split_gen.test_idxs
        self.seed_csv = seed_csv
        self.ext_csv = ext_csv
        self.feature = feature

    def setup(self, stage=None):
        self.train_ds = IptSimDataset(ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.train_idxs)
        # self.val_ds = IptSimDataset(ext_csv=self.ext_csv, feature=self.feature, seed_idxs=self.val_idxs)
        self.test_ds = IptSimDataset(ext_csv=self.ext_csv, feature=self.feature, test_idxs=self.test_idxs)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )
    
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )