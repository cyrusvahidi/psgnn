import os, collections

import gin, torch, torch.nn as nn, numpy as np, pandas as pd, wandb
from stmetric.data.loaders import InstrumentSplitGenerator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import scipy.io as sio
import torch.nn as nn

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from stmetric.data import DissimilarityDataModule, InstrumentSplitGenerator, KFoldsSplitGenerator
from stmetric.modules import TimbreTripletKNNAgreement, PatK
from stmetric.modules import LogRatioTripletLoss, IntraClassCorrelation


@gin.configurable
class TripletLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.0001,
        input_dim: int = 869,
        embedding_dim: int = 64,
        feature_dim: int = None,
        lambda_frob: float = 1e-4,
        lambda_loss=1.0,
        eval_k = 5, 
        learn = False,
        test_ds=None,
        net='linear',
        n_labels=5, 
    ):
        super(TripletLightningModule, self).__init__()

        self.lr = lr
        self.criterion = LogRatioTripletLoss()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim if embedding_dim else input_dim
        self.lambda_frob = lambda_frob
        self.feature_dim = feature_dim
        self.lambda_loss = lambda_loss
        self.learn = learn
        self.test_ds = test_ds
        self.n_labels = n_labels

        if net == 'linear':
            linear = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
            self.net = nn.Sequential(collections.OrderedDict([("linear", linear)]))
            out_dim = self.embedding_dim
        else:
            self.net = nn.Sequential(
                collections.OrderedDict([
                    ('fc1', nn.Linear(self.input_dim, self.input_dim)),
                    ('bn2', nn.BatchNorm1d(self.input_dim)),
                    ('relu2', nn.ReLU()),
                    ('fc1', nn.Linear(self.input_dim, self.input_dim // 2)),
                    ('bn2', nn.BatchNorm1d(self.input_dim // 2)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(self.input_dim // 2, self.input_dim // 4, bias=False))
                ])
            )
            out_dim = self.input_dim // 4
        self.out_dim = out_dim


        self.save_hyperparameters()
        self.metrics = {'train/triplet_agmt': TimbreTripletKNNAgreement(eval_k), 'test/triplet_agmt': TimbreTripletKNNAgreement(eval_k)}

        self.test_loss = []
        self.test_figs = []
        mat = sio.loadmat("/homes/cv300/Documents/timbre-metric/notebooks/lostanlen2020jasmp/experiments/similarity/ticelJudgments.mat")
        self.labels = mat['ensemble'][0]

        self.icc = IntraClassCorrelation()

    def pca(self):
        ds = self.test_ds
        pca = PCA(n_components=3)
        if self.learn:
            batch_sz = 64
            features = torch.zeros(ds.all_features.shape[0], self.out_dim)
            for i in range(features.shape[0] // batch_sz + 1):
                features[i*batch_sz:(i+1)*batch_sz] = self.net(ds.all_features[i*batch_sz:(i+1)*batch_sz].to(self.device)) 
            features = features.detach().cpu().numpy()
        else:
            features = ds.all_features.numpy()
        X = pca.fit_transform(features)

        def _pca():
            fig = plt.figure()
            ax = Axes3D(fig) 
            groups = df_pca.groupby('cluster')
            for name, group in groups:
                sc = ax.scatter(group['0'], group['1'], group['2'], label=name)
            fig.get_axes()[0].legend(*sc.legend_elements(prop='colors', num=len(set(list(df_pca['cluster'])))), 
                                     bbox_to_anchor=(1.05, 1), loc=2)
            return fig

        df_pca = pd.DataFrame(X, columns=['0', '1', '2'])

        df_pca['cluster'] = self.labels[[f['seed_id'] for f in ds.filelist]]
        figs = {'cluster': _pca()}
        for attr in ['pitch', 'dynamics', 'family']:
            df_pca['cluster'] = [f[attr] for f in ds.filelist]
            figs[attr] = _pca()

        return figs
         

    def forward(self, features):
        anchor = self.net(features[:, 0, :]) if self.learn else features[:, 0, :]
        positive = self.net(features[:, 1, :]) if self.learn else features[:, 1, :]
        negative = self.net(features[:, 2, :]) if self.learn else features[:, 2, :]
        return anchor, positive, negative

    def get_c(self):
        c = self.c * torch.exp(torch.tanh(self.eps))
        return c

    def compute_loss(self, anchor, positive, negative, distances):
        embeddings = torch.stack([anchor, positive, negative], dim=-1)

        loss = self.criterion(embeddings, distances)
        return loss

    def infer(self, batch, eval=False):
        features, distances, extended, label = batch
        # features, distances = batch
        if len(features.shape) == 2:
            features = features[None, :, :]
            distances = distances[None, :]
        a, i, j = self(features)

        loss_triplet = self.compute_loss(a, i, j, distances)
        loss_icc = self.icc(features[:, 0, :], extended, label)
        loss = loss_triplet + 0.5 * loss_icc


        loss_dict = {"loss": loss, "loss_triplet": loss_triplet, "loss_icc": loss_icc}
        triplets = torch.stack([a, i, j], dim=1)
        return triplets, loss_dict

    def training_step(self, batch, batch_idx):
        triplets, loss_dict = self.infer(batch)
        triplet_agmt = self.metrics['train/triplet_agmt'](triplets)
        self.log("train_loss", loss_dict["loss"])
        self.log("train_loss_triplet", loss_dict["loss_triplet"])
        self.log("train_loss_icc", loss_dict["loss_icc"])
        self.log("train_triplet_agmt", triplet_agmt)
        return {"loss": loss_dict["loss"], 
                "loss_triplet": loss_dict["loss_triplet"], 
                "loss_icc": loss_dict["loss_icc"], 
                "triplet_agmt": triplet_agmt}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        ta = torch.stack([x["triplet_agmt"] for x in outputs]).mean()
        self.log_metric("train/loss_epoch", loss)
        self.log_metric("train/triplet_agmt", ta)

    def validation_step(self, batch, batch_idx):
        triplets, loss_dict = self.infer(batch, eval=True)
        triplet_agmt = self.metrics['test/triplet_agmt'](triplets)
        self.log("val_loss", loss_dict["loss"])
        self.log("val_loss_triplet", loss_dict["loss_triplet"])
        self.log("val_loss_icc", loss_dict["loss_icc"])
        self.log("val_triplet_agmt", triplet_agmt)
        return {"loss": loss_dict["loss"], 
                "loss_triplet": loss_dict["loss_triplet"], 
                "loss_icc": loss_dict["loss_icc"], 
                "triplet_agmt": triplet_agmt}

    def validation_epoch_end(self, outputs):
        keys = list(outputs[0].keys())
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.test_loss.append(float(loss))
        figs = self.pca()
        for k, v in figs.items():
            v.savefig(os.path.join(self.out_dir, f'{k}_{self.current_epoch}.png'))
        for k in keys:
            metric = torch.stack([x[k] for x in outputs]).mean()
            self.log_metric('val/' + k, metric)


    def test_step(self, batch, batch_idx):
        triplets, loss_dict = self.infer(batch, eval=True)
        triplet_agmt = self.metrics['test/triplet_agmt'](triplets)
        self.log("test_loss", loss_dict["loss"])
        self.log("test_loss_triplet", loss_dict["loss_triplet"])
        self.log("test_loss_icc", loss_dict["loss_icc"])
        self.log("test_triplet_agmt", triplet_agmt)
        return {"loss": loss_dict["loss"], 
                "loss_triplet": loss_dict["loss_triplet"], 
                "loss_icc": loss_dict["loss_icc"], 
                "triplet_agmt": triplet_agmt}

    def test_epoch_end(self, outputs):
        keys = list(outputs[0].keys())
        self.test_figs.append(self.pca())
        for k in keys:
            metric = torch.stack([x[k] for x in outputs]).mean()
            self.log_metric('test/' + k, metric)

    def log_metric(self, metric_id, metric):
        self.log(metric_id, metric, prog_bar=True, on_epoch=True, logger=True)
        if hasattr(self.logger, "log"):
            self.logger.experiment.log({metric_id: metric})  # log to wandb

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # return [optimizer], {"scheduler": scheduler}
        return optimizer


@gin.configurable
class TrainerSOL:
    def __init__(
        self,
        max_epochs=10,
        batch_size=64,
        triplet_eval_k=5,
        learn=True,
        out_dir=None,
        seed_csv="/homes/cv300/Documents/timbre-metric/jasmp/seed_filelist.csv",
        ext_csv="/homes/cv300/Documents/timbre-metric/jasmp/extended_pitch_F4.csv",
        split_gen = 'instr'
    ):
        super().__init__()

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.triplet_eval_k = triplet_eval_k
        self.learn = learn
        self.out_dir = out_dir
        self.split_gen = InstrumentSplitGenerator(seed_csv) \
                         if split_gen == 'instr' \
                         else KFoldsSplitGenerator(seed_csv, n_splits=5)

        self.seed_csv = seed_csv 
        self.ext_csv = ext_csv

        self.precision_at_k = PatK(k=5, pruned=False)

    def train_and_test(self):
    
        self.split_gen.split()

        out_dir_instr = os.path.join(self.out_dir, str(self.split_gen.split_id))
        os.makedirs(out_dir_instr, exist_ok=True)

        self.dataset = DissimilarityDataModule(
            batch_size=self.batch_size,
            learn=self.learn,
            split_idxs=self.split_gen.get_split(),
            seed_csv=self.seed_csv,
            ext_csv=self.ext_csv
        )
        self.dataset.setup()
        self.model = TripletLightningModule(learn=self.learn, 
                                            test_ds=self.dataset.test_ds)
        # wandb_logger = WandbLogger()
        # wandb_logger.watch(self.model, log='all')
        self.trainer = pl.Trainer(
            gpus=-1,
            max_epochs=self.max_epochs,
            progress_bar_refresh_rate=1,
            weights_summary=None,
            checkpoint_callback=True,
            # logger=wandb_logger,
        )
        self.model.out_dir = out_dir_instr
        if self.learn and self.max_epochs > 0:
            self.trainer.fit(self.model, self.dataset)
            self.trainer.save_checkpoint(os.path.join(out_dir_instr, f'model_{self.model.feature_dim}.ckpt'))
        else:
            # if no learning setup the dataset to get the features
            self.dataset.setup()  # speed this up

        result = self.test()
        return result

    def test(self):
        result = {"split": str(self.split_gen.get_split_id()), "loss": np.array(self.model.test_loss)}

        features = self.dataset.test_ds.all_features
        if self.learn:
            batch_sz = 64
            features = torch.zeros(features.shape[0], self.model.out_dim)
            for i in range(features.shape[0] // batch_sz + 1):
                features[i*batch_sz:(i+1)*batch_sz] = self.model.net(features[i*batch_sz:(i+1)*batch_sz].to(self.model.device)) 
            features = features.detach().cpu()

        p_at_k = self.precision_at_k(features, self.dataset.test_ds.filelist)
        result['p@k'] = p_at_k
        print(f'P@5 {result["split"]} = {p_at_k:.4f}')

        x = self.trainer.test(self.model, self.dataset.test_ds)
        result[type(self.model.metrics['test/triplet_agmt']).__name__] = x[0]['test/triplet_agmt']
    
        return result

@gin.configurable
def run_metric_learning_sol(n_folds, out_dir=None):
    torch.manual_seed(42) # seed for partial knowledge

    trainer = TrainerSOL(out_dir=out_dir)

    results = []
    for _ in range(n_folds):
        result = trainer.train_and_test()
        print(f"Triplet Agreement: {result[TimbreTripletKNNAgreement.__name__]:.4f}")
        results.append(result)
    return results
