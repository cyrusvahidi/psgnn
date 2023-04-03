from typing import Any, List

from functools import partial

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from torchmetrics import MaxMetric, MeanMetric

from ipt_sim.modules.eval import PatK


class SolIPTSimLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion, 
        net, 
        prune_accuracy: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.test_acc = PatK(k=5, pruned=prune_accuracy)

        # # for averaging loss across batches

        # for tracking best so far validation accuracy

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y - 1)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/valid", valid_triplets, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`

    #     # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
    #     # this may not be an issue when training on mnist
    #     # but on larger datasets/models it's easy to run into out-of-memory errors

    #     # consider detaching tensors before returning them from `training_step()`
    #     # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
    #     ds = self.trainer.train_dataloader.loaders.dataset
    #     acc = self.train_acc(ds.features, ds.filelist)

    #     self.log("train/acc", acc, prog_bar=True)

    # def validation_step(self, batch: Any, batch_idx: int):
    #     loss = self.model_step(batch)

    #     # update and log metrics
    #     self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    #     # we can return here dict with any tensors
    #     # and then read it in some callback or in `training_epoch_end()` below
    #     # remember to always return loss from `training_step()` or backpropagation will fail!
    #     return {"loss": loss}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     batch_losses = [x["loss"] for x in outputs]
    #     loss = torch.stack(batch_losses).mean() 

    #     self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # # update and log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        ds = self.trainer.test_dataloaders[0].dataset 
        batch_sz = 64
        features = torch.cat(
            [
                self.net(ds.features[i : i + batch_sz].to(self.device))
                for i in range(0, len(ds.features), batch_sz)
            ]
        ) 
        acc = self.test_acc(features, ds.filelist, test_idxs=ds.test_idxs)
        acc_euclidean = self.test_acc(ds.features, ds.filelist, test_idxs=ds.test_idxs)

        self.log("test/acc", acc, prog_bar=True)
        self.log("test/acc_euclidean", acc_euclidean, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SolIPTSimLitModule(None, None)
