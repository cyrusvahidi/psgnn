from typing import List, Optional, Tuple

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    metrics = []
    for k in range(cfg.data.num_splits):
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.data,
            k=k,
            num_splits=cfg.data.num_splits,
        )
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "trainer": trainer,
        }

        if cfg.get("train"):
            print("Training ...")
            trainer.fit(
                model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
            )

        train_metrics = trainer.callback_metrics

        if cfg.get("test"):
            trainer.test(dataloaders=datamodule)

        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}
        metrics.append(metric_dict)

    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metrics = train(cfg)
    for acc in ['test/acc', 'test/acc_euclidean']:
        accs = torch.stack([m[acc] for m in metrics])
        print(f"{acc}: {float(accs.mean()):.4f} ± {float(accs.std()):.4f}")

    # safely retrieve metric value for hydra-based hyperparameter optimization


if __name__ == "__main__":
    main()
