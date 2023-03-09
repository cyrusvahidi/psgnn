import os 

import wandb
from pytorch_lightning.loggers import WandbLogger

def init_logger(gin_config):
    run = wandb.init()
    logger = WandbLogger(experiment=run)
    if gin_config:
        wandb.save(os.path.join(wandb.run.dir, gin_config))

    return logger