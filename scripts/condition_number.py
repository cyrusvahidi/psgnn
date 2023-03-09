import fire
import glob
import torch
from stmetric.model import TripletLightningModule

def cond_num(model_dir, feature_dim):
    ckpts = glob.glob(f"{model_dir}/*/*.ckpt")
    for i, ckpt in enumerate(ckpts):
        model = TripletLightningModule()
        model.load_from_checkpoint(ckpt, feature_dim=feature_dim)
        print(f"Condition number {i}: {torch.linalg.cond(model.net.linear.weight)}")

if __name__ == "__main__":
    fire.Fire(cond_num)