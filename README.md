# Perceptual Musical Similarity Metric Learning with Graph Neural Networks

[![Paper]()](https://hal.science/hal-04178191/)

## Installation
- `git clone`
- `pip install -e .`
- `pip install -r requirements.txt`


## Usage
- `python ipt_sim/train.py`
- `python ipt_sim/train.py data=extended model=proxy data.feature=openl3 model.prune_accuracy=True`

K-folds:

- `python ipt_sim/train_kfold.py data=kfold model=graph data.feature=openl3`
