<div align="center">
  
# Perceptual Musical Similarity Metric Learning with Graph Neural Networks
Modelling auditory similarity judgements between musical instrument playing techniques with graph neural networks

[![Paper]()](https://hal.science/hal-04178191/)
</div>
## Abstract
Sound retrieval for assisted music composition depends on evaluating similarity between musical instrument sounds, which is partly influenced by playing techniques.
Previous methods utilizing Euclidean nearest neighbours over acoustic features show some limitations in retrieving sounds sharing equivalent timbral properties, but potentially generated using a different instrument, playing technique, pitch or dynamic. 
In this paper, we present a metric learning system designed to approximate human similarity judgments between extended musical playing techniques using graph neural networks. Such structure is a natural candidate for solving similarity retrieval tasks, yet have seen little application in modelling perceptual music similarity. 
We optimize a Graph Convolutional Network (GCN) over acoustic features via a proxy metric learning loss to learn embeddings that reflect perceptual similarities. 
Specifically, we construct the graph's adjacency matrix from the acoustic data manifold with an example-wise adaptive k-nearest neighbourhood graph: Adaptive Neighbourhood Graph Neural Network (AN-GNN). 
Our approach achieves 96.4% retrieval accuracy compared to 38.5% with a Euclidean metric and 86.0\% with a multilayer perceptron (MLP), while effectively considering retrievals from distinct playing techniques to the query example. 

```BibTex
@inproceedings{vahidi2023perceptual,
  title={Perceptual musical similarity metric learning with graph neural networks},
  author={Vahidi, Cyrus and Singh, Shubhr and Benetos, Emmanouil and Phan, Huy and Stowell, Dan and Fazekas, Gy{\"o}rgy and Lagrange, Mathieu},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Installation
- `git clone https://github.com/cyrusvahidi/ipt-similarity.git`
- `pip install -e .`
- `pip install -r requirements.txt`

## Data
- `SOL_0.9_HQ` is a proprietary dataset from IRCAM. Please contact the authors for details.
- Similarity judgements coming soon ...

## Training
- `python ipt_sim/train.py`
- `python ipt_sim/train.py data=extended model=graph data.feature=openl3 model.prune_accuracy=True`
    - the `model.prune_accuracy` option determines if the retrieval metric ignores retrievals of the same IMT class as the query
- `python ipt_sim/train_kfold.py data=kfold model=graph data.feature=openl3`
    - trains with K-folds cross validation
