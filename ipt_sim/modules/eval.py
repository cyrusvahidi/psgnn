from typing import Union

import numpy as np
import torch, torch.nn.functional as F
import scipy.io as sio
from torchmetrics import Metric
from .loss import L2dist

def pairwise_euclidean(a, b, p: int = 2):
    return torch.cdist(a, b, p=p)

def pairwise(a, b, p: float = 2):
    return F.pairwise_distance(a, b, p=p)

class EvalMetric(Metric):
    def __init__(self, distance=pairwise, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # self.dissim_mat = dissim_mat
        self.distance = distance
        error = torch.tensor(0)
        self.add_state("error", default=error, dist_reduce_fx="sum")

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        raise NotImplementedError("Can't instantiate abstract base class.")

    def _validate_embeddings(self, embeddings: Union[torch.Tensor, dict]):
        pass


    def _compute_triplet_distances(self, triplets: torch.Tensor):
        dist_a_i = self.distance(triplets[:, 0, :], triplets[:, 1, :])
        dist_a_j = self.distance(triplets[:, 0, :], triplets[:, 2, :])
        distances = torch.column_stack([dist_a_i, dist_a_j])
        return distances


class TimbreTripletKNNAgreement(EvalMetric):
    def __init__(self, 
                 test_triplets,
                 distance=pairwise, 
                 dist_sync_on_step=False,      
                 k=5,
                 k_nn_triplets = False,):
        super().__init__(distance=L2dist(2), dist_sync_on_step=dist_sync_on_step)

        self.test_triplets = test_triplets
        self.k = k 
        self.k_nn_triplets = k_nn_triplets

    def update(self, triplets: Union[torch.Tensor, dict]):
        distances = self._compute_triplet_distances(triplets)
        error = self._compute_item_error(distances)

        self.error = self.error + error

    def compute(self):
        return self.error

    def _compute_item_error(self, distances):
        return (distances[:, 0] < distances[:, 1]).float().sum() / len(distances)

    def get_k_nn_triplets(self, target, anchor_idx):
        ''' Returns all possible pairs (i, j) for anchor a 
            where target[a, i] < target[a, j]

        Args:
            target: symmetrical dissimilarity matrix
            anchor_idx: index of the considered anchor
        Returns:
            i_j_idxs: tensor of shape (N, 2) containing indices (i, j)
        '''
        idxs = self.get_k_nn(target, anchor_idx)
        
        i_j_idxs = torch.stack([idxs[[i, j]] for i in range(len(idxs) - 1) 
                                    for j in range(i + 1, len(idxs))])
        return i_j_idxs

    def get_triplets(self, target, anchor_idx):
        i_idxs = self.get_k_nn(target, anchor_idx)
        j_idxs = self.get_not_k_nn(target, anchor_idx)

        i_j_idxs = torch.stack([torch.tensor([i, j]) for i in i_idxs for j in j_idxs])
        return i_j_idxs

    def get_sorted_idxs(self, target, anchor_idx):
        sorted_idxs = target[anchor_idx].argsort()
        return sorted_idxs

    def get_k_nn(self, target, anchor_idx):
        sorted_idxs = self.get_sorted_idxs(target, anchor_idx)
        k_nn = sorted_idxs[sorted_idxs != anchor_idx][:self.k]
        return k_nn

    def get_not_k_nn(self, target, anchor_idx):
        sorted_idxs = self.get_sorted_idxs(target, anchor_idx)
        j = sorted_idxs[self.k + 1:]
        return j


class RandomTripletAgreement(TimbreTripletKNNAgreement):

    def __init__(
        self,
        dataset=None,
        distance=pairwise_euclidean,
        dist_sync_on_step=False,
        k=5,
        anchor_idx = None,
        k_nn_triplets = False,
        knowledge = 0
    ):
        ''' Randomly evaluate agreements to the triplet constraint D(a, i) < D(a, j)
            from a dissimilarity matrix D
        
        Args:
            k -- int: k-nearest neighborhood to evaluate for each anchor
            anchor_idx -- int: if not None, evaluate around a single anchor,
            k_nn_triplets -- bool: selects triplets from an item's k-nearest neighborhood
        '''
        super().__init__(dataset, distance, dist_sync_on_step, k, anchor_idx, k_nn_triplets)
         
        self.knowledge = knowledge

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        target = target + target.T
        triplet_agreements = []

        anchors = range(target.shape[0]) if not self.anchor_idx else [self.anchor_idx]

        for anchor in anchors:
            if self.k_nn_triplets:
                i_j_idxs = self.get_k_nn_triplets(target, anchor)
            else:
                i_j_idxs = self.get_triplets(target, anchor)
            i = i_j_idxs[:, 0]
            j = i_j_idxs[:, 1]
            
            triplet_agreements.append((distances[anchor, i] < distances[anchor, j]).long())

        triplet_agreements = torch.cat(triplet_agreements)
        num_knowledge = int(len(triplet_agreements) * self.knowledge)

        torch.manual_seed(3407)
        perm = torch.randperm(len(triplet_agreements))
        idxs = perm[:num_knowledge]
        idxs_random = perm[num_knowledge:]

        triplet_agreements[idxs] = 1
        triplet_agreements[idxs_random] = torch.randint(0, 2, (len(idxs_random), ))
        return torch.sum(triplet_agreements) / len(triplet_agreements)


class PatK():
    def __init__(self, 
                 k=5, 
                 pruned=False,
                 sim_mat="/homes/cv300/Documents/timbre-metric/notebooks/lostanlen2020jasmp/experiments/similarity/ticelJudgments.mat"):
        self.k = k 
        self.pruned = pruned 
        mat = sio.loadmat(sim_mat)
        self.labels = mat['ensemble'][0]

    def p_at_k(self, pdists, idx, filelist):
        # get top k queries
        sorted_idxs = pdists[idx].argsort()
        sorted_idxs = sorted_idxs[sorted_idxs != idx][:100]
        # convert all items to their seed idx

        anchor_seed_id = filelist[idx]['seed_id']
        top_k = np.array([filelist[int(i)]['seed_id'] 
                            for i in sorted_idxs 
                            if filelist[int(i)]['seed_id'] != anchor_seed_id or not self.pruned])[:self.k]
        if len(top_k) > 0:
            top_k_labels = self.labels[top_k] 
            anchor_label = self.labels[anchor_seed_id]
            
            n_correct = (anchor_label == top_k_labels).sum() if len(top_k) else 0
            return n_correct, self.k
        else:
            return 0, 0

    def __call__(self, features, filelist):
        pdists = torch.cdist(features, features)
        n_correct = 0
        n_total = 0
        for i in range(len(pdists)):
            _n_correct, _n_total = self.p_at_k(pdists, i, filelist)
            n_correct += _n_correct
            n_total += _n_total
        return n_correct / n_total
    