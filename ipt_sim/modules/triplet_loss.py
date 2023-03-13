"""
    See omoindrot's blog for the really useful breakdown
    and broadcasting techniques
    https://omoindrot.github.io/triplet-loss
"""

import torch
import torch.nn.functional as F

def online_batch_all(embeddings, labels, margin=0.5, squared=False, normalize=False, device='cpu'):
    ''' Returns the triplet loss over a batch of embeddings, given class labels.
        Only 'semi-hard' triplets are counted i.e a_p - a_n + margin > 0
    Args:
        embeddings: tensor of shape (batch_size, embedding_dim)
        labels: integer labels of the batch, of size (batch_size, )
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        num_valid_triplets: total number of mined triplets
        fraction_positive_triplets: fraction of valid triplets used to compute the loss
    '''
    # pairwise embedding distances
    if normalize:
        embeddings = F.normalize(embeddings)
    p_dist = _pairwise_distances(embeddings, squared=squared)
    
    # anchor to positive (batch_size, batch_size, 1)
    a_p = p_dist.unsqueeze(2)
    # anchor to positive (batch_size, 1, batch_size)
    a_n = p_dist.unsqueeze(1)
    
    # mask of valid triplets (batch_size, batch_size, batch_size
    # True if [i, j, k] is a valid (a, p, n) triplet
    valid_triplet_mask = _triplet_mask_all(labels, device)

    # create triplet tensor (batch_size, batch_size, batch_size)
    triplet_loss = a_p - a_n + margin

    # zero the non-valid triplets
    triplet_loss = triplet_loss * valid_triplet_mask.float() 

    # Remove non-semi-hard triplets (easy)
    # i.e when a_p + margin < a_n
    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).type_as(triplet_loss))

    # Get the number of triplets greater than 0
    valid_triplets = (triplet_loss > 1e-16).float()
    
    num_positive_triplets = torch.sum(valid_triplets)
    
    num_valid_triplets = torch.sum(valid_triplet_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, num_positive_triplets, num_valid_triplets

def _pairwise_distances(embeddings, squared=True):
    ''' Returns pairwise distances for a batch of embeddings
        Computational reference:
        https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
    Args:
        embeddings: tensor of shape (batch_size, embedding_dim)
        squared: the squared euclidean distance matrix is computed when true
    Returns:
        pairwise distances between all the embeddings of shape (batch_size, batch_size)
    '''

    gram_matrix = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    
    diag = torch.diag(gram_matrix)

    # D(x, y) = ||x||^2 - 2 <a, b> + ||y||^2
    dists = diag + diag.T - 2 * gram_matrix

    if not squared:
        # sqrt produces zero values for infinite gradiences
        # add double precision epsilon
        dists = torch.sqrt(dists + 1e-16)

        # clamp negative values that occur due to lack of floating point precision
        dists = F.relu(dists)

    return dists
    
def _triplet_mask_all(labels, device):
    '''Returns the 3-D mask [a, p, n] where True is a valid a-p-n triplet
    Args:
        labels: (batch_size, )
    Returns:
        triplet_mask: (batch_size, batch_size, batch_size)
    '''
    # create 3-D tensor [a, p, n] where labels[a] != labels[p] != labels[n]
    # get anchors
    positive_labels = labels.unsqueeze(1) == labels
    # get the anchor idxs i = j
    idxs_not_anchor = (torch.eye(labels.shape[0]).type_as(labels) == 0)
    # combine anchors with positives [a, p, 1]
    anchor_positive = (positive_labels & idxs_not_anchor).unsqueeze(2)
    # get the negative labels [a, 1, n]
    anchor_negative = ~positive_labels.unsqueeze(1)

    # Tensor of the valid triplets [i, j, k] True, if 
    triplet_mask = (anchor_positive & anchor_negative)

    # mask = idxs & valid_triplets
    
    return triplet_mask

def _online_hard(labels):
    # anchor_positive_mask = _get_anchor_positive_mask(labels)
    # anchor_negative_mask = _get_anchor_negative_mask(labels)
    return NotImplemented

def _get_anchor_positive_mask(labels):
    '''Returns the 2-D mask [a, p] where 1 is a valid a-p pair
    Args:
        labels: (batch_size, )
    Returns:
        anchor_positive_mask: (batch_size, batch_size)
    
    '''
    # find equal label idxs
    # broadcast label tensor and perform outer product
    positive_idxs = labels.unsqueeze(1) == labels

    idxs_not_anchor = torch.eye(labels.shape[0]) == 0

    anchor_positive_mask = positive_idxs & idxs_not_anchor

    return anchor_positive_mask

def _get_anchor_negative_mask(labels):
    positive_idxs = labels.unsqueeze(1) == labels
    negative_mask = ~positive_idxs
    return negative_mask
