import torch, torch.nn as nn


class L2dist(nn.Module):
    def __init__(self, p, squared=False):
        super(L2dist, self).__init__()
        self.norm = p
        self.squared = squared

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        try:
            out = torch.pow(diff, self.norm).sum(dim=1)
        except IndexError:
            import pdb

            pdb.set_trace()
        out = out if self.squared else torch.pow(out + eps, 1.0 / self.norm)
        return out


class LogRatioTripletLoss(nn.Module):
    """Log-ratio triplet loss
    Redesigned from Deep Metric Learning Beyond Binary Supervision
    https://github.com/tjddus9597/Beyond-Binary-Supervision-CVPR19
    """

    def __init__(self, reduction=torch.sum):
        super().__init__()

        # self.pdist = nn.PairwiseDistance(p = 2.0)
        self.pdist = L2dist(2)
        self.reduction = reduction

    def forward(self, embeddings, distances, weights=None, eps=1e-6):
        """Compute the log ratio loss between embeddings (a, i, j)
            and groundtruth distances (a, i, j) for triplets without
            positive-negative separation
        Args:
            embeddings: of shape (batch, dim, 3) containing anchor a and its two neighbors i
                and j, respectively at indices 0, 1, 2 of the last dimension.
            distances: of shape (batch, 2) containing groundtruth distances D(a, i)
                and D(a, j) respectively at indices 0, 1 of the last dimension

        Returns:
            loss: aggregate loss over a batch
        """
        a, i, j = embeddings[:, :, 0], embeddings[:, :, 1], embeddings[:, :, 2]

        dist_a_i, dist_a_j = self.pdist(a, i), self.pdist(a, j)
        gt_dist_a_i, gt_dist_a_j = distances[:, 0], distances[:, 1]

        log_dist = torch.log1p(dist_a_i + eps) - torch.log1p(dist_a_j + eps)
        log_gt_dist = torch.log1p(gt_dist_a_i + eps) - torch.log1p(gt_dist_a_j + eps)
        log_ratio_losses = (log_dist - log_gt_dist).pow(2)
        if weights:
            log_ratio_losses *= weights
        loss = self.reduction(log_ratio_losses)

        return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing

    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(
            torch.randn(nb_classes, sz_embed).cuda()
        )  # change to embeddings
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.dist = L2dist(2.0)

    def forward(self, anchors, extended, labels):
        P = self.proxies

        dist = self.dist(X, P)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(self.alpha * (dist - self.mrg))
        neg_exp = torch.exp(-self.alpha * (dist + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


class IntraClassCorrelation(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.dist = L2dist(2.0)

    def forward(self, anchors, Xn, labels):
        N = 0
        intra_class_dist = torch.tensor(0.0).type_as(Xn)
        for i in range(len(anchors)):
            extended = Xn[i]
            if len(extended.shape) == 1:
                extended = extended[None, :]
            intra_class_dist += torch.log1p(self.dist(anchors[i], extended).sum())
            N += len(extended)
        return intra_class_dist / N if N > 0 else torch.tensor(0.0)
