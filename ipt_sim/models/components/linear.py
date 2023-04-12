import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeConv
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.nn import Sequential
from timm.models.layers import DropPath


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.elu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size, activation=F.elu))
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = h + features
        return h


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class RelationNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = input_size // 8
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class GraphNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=1024):
        super().__init__()
        self.gcn = GCN(input_size, hidden_dim, output_size)
        self.relation = RelationNetwork(input_size)
        self.mlp = MLP(output_size, output_size, hidden_dim)

    def forward(self, x):
        # knn_g = dgl.knn_graph(x, 3,dist='cosine')
        # print(knn_g.shape)
        N, d = x.shape[0], x.shape[1]
        eps = np.finfo(float).eps
        self.sigma = self.relation(x)
        emb_x = torch.div(x, (self.sigma + eps))
        emb1 = torch.unsqueeze(emb_x, 1)
        emb2 = torch.unsqueeze(emb_x, 0)
        W = ((emb1 - emb2) ** 2).mean(2)
        W = torch.exp(-W / 2)

        topk, indices = torch.topk(W, 3)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
        W = W * mask
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        src, dst = torch.nonzero(S, as_tuple=True)

        g = dgl.graph((src, dst))
        out = self.mlp(self.gcn(g, x))
        return out


class DynGCN(nn.Module):
    def __init__(self, channels):
        super(DynGCN, self).__init__()
        hidden_dim = channels // 2
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(channels, hidden_dim, activation=F.elu))
        # self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim,activation=F.elu))
        self.dropout = nn.Dropout(0.1)
        self.relation = RelationNetwork(channels)

    def forward(self, x):
        N, d = x.shape[0], x.shape[1]
        eps = np.finfo(float).eps
        self.sigma = self.relation(x)
        emb_x = torch.div(x, (self.sigma + eps))
        emb1 = torch.unsqueeze(emb_x, 1)
        emb2 = torch.unsqueeze(emb_x, 0)
        W = ((emb1 - emb2) ** 2).mean(2)
        W = torch.exp(-W / 2)

        topk, indices = torch.topk(W, 3)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
        W = W * mask
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        src, dst = torch.nonzero(S, as_tuple=True)

        g = dgl.graph((src, dst))

        for i, layer in enumerate(self.layers):

            h = self.dropout(x)
            h = layer(g, x)

        return h


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="relu",
        drop_path=0.0,
    ):

        super().__init__()

        out_features = out_features or in_features

        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.ReLU()

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(0.1)

    def forward(self, x):

        shortcut = x
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.drop_path(x) + shortcut

        return x


class Graph_Block(nn.Module):
    def __init__(self, in_channels):

        super(Graph_Block, self).__init__()
        self.channels = in_channels
        self.graph_conv = DynGCN(in_channels)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(0.1)

    def forward(self, x):
        _tmp = x
        out = self.graph_conv(x)
        out = out.unsqueeze(-1).unsqueeze(-1)
        x = self.fc(out)
        x = x.squeeze(-1).squeeze(-1)
        x += _tmp
        return x


class Graph_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=1024):
        super().__init__()

        self.backbone = nn.ModuleList([])
        self.channels = input_size
        self.blocks = 2
        in_channels = self.channels
        self.backbone += [
            Sequential(Graph_Block(in_channels), FFN(in_channels, in_channels // 4))
            for i in range(self.blocks)
        ]
        self.backbone = Sequential(*self.backbone)

    def forward(self, x):

        x = self.backbone(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.L = nn.Linear(input_size, output_size, bias=False)
        nn.init.kaiming_uniform_(self.L.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        out = self.L(x)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(input_size, hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        L = nn.Linear(hidden_dim, output_size, bias=False)
        nn.init.kaiming_uniform_(L.weight, mode="fan_in", nonlinearity="relu")
        self.layers.append(L)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        return h
