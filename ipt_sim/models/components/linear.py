import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeConv
import numpy as np
import torch 

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.elu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, activation=F.elu))
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = h +features
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

    def __init__(self,input_size):
        super().__init__()
        hidden_size = input_size//8
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4,1)

    def forward(self,x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out 

class GraphNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super().__init__()
        self.gcn = GCN(input_size, hidden_dim, output_size)
        self.relation = RelationNetwork(input_size)
        self.mlp = MLP(output_size, output_size, hidden_dim)
        
    def forward(self, x):
        #knn_g = dgl.knn_graph(x, 3,dist='cosine')
        #print(knn_g.shape)
        N, d    = x.shape[0], x.shape[1]
        eps = np.finfo(float).eps
        self.sigma = self.relation(x)
        emb_x = torch.div(x,(self.sigma+eps))
        emb1 = torch.unsqueeze(emb_x,1) 
        emb2 = torch.unsqueeze(emb_x,0)
        W = ((emb1-emb2)**2).mean(2)
        W = torch.exp(-W/2)
        
        topk, indices = torch.topk(W, 3)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask+torch.t(mask))>0).type(torch.float32)      
        W    = W*mask
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2
        src, dst = torch.nonzero(S,as_tuple=True)
        g = dgl.graph((src, dst))
        out = self.mlp(self.gcn(g,x))
        return out


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
        self.layers += [
            nn.Linear(input_size, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.5)
        ]
        L = nn.Linear(hidden_dim, output_size, bias=False)
        nn.init.kaiming_uniform_(L.weight, mode="fan_in", nonlinearity="relu")
        self.layers.append(L)
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        return h
