import torch
from ipt_sim.data import IptSimDataset
from ipt_sim.modules.eval import PatK

k = 5
ds = IptSimDataset(feature="jtfs")
precision = PatK(k=k, pruned=True)

pdists = torch.tensor([[0 for _ in range(6)] + [1 for _ in range(5)]])
filelist = [{'seed_id': i} for i in range(11)] # unique seed id for every item
labels = torch.tensor([1 for _ in range(6)] + [2 for _ in range(5)])
n_correct, _ = precision.p_at_k(pdists, labels, idx=0, filelist=filelist)
assert n_correct == k

pdists = torch.tensor([[0 for _ in range(6)] + [1 for _ in range(5)]])
filelist = [{'seed_id': i} for i in range(11)] # unique seed id for every item
labels = torch.tensor([1] + [3 for _ in range(5)] + [2 for _ in range(5)])
n_correct, _ = precision.p_at_k(pdists, labels, idx=0, filelist=filelist)
assert n_correct == 0


features = torch.stack([torch.ones(64) + x['label'] for x in ds.filelist])
print(precision(features, ds.labels, ds.filelist))

precision = PatK(k=k, pruned=False)
features = torch.stack([torch.ones(64) + x['seed_id'] for x in ds.filelist])
print(precision(features, ds.labels, ds.filelist))