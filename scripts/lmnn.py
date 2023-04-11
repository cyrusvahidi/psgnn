import torch

from metric_learn import LMNN

from ipt_sim.data import IptSimDataset
from ipt_sim.modules.eval import PatK

ds = IptSimDataset(feature="jtfs")

X = ds.features.numpy()
Y = ds.labels.numpy()

print("Fitting ...")
lmnn = LMNN(k=5, learn_rate=1e-5, verbose=True, max_iter=100, convergence_tol=1e-5)
lmnn.fit(X, Y)

print("Testing ...")
W = torch.from_numpy(lmnn.get_mahalanobis_matrix()).type(torch.float32)
precision = PatK(k=5, pruned=True)

print(f"ii-P@5 Metric: {precision(torch.matmul(W, ds.features.T).T, ds.filelist)}")
print(f"ii-P@5 Euclidean: {precision(ds.features, ds.filelist)}")

precision = PatK(k=5, pruned=False)
print(f"P@5 Metric: {precision(torch.matmul(W, ds.features.T).T, ds.filelist)}")
print(f"P@5 Euclidean: {precision(ds.features, ds.filelist)}")