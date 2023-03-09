import scipy.io as sio, os
import pandas as pd
import gin
from stmetric.data.loaders import LegacySOLTripletRatioDataset
import torch
import torch.nn.functional as F
import numpy as np

sim = "/homes/cv300/Documents/timbre-metric/notebooks/lostanlen2020jasmp/experiments/similarity/ticelJudgments.mat"
SEED_CSV = "/homes/cv300/Documents/timbre-metric/jasmp/seed_filelist.csv"
EXT_CSV = '/homes/cv300/Documents/timbre-metric/jasmp/extended_pitch_F4.csv'
mat = sio.loadmat(sim)
labels = mat['ensemble']

gin.enter_interactive_mode()
# gin.parse_config_file('/homes/cv300/Documents/timbre-metric/gin/doce/sol_template.gin')

def precision_at_k(pdists, idx, labels, k=5, pruned=False):
    # get top k queries
    sorted_idxs = pdists[idx].argsort()
    sorted_idxs = sorted_idxs[sorted_idxs != idx][:200]
    # convert all items to their seed idx

    anchor_seed_id = ds.filelist[idx]['seed_id']
    top_k = np.array([ds.filelist[int(i)]['seed_id'] 
                      for i in sorted_idxs 
                      if ds.filelist[int(i)]['seed_id'] != anchor_seed_id or not pruned])[:k] 
    n_same_seed = (ds.filelist[idx]['seed_id'] == top_k).sum()
    top_k_labels = labels[top_k] 
    anchor_label = labels[anchor_seed_id]
    
    n_correct = (anchor_label == top_k_labels).sum() if len(top_k_labels) else 0
    return n_correct, n_same_seed

df = pd.DataFrame(columns=['feature', 'subject', 'p@5'])

for feature in ['jtfs']: #, 'scat1d_o2', 'scat1d_o1', 'mfcc', 'openl3', 'rand-512']:
    print(f"P@5 metric for {feature}")
    ds = LegacySOLTripletRatioDataset(sol_dir='/import/c4dm-datasets/SOL_0.9_HQ/', 
                                      ext_csv=EXT_CSV,
                                      seed_csv=SEED_CSV,
                                      feature=feature)

    pdists = torch.cdist(ds.features, ds.features)
    p_at_5 = 0
    same_seed = 0
    for n in range(mat['ensemble'].shape[0]):
        labels = mat['ensemble'][n]
        n_correct = 0
        n_seed = 0
        n_total = 0
        for i in range(len(pdists)):
            _n_correct, _n_seed = precision_at_k(pdists, i, labels)
            n_correct += _n_correct
            n_seed += _n_seed
            n_total += 5
        print(f"p@5 for subject {n}: {n_correct / n_total:.3f}")
        df = df.append({'feature': feature, 'subject': n, 'p@5': n_correct / n_total}, ignore_index=True)
        # print(f"proportion top K same as seed for subject {n}: {n_seed / n_total:.3f}")
        p_at_5 += n_correct / n_total
        same_seed += n_seed / n_total
        
    print(f"mean p@5: {p_at_5 / mat['ci'].shape[0]:.3f}")
    # print(f"mean proportion top K same as seed for subject: {same_seed / mat['ci'].shape[0]:.3f}")

df.to_csv('jasmp_sol_p5_jtfs.csv')