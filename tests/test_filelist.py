import os
from pathlib import Path
import pytest, pandas as pd
import torch
from stmetric.data import InstrumentSplitGenerator, LegacySOLTripletRatioDataset, DissimilarityDataModule, SOLTripletRatioDataset
from stmetric.data.loaders import KFoldsSplitGenerator
from stmetric.utils import get_sol_IMT

META_PATH = '/homes/cv300/Documents/timbre-metric/jasmp/'
EXTENDED_FPATH_DYNAMICS = os.path.join(META_PATH, 'extended_dynamics.csv')
SEED_FPATH = os.path.join(META_PATH, 'seed_filelist.csv')
EXTENDED_FPATH_PITCH = os.path.join(META_PATH, 'extended_pitch_F4.csv')
INST_TO_FAM = os.path.join(META_PATH, 'instrument_to_family.csv')

SEED_CSV = '/homes/cv300/Documents/timbre-metric/jasmp/seed_filelist.csv'
SIM_MAT = '/homes/cv300/Documents/timbre-metric/jasmp/sim.npy'
EXT_CSV = '/homes/cv300/Documents/timbre-metric/jasmp/extended_pitch_F4.csv'
SOL_DIR = '/import/c4dm-datasets/SOL_0.9_HQ/'

FEATURE = 'scat1d_o2'


def test_extended_dynamics_filelist():
    
    df_ext = pd.read_csv(EXTENDED_FPATH_DYNAMICS, index_col=0)
    df_seed = pd.read_csv(SEED_FPATH, index_col=0)

    for i in range(len(df_ext)):
        ext_row = df_ext.iloc[i] 
        seed_id = ext_row['seed_id'] # get seed ID for the extended file
        seed_fname = df_seed.iloc[seed_id]['fname'] 
        ext_fname = os.path.basename(ext_row.filepath)

        # check seed IMT matches extended IMT
        assert get_sol_IMT(seed_fname) == get_sol_IMT(ext_fname)


def test_triplet_constraints():

    ds = LegacySOLTripletRatioDataset(filter_idxs=None, 
                                k=40, 
                                feature=FEATURE,  
                                seed_csv=SEED_CSV, 
                                sol_dir=SOL_DIR, 
                                ext_csv=EXT_CSV, 
                                sim_mat=SIM_MAT)
    _, dists, _ = next(iter(ds))
    # check all distances respect constraints
    assert bool((dists[:, 0] < dists[:, 1]).sum() == len(dists))


def test_extended_pitch_filelist():
    
    df_ext = pd.read_csv(EXTENDED_FPATH_PITCH, index_col=0)
    df_seed = pd.read_csv(SEED_FPATH, index_col=0)

    for i in range(len(df_ext)):
        ext_row = df_ext.iloc[i] 
        seed_id = ext_row['seed_id'] # get seed ID for the extended file
        seed_fname = df_seed.iloc[seed_id]['fname'] 
        ext_fname = os.path.basename(ext_row.filepath)

        # check seed IMT matches extended IMT
        assert get_sol_IMT(seed_fname) == get_sol_IMT(ext_fname)


def test_extended_idxs():
    ds = LegacySOLTripletRatioDataset(feature=FEATURE, 
                                filter_idxs=None,
                                k=0, 
                                seed_csv=SEED_CSV, 
                                sol_dir=SOL_DIR, 
                                ext_csv=EXT_CSV, 
                                sim_mat=SIM_MAT)
    for k, ext in ds.seed_to_ext.items():
        for i in ext[1:]:
            assert k == ds.filelist[i]['seed_id']
            seed_fpath = ds.filelist[k]['fpath']
            fpath = ds.filelist[i]['fpath']
            assert get_sol_IMT(os.path.basename(seed_fpath)) == get_sol_IMT(os.path.basename(fpath))
    assert len(ds.filelist) == len(ds.features)


instruments = ['Woodwind', 'Brass', 'Strings', 'PluckedStrings']
@pytest.mark.parametrize("instrument", instruments)
def test_filter_instrument_family(instrument):
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()
    df = pd.read_csv(SEED_CSV, index_col=0)
    inst_to_fam = pd.read_csv(INST_TO_FAM, index_col=0)
    instrs = sorted(list(set(df['instrument family'])))
    instrs.remove(instrument)
    filter_idxs = [i 
                    for i, f in enumerate(list(df['instrument family']))
                    if f in instrs]

    for f in dm.train_ds.filelist:
        assert f['seed_id'] in filter_idxs
        assert inst_to_fam[inst_to_fam['instrument'] == Path(f['fpath']).parts[0]]['family'].item() in instrs

    for f in dm.test_ds.filelist:
        assert f['seed_id'] not in filter_idxs
        assert inst_to_fam[inst_to_fam['instrument'] == Path(f['fpath']).parts[0]]['family'].item() == instrument


instruments = ['Woodwind', 'Brass', 'Strings', 'PluckedStrings']
@pytest.mark.parametrize("instrument", instruments)
def test_k_nn_instruments(instrument):
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()

    for ds in [dm.train_ds]:
        for anchor in ds.filter_idxs:
            k_nn = ds.get_k_nn(ds.dissim_mat, anchor_idx=anchor, filter_idxs=ds.filter_idxs)
            for x in k_nn:
                assert int(x) in ds.filter_idxs
            not_nn = ds.get_not_nn(ds.dissim_mat, anchor_idx=anchor, filter_idxs=ds.filter_idxs)
            for x in not_nn:
                assert int(x) in ds.filter_idxs


instruments = ['Woodwind', 'Brass', 'Strings', 'PluckedStrings']
@pytest.mark.parametrize("instrument", instruments)
def test_seed_triplet_ratio_dataset_distances(instrument):
    N = 1000
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()
    dm.train_ds.load_triplets()

    triplets = dm.train_ds.triplet_idxs[:N]
    dist_aij = dm.train_ds.dissim_mat[triplets[:, 0:1], triplets[:, [1, 2]]]
    assert (dist_aij[:, 0] >= dist_aij[:, 1]).sum() == 0


def test_extended_idxs_triplet_ratio_dataset():
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    instrument = "Woodwind"
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()
    ds = dm.train_ds 
    for k, seed_ext in ds.anchor_to_extended.items():
        for ext in seed_ext:
            assert ds.filelist[k]['seed_id'] == ext['seed_id']
            assert get_sol_IMT(ds.filelist[k]['fpath']) == get_sol_IMT(ext['fpath'])


instruments = ['Woodwind', 'Brass', 'Strings', 'PluckedStrings']
@pytest.mark.parametrize("instrument", instruments)
def test_filelist_triplet_ratio_dataset(instrument):
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()
    ds = dm.train_ds 
    for item in ds.filelist:
        seed_id = item['seed_id']
        seed_item = [f for f in ds.seed_filelist if f['seed_id'] == seed_id][0]
        assert get_sol_IMT(os.path.basename(item['fpath'])) == get_sol_IMT(os.path.basename(seed_item['fpath']))


instruments = ['Woodwind', 'Brass', 'Strings', 'PluckedStrings']
@pytest.mark.parametrize("instrument", instruments)
def test_feature_list(instrument):
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    split_gen.split(instrument)
    dm = DissimilarityDataModule(
        split_idxs=split_gen.get_split(),
        seed_csv=SEED_CSV,
        ext_csv=EXT_CSV
    )
    dm.setup()
    ds = dm.train_ds 

    for idx, item in enumerate(ds.seed_filelist):
        assert (ds.seed_features[idx] == ds.all_features[idx]).all()
    extended_features = torch.cat([x for x in ds.extended_features.values()]) 
    for idx, item in enumerate(extended_features):
        assert (extended_features[idx] == ds.all_features[idx + len(ds.seed_features)]).all()


def test_extended_anchor():
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    instrument = "Woodwind"
    split_gen.split(instrument)

    ds = SOLTripletRatioDataset(filter_idxs=split_gen.get_split()[-3], 
                                seed_csv=SEED_CSV,
                                ext_csv=EXT_CSV)
    # features, distances, triplet_idxs, anchor_ext = next(iter(ds))
    import pdb; pdb.set_trace()


def test_k_fold_split():
    split_gen = KFoldsSplitGenerator(SEED_CSV, n_splits=5)
    split_gen.split()
    assert sum([len(x) for x in split_gen.get_split()]) == 78
    split_gen.split()
    split_gen.split()
    assert split_gen.split_id == 2
    train, val, test = split_gen.get_split()
    for x in test:
        assert x not in train 
        assert x not in val 
    for x in val:
        assert x not in train 
        assert x not in test
     
     
def test_instrument_split():
    split_gen = InstrumentSplitGenerator(SEED_CSV)
    df = pd.read_csv(SEED_CSV, index_col=0)
    instrs = list(set(df['instrument family']))
    for i in instrs:
        split_gen.split()
        assert split_gen.split_id == i