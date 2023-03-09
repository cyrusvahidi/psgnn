
import doce
import time
import numpy as np, pandas as pd

import os
import fire

from stmetric.model import run_metric_learning_sol
from stmetric.utils import gin_register_and_parse
from stmetric.modules import TimbreTripletKNNAgreement

if __name__ == "__main__":
  doce.cli.main()
  
def set(args=None):
  experiment = doce.Experiment(
    name = 'timbre-metric-sol',
    purpose = 'metric learning for timbre dissimilarity',
    author = 'Cyrus Vahidi',
    address = 'c.vahidi@qmul.ac.uk',
    version = '0.1'
  )

  experiment.setPath('output', 'results/'+experiment.name+'/')

  # no metric plan
  experiment.addPlan('none',
                    feature = ['jtfs', 'mfcc', 'openl3', 'scat1d_o2', 'rand-512'], 
                    triplet_ratio_k = [0, 5, 8, 10, 20, 40],
                    split=['instr', 'kfold'])

  # random projection plan
  experiment.addPlan('rand',
                    feature = ['jtfs', 'mfcc', 'openl3', 'scat1d_o2'],  # 'rand-359', 'rand-40','jtfs-pca', 'openl3-pca'
                    max_epochs = np.array([0], dtype=np.intc),
                    learning_rate = np.array([1e-2]),
                    triplet_ratio_k = [0, 5, 8, 10, 40, 20],
                    overfit = [0], 
                    embedding_dim=[0, 64, 128])
  experiment.addPlan('learn',
                     feature = ['jtfs', 'mfcc', 'openl3', 'scat1d_o2', 'jtfs-pca', 'openl3-pca', "rand-512"],
                     max_epochs = np.array([0, 1, 5, 10, 20, 30, 50], dtype=np.intc), 
                     learning_rate = np.array([1e-2, 1e-3, 5e-4, 1e-5]), # use small for seed set only
                     triplet_ratio_k = [0, 5, 8, 10, 20, 40],
                     overfit = [0, 1],
                     embedding_dim=[0, 64, 128],
                     net=['linear', 'mlp'], 
                     split=['instr', 'kfold'])
  pl = 'learn'
  # experiment.default(plan=pl, factor='feature', modality='mfcc')
  experiment.default(plan=pl, factor='max_epochs', modality=10)
  experiment.default(plan=pl, factor='learning_rate', modality=1e-3)
  experiment.default(plan=pl, factor='triplet_ratio_k', modality=0)
  experiment.default(plan=pl, factor='embedding_dim', modality=0)
  experiment.default(plan=pl, factor='net', modality='linear')
  experiment.default(plan=pl, factor='split', modality='instr')
  experiment.default(plan='rand', factor='embedding_dim', modality=0)
  experiment.setMetrics(
    triplet = ['mean*%+', 'std%', 'median*%+'],
    duration = ['mean+*']
  )
  experiment.default(plan='none', factor='triplet_ratio_k', modality=0)
  experiment.default(plan='none', factor='split', modality='instr')

  df = pd.read_csv('/homes/cv300/Documents/timbre-metric/jasmp/seed_filelist.csv', 
                   index_col=0)
  experiment.test_instrs = sorted(list(df['instrument family'].unique()))
  experiment.instr_to_idx = {instr: idx for idx, instr in enumerate(experiment.test_instrs)}

  experiment._display.metricPrecision = 4

  experiment._display.bar = False

  return experiment

def step(setting, experiment):
  # if os.path.exists(experiment.path.output+setting.id()+'_triplet.npy'):
    # return
  n_folds = len(experiment.test_instrs) if setting.split == 'instr' else 5
  setting_triplet = np.zeros((n_folds, )) 
  setting_p5 = np.zeros((n_folds, )) 
  setting_loss = []
  tic = time.time()
  print(setting.id())

  preprocess_gin_file(setting)
  out_dir = os.path.join(experiment.path.output, setting.id())
  os.makedirs(out_dir, exist_ok=True)

  results = run_metric_learning_sol(n_folds, out_dir=out_dir)
  for i, r in enumerate(results):
    # idx = experiment.instr_to_idx[r['instrument']]
    idx = i
    setting_triplet[idx] = r[TimbreTripletKNNAgreement.__name__]
    setting_p5[idx] = r['p@k']
    setting_loss.append(r['loss'])

  setting_loss = np.stack(setting_loss)
  np.save(experiment.path.output+setting.id()+'_triplet.npy', setting_triplet)
  np.save(experiment.path.output+setting.id()+'_p5.npy', setting_p5)
  np.save(experiment.path.output+setting.id()+'_loss.npy', setting_loss)
  duration = time.time()-tic
  np.save(experiment.path.output+setting.id()+'_duration.npy', duration)


def preprocess_gin_file(setting):
  template = '/homes/cv300/Documents/timbre-metric/gin/doce/sol_template.gin'
  temp = '/homes/cv300/Documents/timbre-metric/gin/doce/sol_setting.gin'

  feature_dim = {'jtfs': 509, 
                 'mfcc': 40, 
                 'openl3': 512, 
                 'rand-512': 512, 
                 'rand-359': 359,
                 'rand-40': 40,
                 'jtfs-pca': 40,
                 'openl3-pca': 40,
                 'scat1d_o2': 338}
  # if hasattr(setting, 'overfit'):
  if hasattr(setting, 'learning_rate'): # learn plan
    config = [f'TripletLightningModule.lr = {setting.learning_rate}',
              f'TripletLightningModule.input_dim = {feature_dim[setting.feature]}',
              f'TripletLightningModule.embedding_dim = {setting.embedding_dim}',
              f'TripletLightningModule.net = \'{setting.net}\'',
              f'SOLTripletRatioDataset.k = {setting.triplet_ratio_k}',
              f'SOLTripletRatioDataset.feature = \'{setting.feature}\'',
              f'TrainerSOL.max_epochs = {setting.max_epochs}',
              f'DissimilarityDataModule.overfit = {setting.overfit}',
              f'TrainerSOL.learn = {True}',
              f'TrainerSOL.split_gen = \'{setting.split}\'']
  else:
      config = [f'SOLTripletRatioDataset.feature = \'{setting.feature}\'',
                f'SOLTripletRatioDataset.k = {setting.triplet_ratio_k}',
                f'TrainerSOL.learn = {False}',
                f'TrainerSOL.split_gen = \'{setting.split}\'']


  open(temp, 'w').close() # clear temp
  with open(template,'r') as f_template, open(temp,'a') as f_temp:
    # write template to temp
    for line in f_template:
      f_temp.write(line + '\n')
    # write config
    for line in config:
      f_temp.write(line + '\n')

  gin_config_path = f'/homes/cv300/Documents/timbre-metric/gin/doce/sol_setting.gin'
  gin_register_and_parse(gin_config_path)