import pandas as pd 
import numpy as np

import run_doce_sol


def get_plans():
    experiment = run_doce_sol.set('')
    plans = experiment.plans()
    selector = {}

    data = {}
    settings = {}
    for p in plans:
        plan = getattr(experiment, p)
        (d_triplet, s_triplet, _) = experiment.metric.get('triplet', 
                                          plan.select(selector), 
                                          experiment.path.output)
        (d_loss, s_loss, _) = experiment.metric.get('loss', 
                                          plan.select(selector), 
                                          experiment.path.output)
        (d_p5, _, _) = experiment.metric.get('p5', 
                                          plan.select(selector), 
                                          experiment.path.output)
        data[p] = {s: {'triplet': d_triplet} for s in s_triplet}
        settings[p] = s_triplet
    return data, settings


def get_setting_idx(_data, 
                     _settings,
                     learn=1,
                     feature='jtfs', 
                     epochs=20, 
                     lr='0dot001',
                     net='linear',
                     overfit=0,
                     random=0,
                     embedding_dim=0):
    for idx, s in enumerate(_settings):
        if learn:
            if f'feature={feature}' in s \
               and f'overfit={overfit}' in s \
               and f'max_epochs={epochs}' in s \
               and f'net={net}' in s:
                print(f"Setting found: {feature} - learn: {learn} - overfit: {overfit} - net: {net}")
                return idx
        else:
            if f'feature={feature}' in s:
                return idx

    if learn:
        print(f"Setting not found for feature: {feature} - learn: {learn} - overfit: {overfit}")
    else:
        print(f"Setting not found for feature: {feature} - learn: {learn}")
    return np.array([])


def get_setting_data(_data, 
                     _settings,
                     learn=1,
                     feature='jtfs', 
                     epochs=10, 
                     net='linear',
                     lr='0dot001',
                     overfit=0,
                     random=0,
                     embedding_dim=0,
                     split='instr',
                     ):
    for idx, s in enumerate(_settings):
        if learn:
            #    and f'overfit={overfit}' in s \
            if f'feature={feature}' in s \
               and f'max_epochs={epochs}' in s \
               and f'net={net}' in s \
               and f'split={split}' in s \
               and f'learning_rate={lr}' in s:
                print(f"Y - Setting found! {feature} - learn: {learn} - overfit: {overfit} - epochs: {epochs} - net - {net} - lr: {lr}")
                return _data[s]['triplet'][idx]
        else:
            if f'feature={feature}' in s and f'split={split}' in s:
                return _data[s]['triplet'][idx]

    if learn:
        print(f"WARN: Setting NOT found for feature: {feature} - learn: {learn} - overfit: {overfit} - epochs: {epochs} - net - {net} - lr: {lr}")
    else:
        print(f"WARN: Setting NOT found for feature: {feature} - learn: {learn}")
    return np.array([])


def get_learn_settings(data, settings, feature='jtfs', **kwargs):
    data_no_learn = get_setting_data(data['none'], settings['none'], feature=feature, learn=0, **kwargs)
    # print("Getting random settings")
    # data_rand = get_setting_data(data['rand'], settings['rand'], feature=feature, learn=0, random=1)
    data_learn = get_setting_data(data['learn'], settings['learn'], feature=feature, learn=1, **kwargs)
#     print("Getting overfit settings")
    # data_overfit = get_setting_data(data['learn'], settings['learn'], feature=feature, learn=1, overfit=1, **kwargs)
#     data = data_no_learn, data_rand, data_learn
    # return data_no_learn, data_rand, data_learn, data_overfit
    return {'none': data_no_learn, 'learn': data_learn}


def make_dataframe(setting_data, feature=None, instrs = []):
    feature = [feature for _ in instrs] 
    df_no_learn = pd.DataFrame(list(zip(list(setting_data['none']), instrs, feature)), 
                            columns=['triplet agreement', 'instrument', 'feature'])
        # df_rand = pd.DataFrame(list(zip(setting_data[1], instrs, feature)), 
    #                        columns=['triplet agreement', 'instrument', 'feature'])
    df_learn = pd.DataFrame(list(zip(list(setting_data['learn']), instrs, feature)), 
                            columns=['triplet agreement', 'instrument', 'feature'])
    # df_overfit = pd.DataFrame(list(zip(setting_data[2], instrs, feature)), 
    #                           columns=['triplet agreement', 'instrument', 'feature'])

    df_no_learn['setting'] = 'none'
    # df_rand['setting'] = 'rand'
    df_learn['setting'] = 'train'
    # df_overfit['setting'] = 'train + test'

    # df = pd.concat([df_no_learn, df_rand, df_learn, df_overfit])
    df = pd.concat([df_no_learn, df_learn])
#     df = pd.concat([df_no_learn, df_rand, df_learn])
    return df


def make_dfs(data, 
            settings, 
            test_instrs,
            features = ['jtfs', 'mfcc', 'openl3', 'rand-512', 'rand-359', 'rand-40', 'jtfs-pca', 'openl3-pca'],
            **kwargs):
    dfs = []
    for feature in features:
        if '-' in feature:
            feature = feature.replace('-', 'dash')
        _data = get_learn_settings(data, settings, feature=feature, **kwargs)
        df = make_dataframe(_data, feature=feature, instrs=test_instrs)
        dfs.append(df)

    # # MANUAL NO LEARN SETTINGS
    # for feature in ['randdash512', 'scat1dunderscoreo2']:
    #     data_no_learn = get_setting_data(data['none'], settings['none'], feature=feature, learn=0)
    #     df_rand = pd.DataFrame(list(zip(data_no_learn, test_instrs)), 
    #                                 columns=['triplet agreement', 'instrument'])
    #     df_rand['feature'] = feature
    #     df_rand['setting'] = 'none'
    #     dfs.append(df_rand)

    df = pd.concat(dfs)
    return df


def get_test_instruments():
    SEED_FILELIST = '/homes/cv300/Documents/timbre-metric/jasmp/seed_filelist.csv'
    df = pd.read_csv(SEED_FILELIST, index_col=0)
    test_instrs = list(df['instrument'].unique())
    instr_to_idx = {instr: idx for idx, instr in enumerate(test_instrs)}
    return test_instrs, instr_to_idx