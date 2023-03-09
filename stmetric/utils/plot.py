import timbremetrics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.cm as cm

from librosa.feature import spectral_centroid
import essentia.standard as es

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import Union

from stmetric.data import DissimilarityTripletRatioDataset

class TimbreMetricPlots:

    def __init__(self, 
                 metric_meta: dict,
                 datasets: Union[str] = ['Barthet2010', 'Grey1977', 'Grey1978', 'Iverson1993_Whole', 'Lakatos2000_Harm', 'Lakatos2000_Perc', 'Patil2012_A3', 'Patil2012_DX4', 'Patil2012_GD4', 'Siedenburg2016_e2set2', 'Siedenburg2016_e3', 'Vahidi2020', 'Zacharakis2014_greek', 'Zacharakis2014_english', 'Zacharakis2014_merged'],
                 ):
        self.metric_meta = metric_meta 
        self.datasets = datasets

        self.get_dataset_stats()

        self.get_dataset_features()

        self.create_df()
        sns.set_theme(); sns.set(rc={"font.size": 18, 'figure.figsize':(15,8)})

    def get_dataset_stats(self):
        datasets = [(ds, 
            len(timbremetrics.utils.load_dissimilarity_matrix(ds)),
            timbremetrics.utils.load_dissimilarity_matrix(ds).mean(),
            timbremetrics.utils.load_dissimilarity_matrix(ds).std()) 
            for ds in timbremetrics.list_datasets() if ds in self.datasets]
        df = pd.DataFrame(datasets, columns=['dataset', '# stimuli', 'mean', 'std'])
        self.df_stats = df

    def get_dataset_features(self):
        def log_attack_time(audio):
            env = es.Envelope()(audio)
            lat, _, _ = es.LogAttackTime(startAttackThreshold=0)(env)
            return lat

        def ds_lats(dataset):
            lat = np.array([log_attack_time(a['audio']) for a in timbremetrics.get_audio(dataset)])
            return lat

        def ds_centroids(dataset):
            centroid = np.array([spectral_centroid(a['audio'], sr=44100).mean() 
                        for a in timbremetrics.get_audio(dataset)])
            return centroid

        self.ds_features = {ds: {'centroids': ds_centroids(ds), 'lats': ds_lats(ds)} for ds in self.datasets}

    def set_datasets(self, datasets):
        self.datasets = datasets

    def create_df(self):
        features = list(self.metric_meta.keys())
        def process_df(csv_path, feature):
            df = pd.read_csv(csv_path).sort_values('dataset')
            df['feature'] = feature
            df['marker'] = np.where(df['triplet agreement'] >= df['triplet agreement id'], '+', '-')
            return df

        def average_metric(df):
            df_avg = df.groupby('feature')['triplet agreement', 'triplet agreement id'].mean()
            df_avg['marker'] = np.where(df_avg['triplet agreement'] >= df_avg['triplet agreement id'], '+', '-')
            df_avg['dataset'] = "average"
            df = df.append(df_avg.reset_index())
            return df
        
        df = pd.concat([process_df(self.metric_meta[f]['csv'], f) for f in features])
        df_all = average_metric(df)
        df_all = df
        self.df_all = df_all


    def plot_dataset_counts(self):
        df = self.df_stats
        sns.set_theme(style="whitegrid"); sns.set(rc={'figure.figsize':(11.7,8.27), "font.size": 18})
        ax = sns.barplot(y="dataset", x="# stimuli", data=df, palette="tab10").set_title('Timbre Dissimilarity Datasets')
        plt.show()

    def plot_dataset_stats(self):
        df = self.df_stats
        df_std, df_mean = pd.DataFrame(df[['std', 'dataset']]), pd.DataFrame(df[['mean', 'dataset']])
        df_std['type'], df_mean['type'] = 'std', 'mean'
        df_std.columns, df_mean.columns = ['value', 'dataset', 'stat'], ['value', 'dataset', 'stat']

        df_stats = pd.concat((df_mean, df_std))

        sns.set_theme(style="whitegrid"); sns.set(rc={"font.size": 18})
        ax = sns.catplot(y="dataset", x="value", data=df_stats, hue="stat", s=10)
        ax.fig.set_size_inches(10,5)    
        plt.show()

    def plot_dissimilarity_histograms(self):
        fig, axes = plt.subplots(4, 4, figsize=(15,12), sharex=True)
        axes = axes.flatten()
        for i, ds in enumerate(self.datasets):
            dissim = timbremetrics.utils.load_dissimilarity_matrix(ds)    
            sns.histplot(dissim[dissim.nonzero()], 
                        binwidth=0.05, 
                        ax=axes[i],
                        kde=True).set(title=ds)
        fig.delaxes(axes[-1]);
        fig.tight_layout()
        plt.show()

    def plot_dataset_features(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = cm.hsv(np.linspace(0, 1, len(self.datasets)))
        for i, ds in enumerate(self.datasets):
            lats = self.ds_features[ds]['lats'] 
            centroids = self.ds_features[ds]['centroids']
            p = plt.scatter(lats, 
                            centroids, 
                            color=colors[i], 
                            label=ds)
            confidence_ellipse(lats, 
                               centroids, 
                               ax, 
                               n_std=1, 
                               edgecolor=colors[i], 
                               label=ds,
                               linewidth=3)
            p.remove()

        plt.xlabel("log-attack time")
        plt.ylabel("spectral centroid")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_metric_bars(self, palette=sns.color_palette("Set2")):
        df = self.df_all

        scatter(df[df['feature'].str.contains('MFCC')], -16)
        scatter(df[df['feature'].str.contains('TS')], -5)
        scatter(df[df['feature'].str.contains('JTFS')], 5)
        scatter(df[df['feature'].str.contains('OPEN-L3')], 16)
        ax = sns.barplot(x="dataset", 
                         y="triplet agreement id",      
                         data=df, 
                         hue="feature",
                         palette=palette)
        ax.set(yticks=np.linspace(0.5, 1, 11), ylim=(0.4, None))
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
        plt.show()


    def plot_tsne(self, feature_path='/homes/cv300/Documents/timbre-metric/datasets/jtfs_Q3_F2', feature_type='jtfs'):
        ds = DissimilarityTripletRatioDataset(datasets=self.datasets,
                                              k=8,
                                              feature_type=feature_type,
                                              feature_path=feature_path)

        ds_features = ds.features
        ds_to_idx = {ds: i for i, ds in enumerate(ds_features.keys())}

        X = torch.cat([f for f in ds_features.values()])
        Y = torch.cat([torch.tensor(ds_to_idx[ds]).repeat(len(feat)) 
                    for ds, feat in ds_features.items()]).numpy()

        # pca = PCA(n_components=2)
        # pca.fit(X.numpy())
        # print(pca.explained_variance_ratio_)

        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = cm.hsv(np.linspace(0, 1, len(self.datasets)))
        for ds, idx in ds_to_idx.items():
            X = X_embedded[np.where(Y == idx)[0], :]
            p = plt.scatter(X[:, 0], X[:, 1], color=colors[idx], label=ds)
            confidence_ellipse(X[:, 0], 
                            X[:, 1], 
                            ax,
                            n_std=1, 
                            edgecolor=colors[idx], 
                            label=ds,
                            linewidth=3)
            p.remove()

        plt.legend()
        plt.title(f"Confidence Ellipses: T-SNE of {feature_type} - σ = 1")
        plt.tight_layout()
        plt.show()
    
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', label=None, **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse.set(label=label)
    return ax.add_patch(ellipse)


def scatter(data, offset_val):
    offset = lambda p: transforms.ScaledTranslation(p/72.,0,  plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData
    markers = {'+': 6, '-': 7}
    ax = sns.scatterplot(y='triplet agreement', 
                x="dataset", 
                data=data, 
                style='marker',
                markers=markers,
                color='k',
                s=50,
                zorder=10,
                linewidths=5,
                transform=trans+offset(offset_val),
                legend=False)