
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score, 
    completeness_score,
    homogeneity_score, 
    v_measure_score, 
    silhouette_samples, 
    silhouette_score
)
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from plotting_utils import save_figure_tight, plot_clusters, set_axis_title_labels, plot_ic_bars

from abc import abstractmethod, ABC

class ClusteringExperiment(ABC):
    def __init__(self, model_name, n_clusters, most_clusters, random_seed, name_param):
        self.name = model_name
        self.model = None
        self.n_clusters = n_clusters
        self.max_clusters = most_clusters
        self.seed = random_seed
        self.name_param = name_param




    @abstractmethod
    def plot_model_complexity(self, x, data_name):
        pass

    def run_experiment(self, train_x, test_x, train_y, data_name, add_complex):
        if add_complex:
            self.plot_model_complexity(train_x, data_name)

        self.train(train_x, train_y)
        self.vis_clusters(train_x, train_y, data_name)

        return self.clusters, self.predict(test_x)

    def train(self, x, y):
        self.clusters = self.model.fit_predict(x)
        self.model_analysis(x, y, self.clusters)

    def predict(self, x):
        return self.model.predict(x)

    def vis_clusters(self, x, y, data_name):
        ## PCA vis
        pca = PCA(n_components=2, random_state=self.seed)
        clust_pca = pca.fit_transform(x)

        ## TSNE vis
        tsne = TSNE(n_components=2, random_state=self.seed, learning_rate='auto', init='random')
        clust_tsne = tsne.fit_transform(x)

        n_classes = len(np.unique(y))

        # cloning model to make multiple uses of it
        model = clone(self.model)
        params = model.get_params()
        params[self.name_param] = n_classes
        model.set_params(**params)

        clusters = model.fit_predict(x)

        df = pd.DataFrame(clust_tsne, columns=['tsne1', 'tsne2'])
        df['pca1'] = clust_pca[:, 0]
        df['pca2'] = clust_pca[:, 1]
        df['y'] = y
        df['c'] = self.clusters

        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 8))
        plot_clusters(ax1, 'pca1', 'pca2', df, self.name)
        plot_clusters(ax2, 'tsne1', 'tsne2', df, self.name)

        save_figure_tight('{}_{}_clusters'.format(data_name, self.name))

    @staticmethod
    def model_analysis(x, y, clusters):
        print(f"Homogeneity: {homogeneity_score(y, clusters)}")
        print(f"Completeness: {completeness_score(y, clusters)}")
        print(f"V Measure: {v_measure_score(y, clusters)}")
        print(f"Adjusted Random: {adjusted_rand_score(y, clusters)}")
        print(f"Adjusted MI: {adjusted_mutual_info_score(y, clusters)}")
        print(f"Silhouette: {silhouette_score(x, clusters, metric='euclidean')}")


    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.set_model_param("n_clusters", n_clusters)

    def set_model_param(self, param, param_value):
        self.model.set_params(**{param: param_value})

    def get_model_param(self, param):
        return self.model.get_params()[param]

class KMeansExperiment(ClusteringExperiment):
    def __init__(self, n_clusters=2, most_clusters=10, random_seed=5555555):
        super(KMeansExperiment, self).__init__(model_name='kmeans', 
                                               n_clusters=n_clusters,
                                               most_clusters=most_clusters,
                                               name_param='n_clusters',
                                               random_seed=random_seed   
                                            )
        self.model = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=1000, random_state=random_seed)

    def plot_model_complexity(self, x, data_name):
        # silhouette complexity analysis plots adapted from
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        inertia, inertia_diff = [], []
        k_range = np.arange(1, self.max_clusters + 2)  

        clusters_list = []
        silhouette_averages = []
        silhouettes = []
        maximum_silhouettes = []
        # For each k in the range
        for k in k_range:

            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.seed)
            model.fit(x)
            inertia.append(model.inertia_)
            print('k = {} -->  inertia = {:.3f}'.format(k, inertia[-1]))

            if k > 1:
                inertia_diff.append(abs(inertia[-1] - inertia[-2]))
                clusters = model.predict(x)
                clusters_list.append(clusters)
                silhouette = silhouette_samples(x, clusters)
                
                silhouette_averages.append(silhouette_score(x, clusters))
                silhouettes.append(silhouette)
                maximum_silhouettes.append(np.max(silhouette))


        fig, ax = plt.subplots(2, math.ceil(self.max_clusters / 2), figsize=(24, 12))
        ax = ax.flatten()
        ax[0].plot(k_range, inertia, '-o', markersize=2, label='Inertia')
        ax[0].plot(k_range[1:], inertia_diff, '-o', markersize=2, label=r'Inertia |$\Delta$|')

        ax[0].legend(loc='best')
        set_axis_title_labels(ax[0], title='K-MEANS - K vs Inertia',
                                    x_label='Number of clusters', y_label='Inertia')

        for k, clusters, silhouette, silhouette_average, maximum_silhouette in zip(k_range[2:],clusters_list, silhouettes, silhouette_averages, maximum_silhouettes):
            ax[k-2].axvline(x=silhouette_average, color='red',linestyle='--')
            lb_y = 10
            for j in range(k-2):
                s_clust = silhouette[clusters==j]
                s_clust.sort()
                ub_y = lb_y + s_clust.shape[0]

                area_shade = plt.cm.nipy_spectral(float(j)/k)
                ax[k-2].fill_betweenx(np.arange(lb_y, ub_y), 0, s_clust, facecolor=area_shade, edgecolor=area_shade, alpha=0.75)
                ax[k-2].text(-0.1, lb_y + 0.5 * s_clust.shape[0], str(j))

                lb_y = ub_y + 10
            # Set title and labels
            set_axis_title_labels(ax[k-2], title='K-MEANS - Silhouette for k = {}'.format(k),
                                        x_label='Silhouette', y_label='Silhouette distribution per Cluster')

            # Clear the y axis labels and set the x ones
            ax[k-2].set_yticks([])
            ax[k-2].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # Set x and y limits
            ax[k-2].set_xlim(-0.2, 0.1 + round(maximum_silhouette, 1))
            ax[k-2].set_ylim(0, len(x) + (k + 1) * 10)

        save_figure_tight('{}_{}_model_complexity'.format(data_name, self.name))

class EMExperiment(ClusteringExperiment):
    def __init__(self, n_clusters=2, covariance='full', most_clusters=10, random_seed=42):
        super(EMExperiment, self).__init__(model_name="em",
                                           n_clusters=n_clusters,
                                           most_clusters=most_clusters,
                                           random_seed=random_seed,
                                           name_param="n_components",
        )
        self.model = GaussianMixture(n_components=n_clusters, covariance_type=covariance, max_iter=1000, n_init=10, init_params='random',random_state=random_seed)

    def plot_model_complexity(self, x, dataset):
        # GMM model selection code adapted from
        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py

        aic, bic = [], []
        k_range = np.arange(2, self.max_clusters + 1)

        cv_types = ['spherical', 'tied', 'diag', 'full'] 

        for cv_type in cv_types:
            for k in k_range:

                gmm = GaussianMixture(n_components=k, covariance_type=cv_type, max_iter=1000,
                                      n_init=10, init_params='random', random_state=self.seed)
                gmm.fit(x)
                aic.append(gmm.aic(x))
                bic.append(gmm.bic(x))

                print('cv = {}, k = {} --> aic = {:.3f}, bic = {:.3f}'.format(cv_type, k, aic[-1], bic[-1]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        plot_ic_bars(np.array(aic), 'AIC', cv_types, k_range, ax1)
        plot_ic_bars(np.array(bic), 'BIC', cv_types, k_range, ax2)

        save_figure_tight('{}_{}_model_complexity'.format(dataset, self.name))





def run_clustering_experiments(train_x, test_x, train_y, data_name, complexity_flag, **model_kwargs):
    print("KMEANS")

    kmeans = KMeansExperiment(n_clusters=model_kwargs['kmeans_n'])
    km_clusters = kmeans.run_experiment(train_x, test_x, train_y, data_name, complexity_flag)

    print("EM")
    em = EMExperiment(n_clusters=model_kwargs["em_n"], covariance=model_kwargs["em_cov"])
    em_clusters = em.run_experiment(train_x, test_x, train_y, data_name, complexity_flag)

    return km_clusters, em_clusters

