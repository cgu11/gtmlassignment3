from abc import abstractmethod, ABC

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from plotting_utils import *

from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import SparseRandomProjection

from scipy.stats import kurtosis

class DimReductionExperiment(ABC):

    def __init__(self, name, n_components, random_seed):
        self.name = name
        self.n_components = n_components
        self.seed = random_seed

        self.model = None
    
    def run_experiment(self, train_x, test_x, train_y, data_name, complexity_plot):
        if complexity_plot:
            self.plot_complexity(train_x, data_name)

        x_train_dr, mse = self.train(train_x)
        x_test_dr = self.model.transform(test_x)

        print(f"{self.name}, {data_name} recon: {mse}")

        self.plot_components(x_train_dr, train_y, data_name)

        return x_train_dr, x_test_dr

    @abstractmethod
    def plot_complexity(train_x, data_name):
        pass

    def reconstruction_loss(self, x, x_dr):
        x_approx = self.model.inverse_transform(x_dr)

        return np.mean((x - x_approx)**2)

    def transform(self, x):
        return self.model.transform(x)

    def set_n_components(self, n):
        self.n_components = n
        self.model.set_params(**{'n_components':n})

    def train(self, x):
        x_dr = self.model.fit_transform(x)
        mse = self.reconstruction_loss(x, x_dr)
        return x_dr, mse

    def plot_components(self, x_dr, y, data_name):
        c1 = f'{self.name}-1'
        c2 = f'{self.name}-2'

        df_vis = pd.DataFrame(x_dr[:,:2], columns=[c1, c2])
        df_vis['y'] = y

        plot_components(c1, c2, df_vis, self.name)
        save_figure(f"{data_name}_{self.name}_components")
        

class PCAExperiment(DimReductionExperiment):

    def __init__(self, n_components=2, random_seed=5555555):
        super(PCAExperiment, self).__init__(name='pca',
                                            n_components=n_components,
                                            random_seed=random_seed
                                        )
        self.model = PCA(n_components=n_components, svd_solver='randomized', random_state=random_seed)

    def plot_complexity(self, train_x, data_name):
        k_range = np.arange(
            1, 
            train_x.shape[1]+1)
        pca = PCA(svd_solver='randomized', random_state=self.seed)
        pca.fit(train_x)

        explained_variance = np.sum(pca.explained_variance_ratio_[:self.n_components])
        print('Explained variance [n components = {}]= {:.3f}'.format(self.n_components, explained_variance))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

        ax1.bar(k_range, np.cumsum(pca.explained_variance_ratio_), color='red')
        set_axis_title_labels(ax1, title='PCA - Choosing k with the Variance method',
                                    x_label='Number of components k', y_label='Cumulative Variance (%)')

        ax2.bar(k_range, pca.explained_variance_ratio_, color='cyan')
        set_axis_title_labels(ax2, title='PCA - Eigenvalues distributions',
                                    x_label='Number of components k', y_label='Variance (%)')

        save_figure_tight('{}_pca_model_complexity'.format(data_name))

class ICAExperiment(DimReductionExperiment):
 
    def __init__(self, n_components=2, random_seed=555555):
        super(ICAExperiment, self).__init__(name='ica', n_components=n_components, random_seed=random_seed)
        self.model = FastICA(n_components=n_components, tol=0.01, max_iter=1000, random_state=random_seed)

    def plot_complexity(self, x, dataset):
        average_kurtosis = []  
        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        for k in k_range:

            ica = FastICA(n_components=k, tol=0.01, max_iter=1000, random_state=self.seed)
            ica.fit(x)

            components_kurtosis = kurtosis(ica.components_, axis=1, fisher=False)
            average_kurtosis.append(np.min(components_kurtosis))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.plot(k_range, average_kurtosis,  '-o', markersize=1, label='Kurtosis')

        set_axis_title_labels(ax1, title='ICA - Minimum Kurtosis method',
                                    x_label='Number of components k', y_label='Minimum Kurtosis')

        self.model.fit(x)

        if x.shape[1] < 100:
            x_ticks = np.arange(1, self.n_components + 1)
        else:
            x_ticks = np.arange(0, self.n_components + 1, 50)
            x_ticks[0] = 1

        components_kurtosis = kurtosis(self.model.components_, axis=1, fisher=False)

        ax2.bar(np.arange(1, self.n_components + 1), components_kurtosis, color='cyan')

        ax2.set_xticks(x_ticks)
        set_axis_title_labels(ax2, title='ICA - Components Kurtosis Distribution',
                                    x_label='Independent component', y_label='Kurtosis')

        save_figure_tight('{}_ica_model_complexity'.format(dataset))


class KernelPCExperiment(DimReductionExperiment):

    def __init__(self, n_components=2, kernel='rbf', random_seed=5555555):
        super(KernelPCExperiment, self).__init__(name='kpca', n_components=n_components, random_seed=random_seed)
        self.model = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True,
                               random_state=random_seed, n_jobs=-1)
        self.kernel = kernel

    def plot_complexity(self, x, dataset):

        k_range = np.arange(1, x.shape[1] + 1)  # range of number of components k to plot over
        kernels = ['rbf', 'poly', 'sigmoid', 'cosine']  # kernels to plot with

        fig, ax = plt.subplots(2, 4, figsize=(15, 10))
        ax = ax.ravel()

        for i, kernel in enumerate(kernels):

            kpca = KernelPCA(n_components=x.shape[1], kernel=kernel, random_state=self.seed, n_jobs=-1)
            kpca.fit(x)


            explained_variance_ratio = kpca.eigenvalues_ / np.sum(kpca.eigenvalues_)
            explained_variance = np.sum(explained_variance_ratio[:self.n_components])
            print('Kernel = {} - Explained variance [n components = {}]= {:.3f}'.format(kernel,
                                                                                        self.n_components,
                                                                                        explained_variance))

            ax[2*i].bar(k_range, np.cumsum(explained_variance_ratio), color='red', label=kernel)

            ax[2*i].legend(loc='best')
            set_axis_title_labels(ax[2*i], title='KPCA - Choosing k with the Variance method',
                                        x_label='Number of components k', y_label='Cumulative Variance (%)')

            ax[2*i+1].bar(k_range, explained_variance_ratio, color='cyan', label=kernel)

            ax[2*i+1].legend(loc='best')
            set_axis_title_labels(ax[2*i+1], title='KPCA - Eigenvalues distributions',
                                        x_label='Number of components k', y_label='Variance (%)')

        save_figure_tight('{}_kpca_model_complexity'.format(dataset))



class RPExperiment(DimReductionExperiment):

    def __init__(self, n_components=2, random_runs=10, random_seed=555555):
        super(RPExperiment, self).__init__(name='rp', n_components=n_components, random_seed=random_seed)
        self.model = SparseRandomProjection(n_components=n_components, random_state=random_seed)
        self.random_runs = random_runs

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):
        if perform_model_complexity:
            self.plot_complexity(x_train, dataset)

        train_errors = []
        x_train_reduced = np.zeros((x_train.shape[0], self.n_components))
        x_test_reduced = np.zeros((x_test.shape[0], self.n_components))

        for seed in range(self.seed, self.seed + self.random_runs):

            self.model = SparseRandomProjection(n_components=self.n_components, random_state=seed)

            x_reduced, mse = self.train(x_train)

            x_train_reduced += x_reduced
            train_errors.append(mse)

            x_reduced = self.reduce(x_test)
            x_test_reduced += x_reduced

        print('Reconstruction error = {:.3f} with std = {:.3f}'.format(np.mean(train_errors), np.std(train_errors)))

        x_train_reduced /= self.random_runs
        x_test_reduced /= self.random_runs

        self.plot_components(x_train_reduced, y_train, dataset)

        return x_train_reduced, x_test_reduced

 
    def plot_complexity(self, x, dataset):


        mse_random_runs = [] 

        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        for seed in range(self.seed, self.seed + self.random_runs):

            mse = []  
            for k in k_range:

                rp = SparseRandomProjection(n_components=k, random_state=seed)
                x_reduced = rp.fit_transform(x)

                P_inv = np.linalg.pinv(rp.components_.toarray()) 
                x_reconstructed = (P_inv @ x_reduced.T).T  
                mse.append(np.mean((x - x_reconstructed) ** 2))

            mse_random_runs.append(mse)  
        np.set_printoptions(precision=2)
        print('k = [2, ..., {}] --> \nReconstruction errors = {}'.format(k_range[-1], np.mean(mse_random_runs, axis=0)))

        plt.figure()
        plot_multiple_random_runs(k_range, mse_random_runs, 'MSE')
        set_plot_title_labels(title='RP - Choosing k with the Reconstruction Error',
                                    x_label='Number of components k', y_label='MSE')

        save_figure('{}_rp_model_complexity'.format(dataset))

    def reconstruction_loss(self, x, x_reduced):
        pseudo_inverse = np.linalg.pinv(self.model.components_.toarray()) 
        recon_x = (pseudo_inverse @ x_reduced.T).T  
        mse = np.mean((x - recon_x) ** 2)
        return mse

def run_dr_experiments(train_x, test_x, train_y, data_name, complexity, **model_kwargs):
    data_return = {}
    print('PCA')
    pca = PCAExperiment(n_components=model_kwargs['pca_n'])
    pca_x_train, pca_x_test = pca.run_experiment(train_x, test_x, train_y, data_name, complexity)
    data_return['PCA'] = {'train': pca_x_train, 'test': pca_x_test}
    print('ICA')
    ica = ICAExperiment(n_components=model_kwargs['ica_n'])
    ica_x_train, ica_x_test = ica.run_experiment(train_x, test_x, train_y, data_name, complexity)
    data_return['ICA'] = {'train': ica_x_train, 'test': ica_x_test}

    print("Kernel PCA")
    kpca = KernelPCExperiment(n_components=model_kwargs['kpca_n'], kernel=model_kwargs['kpca_kernel'])
    kpca_x_train, kpca_x_test = kpca.run_experiment(train_x, test_x, train_y, data_name, complexity)
    data_return['KPCA'] = {'train': kpca_x_train, 'test': kpca_x_test}

    print("RP")
    rp = RPExperiment(n_components=model_kwargs['rp_n'], random_runs=model_kwargs['rp_runs'])
    rp_x_train, rp_x_test = rp.run_experiment(train_x, test_x, train_y, data_name, complexity)
    data_return['RP'] = {'train': rp_x_train, 'test': rp_x_test}

    return data_return