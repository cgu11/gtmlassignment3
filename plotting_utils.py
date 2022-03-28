

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_clusters(ax, component1, component2, df, name):
    y_colors = sns.color_palette('hls', len(np.unique(df['y'])))
    c_colors = sns.color_palette('hls', len(np.unique(df['c'])))
    if len(np.unique(df['y'])) == len(np.unique(df['c'])):
      c_colors = y_colors
    sns.scatterplot(x=component1, y=component2, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax[0])
    sns.scatterplot(x=component1, y=component2, hue='c', palette=c_colors, data=df, legend='full', alpha=0.3, ax=ax[1])


    ax[0].set_title('True Clusters represented with {}'.format(component1[:-1].upper()))
    ax[1].set_title('{} Clusters represented with {}'.format(name.upper(), component1[:-1].upper()))


    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)


def plot_components(component1, component2, df, name):
    plt.figure()
    colors = sns.color_palette('hls', len(np.unique(df['y'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=colors, data=df, legend='full', alpha=0.3)


    plt.annotate('', xy=(np.std(df[component1]), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df[component2])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))


    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)


def plot_multiple_random_runs(x_axis, y_axis, label):
    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, '-o', markersize=1, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def plot_ic_bars(ic, ic_label, cv_types, k_range, ax):

    bar_width = 0.2  # width of a bar

    for i, cv_type in enumerate(cv_types):
        x = k_range + bar_width / 2 * (i - 2)  # compute x
        ax.bar(x, ic[i * len(k_range):(i + 1) * len(k_range)], width=bar_width, label=cv_type)

    ax.set_xticks(k_range)
    ax.set_ylim([ic.min() * 1.01 - .01 * ic.max(), ic.max()])
    ax.legend(loc='best')

    set_axis_title_labels(ax=ax, title='EM - Choosing k with the {} method'.format(ic_label),
                          x_label='Number of components k', y_label='{} score'.format(ic_label))


def save_figure(title):
    plt.savefig(title)
    plt.close()


def save_figure_tight(title):
    plt.tight_layout()
    save_figure(title)


def set_axis_title_labels(ax, title, x_label, y_label):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def set_plot_title_labels(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
