import io

import dendropy
import matplotlib.pyplot as plt
import networkx as nx
from Bio import Phylo
from matplotlib.colors import ListedColormap

import seaborn as sns
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout


def plot_cn_profile(cnp, ax=None, **kwargs):
    """
    Plot copy number profile.
    Parameters
    ----------
    cnp: np.ndarray, copy number profile of shape (n_cells, n_bins)
    ax: matplotlib.axes.Axes, axis to plot on (default: None)
    kwargs: keyword arguments, passed to plt.imshow
    """

    # color: integer map 0 to blue, 1 to light blue, 2 to white, 3 to light red, 4 to red and 5+ to dark red
    vmax = kwargs.get('vmax', np.max(cnp).astype(int))
    integer_cmap = create_integer_colormap(vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cnp, aspect="auto", cmap=integer_cmap, vmin=0, vmax=vmax, interpolation='none')
    ax.set_xlabel(kwargs.get('xlabel', 'bins'))
    ax.set_ylabel(kwargs.get('ylabel', 'cells'))
    ax.set_title(kwargs.get('title', 'Copy number profile'))
    plt.colorbar(im, label='state', ax=ax, ticks=range(vmax + 1))
    return ax

def plot_cell_pairwise_heatmap(matrix, ax=None, label=None, full=False, **kwargs):
    """
    Plot heatmap for pairs of cells. If full is False, only the upper triangle is shown.
    Parameters
    ----------
    matrix: np.ndarray, pairwise matrix of shape (n_cells, n_cells) (e.g. likelihoods, distances, etc.)
    ax: matplotlib.axes.Axes, axis to plot on (default: None)
    full: bool, whether to show the full matrix or only the upper triangle (default: False)
    kwargs: keyword arguments, passed to sns.heatmap
    """
    if type(matrix) == dict:
        # convert to full matrix
        n_cells = max(max(k) for k in matrix.keys()) + 1
        full_matrix = np.zeros((n_cells, n_cells))
        for (i, j), v in matrix.items():
            full_matrix[i, j] = v
            full_matrix[j, i] = v
        matrix = full_matrix
    if ax is None:
        fig, ax = plt.subplots()
    # mask the lower triangle
    if full:
        mask = None
    else:
        mask = np.tril(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, ax=ax, cmap=kwargs.get('cmap', 'viridis'),
                cbar_kws={"label": kwargs.get('cbar_label', 'Value')},
                square=kwargs.get('square', True),
                xticklabels=kwargs.get('xticklabels', True),
                yticklabels=kwargs.get('yticklabels', True),
                **{k: v for k, v in kwargs.items() if k not in ['cmap', 'cbar_label', 'square', 'xticklabels', 'yticklabels']})
    ax.set_title(kwargs.get('title', 'Pairwise Heatmap' if label is None else f'{label}'))
    ax.set_xlabel(kwargs.get('xlabel', 'Cells'))
    ax.set_ylabel(kwargs.get('ylabel', 'Cells'))
    return ax


def create_integer_colormap(vmax=11):
    vmax = min(11, np.max(vmax)) # limit to 11 colors
    # Define the colors for each integer
    colors = [
        '#3182BD',  # blue
        '#9ECAE1',  # light blue
        '#CCCCCC',  # grey
        '#FDCC8A',  # light orange
        '#FC8D59',  # orange
        '#E34A33',  # red
        '#B30000',  # dark red
        '#980043',  # dark red
        '#DD1C77',  # dark red
        '#DF65B0',  # dark red
        '#C994C7',  # dark red
        '#D4B9DA'  # dark red
    ]

    # Create the colormap
    return ListedColormap(colors[:(vmax + 1)])


# Example usage:
def plot_integer_matrix(matrix):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the colormap
    cmap = create_integer_colormap()

    # Plot with imshow
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=5)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    return fig, ax

def draw_graph(G: nx.DiGraph, save_path=None, node_color: str | list = '#1f78b4', epsilon=None,
               node_size=300, figsize=(5, 5), dpi=100, ax=None, edge_label_pos=0.5, num_cells: dict = None, **kwargs) -> dict:
    with_labels = kwargs.get('with_labels', True)
    layout = kwargs.get('layout', 'dot')
    pos = graphviz_layout(G, prog=layout)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi) if ax is None else (0, ax)
    # draw without labels if num_cells is given, then draw with labels separately
    nx.draw(G, pos=pos, with_labels=with_labels if num_cells is None else False,
            ax=ax,
            node_size=node_size,
            node_color=node_color)
    if num_cells is not None:
        assert all([n in num_cells for n in G.nodes])
        labels = {n: f"{n}\n({num_cells[n]})" for n in G.nodes()}
        nx.draw(G, pos=pos, labels=labels, ax=ax, node_size=node_size, font_size=8, node_color=node_color)
    if epsilon is not None:
        if type(epsilon) is np.ndarray:
            edge_labels = {e: round(epsilon[e].item(), 4) for e in G.edges()}
        else:
            edge_labels = {e: round(epsilon[e], 4) for e in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, label_pos=edge_label_pos)
    if save_path is not None:
        fig.savefig(save_path, format="png")

    return fig, ax


# Example:

if __name__ == '__main__':
    # enable interactive mode
    plt.ion()

    # matrix = np.array([
    #     [0, 1, 2, 3],
    #     [4, 5, 5, 2],
    #     [3, 4, 1, 0]
    # ])
    # fig, ax = plot_integer_matrix(matrix)
    # plt.show()

    fig, ax = plt.subplots()
    plot_cn_profile(np.random.randint(0, 6, (10, 100)), ax=ax)  # Plot a random copy number profile
    plt.show()


def plot_tree_phylo(tree: str | dendropy.Tree,
                    out_dir,
                    filename='tree',
                    show=False, save=True,
                    title=None):
    """
    Plot a tree using Bio.Phylo and Matplotlib.
    """
    if type(tree) is dendropy.Tree:
        true_tree_nw = tree.as_string("newick")
    else:
        true_tree_nw = tree

    handle = io.StringIO(true_tree_nw)
    bio_tree = Phylo.read(handle, "newick")

    fig, ax = plt.subplots(figsize=(6, 4))
    if title is not None:
        ax.set_title(title)
    Phylo.draw(bio_tree, axes=ax)
    if show:
        plt.show()
    if save:
        fig.savefig(out_dir + '/' + filename + ".png", format="png")
