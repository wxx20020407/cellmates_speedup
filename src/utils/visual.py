import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
import numpy as np

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
    vmax = kwargs.get('vmax', min(11, np.max(cnp)))
    integer_cmap = create_integer_colormap(vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cnp, aspect="auto", cmap=integer_cmap, vmin=0, vmax=vmax)
    ax.set_xlabel(kwargs.get('xlabel', 'bins'))
    ax.set_ylabel(kwargs.get('ylabel', 'cells'))
    ax.set_title(kwargs.get('title', 'Copy number profile'))
    plt.colorbar(im, label='state', ax=ax, ticks=range(vmax + 1))
    return ax

def create_integer_colormap(vmax=11):
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
    return ListedColormap(colors[:vmax + 1])


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

