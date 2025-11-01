import copy
import itertools
import os
import random
from unittest.mock import MagicMock

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, float64
import dendropy as dpy

from cellmates import ROOT_DIR
from cellmates.inference.em import EM
import cellmates.inference.em as em_module
from cellmates.utils import tree_utils, math_utils, visual


def create_output_test_folder(sub_folder_name=None) -> str:
    """
    Create a test output folder based on the current pytest test name.
    The folder will be created under the "output" directory in the general tests folder.
    Returns the path to the created folder.
    Returns:
        str: Path to the created test output folder.
    """
    test_context = os.environ.get('PYTEST_CURRENT_TEST').split(':')
    test_file = test_context[0].split(' ')[0]
    test_dir = test_file.replace('.py', '')
    test_name = test_context[4].split(' ')[0]
    test_name = test_name.replace(']', '').replace('[', '_').replace('/', '_')
    test_name += '' if sub_folder_name is None else f'/{sub_folder_name}'
    test_folder = os.path.join(ROOT_DIR, 'output', test_dir, test_name)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    return test_folder

def get_expected_changes(cnps, tree_nx, cell_pairs=None, ancestor_labeling: dict=None)-> tuple[dict, dict]:
    """
    Compute the expected changes between cell pairs w.r.t. their lowest common ancestor (LCA) in the tree based.
    I.e. constructs a quadruplet for each cell pair (root, LCA, v, w) and computes the changes D and D' between:
    cnp_root and cnp_LCA (D_ru, Dp_ru), cnp_v and cnp_LCA (D_uv, Dp_uv), cnp_w and cnp_LCA (D_uw, Dp_uw).
    Returns: D, Dp - dicts with keys specified by the cell pairs and each value is: (D_ru, D_uv, D_uw) and (Dp_ru, Dp_uv, Dp_uw).
    """
    n_cells, n_sites = cnps.shape
    cell_pairs = cell_pairs if cell_pairs is not None else list(itertools.combinations(range(n_cells), r=2))
    D = {}
    Dp = {}
    root = [v for v in tree_nx.nodes() if tree_nx.in_degree(v) == 0][0]
    for i, pair in enumerate(cell_pairs):
        v,w = pair
        lca = tree_utils.get_lowest_common_ancestor(tree_nx, v, w)
        D_ru = math_utils.compute_cn_changes(cnps, pairs=[(root, lca)])[0]
        D_uv = math_utils.compute_cn_changes(cnps, pairs=[(v, lca)])[0]
        D_uw = math_utils.compute_cn_changes(cnps, pairs=[(w, lca)])[0]
        Dp_ru = n_sites - D_ru
        Dp_uv = n_sites - D_uv
        Dp_uw = n_sites - D_uw
        D_pair = np.array([D_ru, D_uv, D_uw])
        Dp_pair = np.array([Dp_ru, Dp_uv, Dp_uw])
        D[pair] = D_pair
        Dp[pair] = Dp_pair
    return D, Dp

def get_expected_distances(D:dict, Dp:dict, n_states, cell_pairs=None)-> tuple[dict, dict]:
    """
    Compute the expected distances and expected pairwise distances between cell pairs from the expected changes D and Dp.
    Parameters
    ----------
    D: dict, expected changes between cell pairs
    Dp: dict, expected non-changes between cell pairs
    n_states: int, number of copy number states
    cell_pairs: list of tuples, cell pairs to compute distances for (default: all pairs in D)
    Returns
    -------
    expected_distances: dict, expected distances between cell pairs
    expected_pairwise_distances: dict, expected pairwise distances between cell pairs
    """
    cell_pairs = cell_pairs if cell_pairs is not None else list(D.keys())
    expected_distances = {}
    expected_pairwise_distances = {}
    for i, (v, w) in enumerate(cell_pairs):
        D_pair = D[v, w]
        Dp_pair = Dp[v, w]
        expected_distances[v, w] = math_utils.l_from_p(D_pair / Dp_pair, n_states)
        expected_pairwise_distances[v, w] = expected_distances[v, w][1] + expected_distances[v, w][2]

    return expected_distances, expected_pairwise_distances


def get_expected_psi(param, obs_model):
    raise NotImplementedError


def get_marginals_from_cnp(cnp: ndarray, n_states: int, noise=0.0,
                           cut_max=True)-> tuple[ndarray, ndarray]:
    """
    Get the one slice and two slice marginals from one copy number profile.
    """
    M = cnp.shape[0]
    cnp = cnp.astype(int)
    if cut_max:
        cnp[cnp >= n_states] = n_states - 1
    one_slice_marginals = np.zeros((M, n_states))
    two_slice_marginals = np.zeros((M - 1, n_states, n_states))
    for m in range(M):
        one_slice_marginals[m, cnp[m]] = 1.0
        if m < M - 1:
            two_slice_marginals[m, cnp[m], cnp[m + 1]] = 1.0
    if noise > 0:
        one_slice_marginals = (1 - noise) * one_slice_marginals + noise / n_states
        two_slice_marginals = (1 - noise) * two_slice_marginals + noise / (n_states ** 2)
    return one_slice_marginals, two_slice_marginals

def run_ideal_cellmates_em_from_cnps(x, cnps, tree_nx, cell_pairs, n_states,
                                    evo_model, obs_model, psi_init,
                                    max_iter=20, rtol=1e-3) -> list:
    """
    Run the ideal Cellmates EM algorithm on the given copy number profiles and tree.
    Parameters
    ----------
    x: ndarray, observed data of shape (n_cells, n_bins, n_features)
    cnps: ndarray, copy number profiles of shape (n_cells, n_bins)
    tree_nx: networkx.DiGraph, tree structure
    cell_pairs: list of tuples, cell pairs to run EM on
    n_states: int, number of copy number states
    evo_model: EvolutionaryModel, evolutionary model to use
    Returns
    -------
    results: list of EM results for each cell pair
    """
    D, Dp = get_expected_changes(cnps, tree_nx, cell_pairs)

    # Set up Mocking for EM algorithm
    evo_model_temp = copy.deepcopy(evo_model)
    evo_model_temp.new = MagicMock(return_value=evo_model)  # bypass new model creation to enable mocking

    # Run Mocked EM algorithm on each cell pair
    results = []
    for i, (v,w) in enumerate(cell_pairs):
        theta_init = np.array([0.25, 0.25, 0.25])
        pC1_v = get_marginals_from_cnp(cnps[v], n_states)[0]
        pC1_w = get_marginals_from_cnp(cnps[w], n_states)[0]
        evo_model.get_one_slice_marginals = MagicMock(return_value=(pC1_v, pC1_w))
        evo_model.multi_chr_expected_changes = MagicMock(return_value=(D[v,w], Dp[v,w], -1.0))
        res_vw = em_module.fit_quadruplet(v, w, x,
                                          theta_init=theta_init, psi_init=psi_init,
                                          evo_model_template=evo_model_temp,
                                          obs_model_template=obs_model,
                                          max_iter=max_iter, rtol=rtol)
        results.append(res_vw)

    return results, D, Dp


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    dpy.utility.GLOBAL_RNG.seed(seed)


def plot_diagnostics(diagnostics: dict, out_dir: str) -> None:
    """
    Plot diagnostics from EM algorithm.
    """
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].plot(diagnostics['loglikelihoods'], marker='o')
    ax[0].set_title('Log-likelihoods over iterations')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Log-likelihood')
    labels = [r"$\theta_{ru}$", r"$\theta_{uv}$", r"$\theta_{uw}$"]
    for i in range(3):
        thetas_i = [theta[i] for theta in diagnostics['thetas']]
        ax[1].plot(thetas_i, marker='o', color='C'+str(i), label=labels[i])
    ax[1].legend()
    ax[1].set_title('Theta parameters over iterations')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Theta')
    ax[2].plot(diagnostics['psis'], marker='o', color='green')
    ax[2].set_title('Psi parameters over iterations')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Psi')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'em_diagnostics.png'))

    return None