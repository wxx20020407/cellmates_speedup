import itertools
import os

import numpy as np
from numpy import ndarray, dtype, float64

from cellmates import ROOT_DIR
from cellmates.utils import tree_utils, math_utils


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

def get_expected_changes(cnps, tree_nx, cell_pairs=None)-> tuple[ndarray, ndarray]:
    """
    Compute the expected changes between cell pairs w.r.t. their lowest common ancestor (LCA) in the tree based.
    I.e. constructs a quadruplet for each cell pair (root, LCA, v, w) and computes the changes D and D' between:
    cnp_root and cnp_LCA (D_ru, Dp_ru), cnp_v and cnp_LCA (D_uv, Dp_uv), cnp_w and cnp_LCA (D_uw, Dp_uw).
    Returns D, Dp of shape (n_pairs, 3) where each row corresponds to a cell pair and columns are: (D_ru, D_uv, D_uw) and (Dp_ru, Dp_uv, Dp_uw).
    """
    n_cells, n_sites = cnps.shape
    cell_pairs = cell_pairs if cell_pairs is not None else list(itertools.combinations(range(n_cells), r=2))
    n_pairs = len(cell_pairs)
    D = np.zeros((n_pairs, 3))
    Dp = np.zeros((n_pairs, 3))
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
        D[i] = D_pair
        Dp[i] = Dp_pair
    return D.squeeze(), Dp.squeeze()

def get_expected_distances(D: np.ndarray, Dp:np.ndarray, n_states, cell_pairs=None):
    n_cells = D.shape[0]
    cell_pairs = cell_pairs if cell_pairs is not None else list(range(n_cells))
    expected_distances = -np.ones((n_cells, n_cells, 3))
    expected_pairwise_distances = -np.ones((n_cells, n_cells))
    for v, w in cell_pairs:
        D_pair = D[v, w]
        Dp_pair = Dp[v, w]
        D_uv = D_pair[1]
        D_uw = D_pair[2]
        expected_distances[v, w, :] = math_utils.l_from_p(D_pair / Dp_pair, n_states)
        expected_pairwise_distances[v, w] = D_uv + D_uw

    return expected_distances, expected_pairwise_distances


def get_expected_psi(param, obs_model):
    return


def get_marginals_from_cnp(cnp, n_states, noise=0.0):
    """
    Get the one slice and two slice marginals from one copy number profile.
    """
    M = cnp.shape[0]
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