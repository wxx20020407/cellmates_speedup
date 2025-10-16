import itertools
import os

import numpy as np

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

def get_expected_changes(cnps, tree_nx, n_states, cell_pairs=None):
    n_cells, n_sites = cnps.shape
    cell_pairs = cell_pairs if cell_pairs is not None else list(itertools.combinations(range(n_cells), r=2))
    D = []
    Dp = []
    expected_distances = -np.ones((n_cells, n_cells, 3))
    expected_pairwise_distances = -np.ones((n_cells, n_cells))
    root = [v for v in tree_nx.nodes() if tree_nx.out_degree(v) == 2][0]
    for pair in cell_pairs:
        lca = tree_utils.get_lowest_common_ancestor(tree_nx, pair[0], pair[1])
        D_ru = math_utils.compute_cn_changes(cnps, pairs=[(root, lca)])[0]
        D_uv = math_utils.compute_cn_changes(cnps, pairs=[(pair[0], lca)])[0]
        D_uw = math_utils.compute_cn_changes(cnps, pairs=[(pair[1], lca)])[0]
        Dp_ru = n_sites - D_ru
        Dp_uv = n_sites - D_uv
        Dp_uw = n_sites - D_uw
        D_pair = np.array([D_ru, D_uv, D_uw])
        Dp_pair = np.array([Dp_ru, Dp_uv, Dp_uw])
        D.append(D_pair)
        Dp.append(Dp_pair)
        expected_distances[pair[0], pair[1], :] = math_utils.l_from_p(D_pair / Dp_pair, n_states)
        expected_pairwise_distances[pair[0], pair[1]] = D_uv + D_uw
    return D, Dp, expected_distances, expected_pairwise_distances
