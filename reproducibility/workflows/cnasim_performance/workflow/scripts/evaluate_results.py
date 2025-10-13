import networkx as nx
import pandas as pd
import anndata
import numpy as np
import dendropy as dpy
from dendropy.calculate import treecompare

from cellmates.simulation.datagen import get_ctr_table
from cellmates.utils import tree_utils
from cellmates.utils import math_utils
from cellmates.utils.tree_utils import convert_networkx_to_dendropy, newick_to_nx

def relabel_name_to_int(nxtree: nx.DiGraph, cell_names: list) -> nx.DiGraph:
    cells_mapping = {name: i for i, name in enumerate(cell_names)}
    ancestors_mapping = {n: i + len(cell_names) for i, n in enumerate(nxtree.nodes()) if not n.startswith('cell')}
    full_mapping = {**cells_mapping, **ancestors_mapping}
    return nx.relabel_nodes(nxtree, full_mapping, copy=True)

def make_gt_tree_dist(ad, n_states, cell_names: list) -> tuple[dpy.Tree, np.ndarray]:
    # traverse the tree, write lengths to branches and, for each pair, sum lengths between them
    nxtree = tree_utils.newick_to_nx(ad.uns['cell-tree-newick'])
    nxtree = nx.dfs_tree(nxtree, source='founder')  # make sure it's rooted
    # get copy number (ancestors) at each node and compute the length based on changes
    ancestor_idx = {a: i for i,a in enumerate(ad.uns['ancestral-names'])}  # names as in the tree (index for ancestral-cn)
    ancestor_cn = ad.uns['ancestral-cn'] # shape (n_ancestors, n_bins)
    for u, v in nxtree.edges():
        if not v.startswith('cell'):
            i,j = ancestor_idx[u], ancestor_idx[v]
            target_length = math_utils.l_from_p(math_utils.compute_cn_changes(ancestor_cn, pairs=[(i, j)])[0], n_states)
            nxtree[u][v]['length'] = target_length
        else:
            v_cn = ad[v].layers['state']
            u_cn = ancestor_cn[ancestor_idx[u]]
            target_length = math_utils.l_from_p(math_utils.compute_cn_changes(np.vstack([u_cn, v_cn]), pairs=[(0, 1)])[0], n_states)
            nxtree[u][v]['length'] = target_length
    nxtree = relabel_name_to_int(nxtree, cell_names)
    dpy_tree = convert_networkx_to_dendropy(nxtree, edge_length='length', internal_nodes_label='int')
    dist_matrix = get_ctr_table(dpy_tree)
    return dpy_tree, dist_matrix


def main(snakemake):
    truth_ad = snakemake.input['truth_ad']
    cm_dist = snakemake.input['cm_dist']
    cm_tree = snakemake.input['cm_tree']
    cm_cell_names = snakemake.input['cm_cells']
    n_states = snakemake.params['n_states']
    seed = snakemake.params['seed']
    out_csv = snakemake.output[0]

    print(f"Processing [{seed}] {truth_ad}, {cm_dist}, {cm_tree}, {cm_cell_names} to {out_csv} with n_states={n_states}")
    # load inputs
    ad = anndata.read_h5ad(truth_ad)
    cm_dist = np.load(cm_dist)
    cm_tree = open(cm_tree).read()
    cell_names = open(cm_cell_names).read().strip().split('\n')

    # validate inputs
    print(ad)
    n_cells = ad[~ad.obs['normal']].shape[0]
    assert len(cell_names) == n_cells
    assert cm_dist.shape == (n_cells, n_cells, 3), f"EM distance matrix has incorrect shape: {cm_dist.shape} vs {(n_cells, n_cells, 3)}"
    print("EM tree", cm_tree)
    # transform CNAsim adata to match cell_names, tree and lengths
    gt_dpy_tree, gt_dist_matrix = make_gt_tree_dist(ad, n_states, cell_names)
    # FIXME: GT distance has NaNs for some pairs, why?

    # compute errors
    err_list = []
    for i in range(3):
        print(f'[{i}] DIST')
        tri = np.triu_indices(n_cells)
        print(cm_dist[:, :, i])
        print(gt_dist_matrix[:, :, i])
        se_mat = (cm_dist[:, :, i] - gt_dist_matrix[:, :, i])**2
        # print(se_mat)
        err = np.mean(se_mat[tri])
        print(f"MSE: {err}")
        err_list.append(err)
    # compute tree RF
    em_nxtree = newick_to_nx(cm_tree)
    assert em_nxtree.number_of_nodes() == 2 * n_cells - 1, "inferred and true tree have different number of nodes"
    em_nxtree = relabel_name_to_int(em_nxtree, cell_names)
    em_dpy_tree = convert_networkx_to_dendropy(em_nxtree, edge_length='weight', internal_nodes_label='int', taxon_namespace=gt_dpy_tree.taxon_namespace)
    print("GT tree:")
    gt_dpy_tree.print_plot(plot_metric='length')
    print("EM tree:")
    em_dpy_tree.print_plot(plot_metric='length')
    rf = treecompare.weighted_robinson_foulds_distance(gt_dpy_tree, em_dpy_tree)
    print(f"Tree RF: {rf}")
    # save results
    # for each edge, plot error, edge depth (TODO)
    results = pd.DataFrame({'seed': [seed], 'cm_ru_mse': [err_list[0]],
                            'cm_uv_mse': [err_list[1]], 'cm_uw_mse': [err_list[2]],
                            'cm_tree_rf': [rf]})
    results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__=="__main__":
    # mock snakemake object for local testing
    # class MockSnakeMake:
    #     input = {
    #         'truth_ad': '../../results/data/R1_N10_M100_K2_L2_E10.04_E20.1_C2/input.h5ad',
    #         'cm_dist': '../../results/data/R1_N10_M100_K2_L2_E10.04_E20.1_C2/cm_out/distance_matrix.npy',
    #         'cm_tree': '../../results/data/R1_N10_M100_K2_L2_E10.04_E20.1_C2/cm_out/tree.nwk'
    #     }
    #     params = {
    #         'n_states': 7,
    #         'seed': 1
    #     }
    #     output = ['../../results/data/R1_N10_M100_K2_L2_E10.04_E20.1_C2/eval_tmp.csv']
    #
    # snakemake = MockSnakeMake()

    main(snakemake)