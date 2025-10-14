import pandas as pd
import anndata
import numpy as np
from dendropy.calculate import treecompare

from cellmates.utils.tree_utils import convert_networkx_to_dendropy, newick_to_nx, make_gt_tree_dist, \
    relabel_name_to_int, normalized_rf_distance, f1_score_clades


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
    urf = treecompare.symmetric_difference(gt_dpy_tree, em_dpy_tree)
    nrf = normalized_rf_distance(gt_dpy_tree, em_dpy_tree)
    print(f"Tree RF: {rf} (unweighted {urf}) - normalized: {nrf}")
    clone_assignments = ad[~ad.obs['normal']].obs['clone'].tolist()
    f1_gt = f1_score_clades(gt_dpy_tree, clone_assignments)
    f1_em = f1_score_clades(em_dpy_tree, clone_assignments)
    print(f"F1 score clades: GT {f1_gt}, EM {f1_em}")
    # save results
    # for each edge, plot error, edge depth (TODO)
    results = pd.DataFrame({'seed': [seed], 'n_cells': [n_cells], 'n_states': [n_states], 'n_clones': [len(set(clone_assignments))],
                            'lambda': [ad.uns['cnasim-params']['lambda']],
                            'ru_mse': [err_list[0]],
                            'uv_mse': [err_list[1]], 'uw_mse': [err_list[2]],
                            'rf': [rf], 'urf': [urf], 'nrf': [nrf], 'f1_gt': [f1_gt], 'f1_em': [f1_em]})
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