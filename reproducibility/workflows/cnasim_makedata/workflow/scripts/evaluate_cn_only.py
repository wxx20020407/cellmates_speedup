import os
import dendropy
import pandas as pd
import anndata
import numpy as np
from dendropy.calculate import treecompare

from cellmates.utils.tree_utils import convert_networkx_to_dendropy, newick_to_nx, make_gt_tree_dist, \
    relabel_name_to_int, normalized_rf_distance, f1_score_clades

def main(snakemake):
    truth_ad_path = snakemake.input['truth_ad']
    cm_dist_path = snakemake.input['cm_dist']
    cm_tree_path = snakemake.input['cm_tree']
    cm_cell_names_path = snakemake.input['cm_cells']
    n_states = snakemake.params['n_states']
    dataset = snakemake.params['dataset']
    seed = snakemake.params['seed']
    out_csv = snakemake.output[0]

    print(f"Processing [{seed}] {truth_ad_path}, {cm_dist_path}, {cm_tree_path}, {cm_cell_names_path} to {out_csv} with n_states={n_states}")
    # load inputs
    ad = anndata.read_h5ad(truth_ad_path)
    cm_dist = np.load(cm_dist_path)
    cm_tree = open(cm_tree_path).read()
    cell_names = open(cm_cell_names_path).read().strip().split('\n')

    # validate inputs
    print(ad)
    n_cells = ad.n_obs
    assert len(cell_names) == n_cells
    assert cm_dist.shape == (n_cells, n_cells, 3), f"EM distance matrix has incorrect shape: {cm_dist.shape} vs {(n_cells, n_cells, 3)}"
    print("EM tree", cm_tree)
    # transform CNAsim adata to match cell_names, tree and lengths
    nxtree = newick_to_nx(ad.uns['cell-tree-newick'], edge_attr='length')
    nxtree = relabel_name_to_int(nxtree, cell_names)
    gt_dpy_tree = convert_networkx_to_dendropy(nxtree, internal_nodes_label='int', edge_length='length')

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
    print(f"====RESULTS====\nTree RF\n\tnormalized: {nrf}\n\tunweighted: {urf}\n\tweighted {rf} (not relevant)")
    # FIXME: F1 score cannot be computed cause clone assignments are not available in GT data, check with authors
    # clone_assignments = ad[~ad.obs['normal']].obs['clone'].tolist()
    # f1_gt = f1_score_clades(gt_dpy_tree, clone_assignments)
    # f1_em = f1_score_clades(em_dpy_tree, clone_assignments)
    # n_clones = len(set(clone_assignments))
    # lamda = ad.uns['cnasim-params']['placement_param']
    # wgd = ad.uns['cnasim-params']['WGD']
    f1_gt = f1_em = None
    n_clones = lamda = wgd = None

    print(f"F1 score clades: GT {f1_gt}, EM {f1_em}")
    # save results
    # for each edge, plot error, edge depth (TODO)
    results = pd.DataFrame({'dat_path': [truth_ad_path], 'dataset': [dataset], 'seed': [seed], 'n_cells': [n_cells], 'n_states': [n_states], 'n_clones': [n_clones],
                            'n_sites': [ad.n_vars],
                            'lambda': [lamda],
                            'rf': [rf], 'urf': [urf], 'nrf': [nrf], 'f1_gt': [f1_gt], 'f1_em': [f1_em],
                            'wgd': [wgd]})
    results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__=="__main__":
    # mock snakemake object for local testing if not executed via snakemake

    # dat_path = '/home/vittorio.zampinetti/Cellmates/reproducibility/workflows/cnasim_makedata/results/A3_0/0'
    dat_path = '/home/vittorio.zampinetti/data/cnasim_simulated_datasets/CNAsim/A2_1/1'
    cm_out_dir = 'cm_out'
    class MockSnakeMake:
        input = {
            'truth_ad': os.path.join(dat_path, 'anndata.h5ad'),
            'cm_dist': os.path.join(dat_path, cm_out_dir, 'distance_matrix.npy'),
            'cm_tree': os.path.join(dat_path, cm_out_dir, 'tree.nwk'),
            'cm_cells':os.path.join(dat_path, cm_out_dir, 'cell_names.txt'),
        }
        params = {
            'n_states': 7,
            'seed': 1,
            'dataset': 'A2_1'
        }
        output = [os.path.join(dat_path, 'eval_tmp.csv')]

    snakemake = MockSnakeMake()

    main(snakemake)