import os
import dendropy
import pandas as pd
import anndata
import numpy as np
from dendropy.calculate import treecompare

from cellmates.utils.tree_utils import convert_networkx_to_dendropy, newick_to_nx, make_gt_tree_dist, \
    relabel_name_to_int, normalized_rf_distance, f1_score_clades


def group_ties(gt_dpy_tree: dendropy.Tree, atol: float = 1e-7) -> bool:
    """
    Collapse zero-length edges in a DendroPy tree and report groups of cells
    sharing the same ancestor where zero-length edges occur.
    Returns True if any ties (groups with >2 leaves) are found, else False.
    """
    gt_ties = False
    ties_ancestors = {}

    for edge in gt_dpy_tree.postorder_edge_iter():
        if edge.length is not None and np.isclose(edge.length, 0.0, atol=atol):
            node = edge.head_node
            if node.taxon is None:
                continue
            cell_lab = node.taxon.label

            # climb up through zero-length ancestors
            ancestor = node
            while ancestor.parent_node is not None:
                parent_edge = ancestor.parent_node.edge
                if parent_edge.length is None or not np.isclose(parent_edge.length, 0.0, atol=atol):
                    break
                ancestor = ancestor.parent_node

            anc_label = ancestor.label if ancestor.label else f"node_{id(ancestor)}"
            ties_ancestors.setdefault(anc_label, []).append(cell_lab)

    # filter and print only groups with >2 leaves
    for anc, cells in ties_ancestors.items():
        gt_ties = True
        print(f"Tie ancestor: {anc} -> cells: {cells}")

    return gt_ties


def main(snakemake):
    truth_ad_path = snakemake.input['truth_ad']
    cm_dist_path = snakemake.input['cm_dist']
    cm_tree_path = snakemake.input['cm_tree']
    cm_cell_names_path = snakemake.input['cm_cells']
    n_states = snakemake.params['n_states']
    dataset = snakemake.params['dataset']
    seed = snakemake.params['seed']
    data_type = snakemake.params.get('method', 'reads')
    out_csv = snakemake.output[0]

    print(f"Processing [{seed}] {truth_ad_path}, {cm_dist_path}, {cm_tree_path}, {cm_cell_names_path} to {out_csv} with n_states={n_states}, data_type={data_type}")
    # load inputs
    ad = anndata.read_h5ad(truth_ad_path)
    cm_dist = np.load(cm_dist_path)
    cm_tree = open(cm_tree_path).read()
    cell_names = open(cm_cell_names_path).read().strip().split('\n')

    # validate inputs
    print(ad)
    n_cells = ad.n_obs
    if 'normal' in ad.obs.keys():
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
    # check for ties in ground truth tree (zero-length edges) and print groups of cells involved
    gt_ties = group_ties(gt_dpy_tree)

    # save results
    # for each edge, plot error, edge depth (TODO)
    results = pd.DataFrame({'dat_path': [truth_ad_path], 'dataset': [dataset], 'seed': [seed], 'n_cells': [n_cells], 'n_states': [n_states], 'n_clones': [len(set(clone_assignments))],
                            'n_sites': [ad.n_vars],
                            'lambda': [ad.uns['cnasim-params']['placement_param']],
                            'ru_mse': [err_list[0]],
                            'uv_mse': [err_list[1]], 'uw_mse': [err_list[2]],
                            'rf': [rf], 'urf': [urf], 'nrf': [nrf], 'f1_gt': [f1_gt], 'f1_em': [f1_em], 'wgd': [ad.uns['cnasim-params']['WGD']], 'gt_ties': [gt_ties], 'data_type': [data_type]})
    results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__=="__main__":
    # mock snakemake object for local testing if not executed via snakemake

    dat_path = '/home/vittorio.zampinetti/Cellmates/reproducibility/workflows/cnasim_makedata/results/A3_0/0'
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
            'seed': 0,
            'dataset': 'A3_0'
        }
        output = [os.path.join(dat_path, 'eval_tmp.csv')]

    snakemake = MockSnakeMake()

    main(snakemake)