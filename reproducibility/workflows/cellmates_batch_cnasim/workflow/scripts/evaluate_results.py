import os
import argparse
import pandas as pd
import anndata
import numpy as np
from dendropy.calculate import treecompare

from cellmates.utils.tree_utils import convert_networkx_to_dendropy, newick_to_nx, make_gt_tree_dist, \
    relabel_name_to_int, normalized_rf_distance, f1_score_clades

def parse_args():
    """Parse CLI args when run outside Snakemake."""
    p = argparse.ArgumentParser(description="Evaluate results (Snakemake-compatible).")
    p.add_argument("--adata-path", required=True)
    p.add_argument("--dist-path", required=True)
    p.add_argument("--tree-path", required=True)
    p.add_argument("--cells-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--n-states", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--datatype", default="reads")
    p.add_argument("--medicc2-tree", default=None, help="Path to MEDICC2 inferred tree (newick) for RF comparison.")
    p.add_argument("--dice-tree", default=None, help="Path to DiCE inferred tree (newick) for RF comparison.")
    return p.parse_args()

def group_ties(gt_dpy_tree, atol: float = 1e-7) -> bool:
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


def extract_from_snakemake(snakemake):
    """Extract args from Snakemake object into a namespace."""
    class Namespace:
        pass

    args = Namespace()
    args.adata_path = snakemake.input["truth_ad"]
    args.dist_path = snakemake.input["cm_dist"]
    args.tree_path = snakemake.input["cm_tree"]
    args.cells_path = snakemake.input["cm_cells"]
    args.out_path = snakemake.output[0]
    args.n_states = snakemake.params["n_states"]
    args.dataset = snakemake.params["dataset"]
    args.seed = snakemake.params["seed"]
    args.datatype = snakemake.params.get("method", "reads")
    args.medicc2_tree = snakemake.params.get("medicc2_tree", None)
    args.dice_tree = snakemake.params.get("dice_tree", None)
    return args


def main(args):
    truth_ad_path = args.adata_path
    cm_dist_path = args.dist_path
    cm_tree_path = args.tree_path
    cm_cell_names_path = args.cells_path
    n_states = args.n_states
    dataset = args.dataset
    seed = args.seed
    data_type = args.datatype
    out_csv = args.out_path

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
    else:
        ad.obs['normal'] = [False] * ad.n_obs  # all tumor cells
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
    # compute copy number MAD if available (# check if file exists in same folder as cm_dist
    # load this
    #        cn_path = os.path.join(out_path, 'predicted_copy_numbers.npz')
    #        np.savez(cn_path, data=predicted_cn[0], labels=predicted_cn[1])
    #        logger.info(f"Saved predicted copy number profiles → {cn_path}")
    try:
        cn_mad = None
        if os.path.exists(os.path.join(os.path.dirname(cm_dist_path), 'predicted_copy_numbers.npz')):
            cn_data = np.load(os.path.join(os.path.dirname(cm_dist_path), 'predicted_copy_numbers.npz'))
            pred_cn = cn_data['data']
            # get cells from labels in the same order as in ad
            pred_cn_labels = cn_data['labels']
            label_to_index = {label: idx for idx, label in enumerate(pred_cn_labels)}
            ordered_indices = [label_to_index[name] for name in cell_names]
            pred_cn = pred_cn[ordered_indices, :]
            true_cn = ad[~ad.obs['normal']].layers['state']
            assert pred_cn.shape == true_cn.shape, f"Predicted CN shape {pred_cn.shape} does not match true CN shape {true_cn.shape}"
            cn_mad = np.mean(np.abs(pred_cn - true_cn))
            print(f"Copy number MAD: {cn_mad}")
        else:
            print("Predicted copy number file not found, skipping CN MAD computation.")
            cn_mad = None
    except Exception as e:
        print(f"Error computing CN MAD: {e}")
        cn_mad = None

    medicc2_nrf = None
    if args.medicc2_tree is not None:
        # compute MEDICC2 tree RF if provided
        medicc2_tree_path = args.medicc2_tree
        medicc2_tree = open(medicc2_tree_path).read()
        medicc2_nxtree = newick_to_nx(medicc2_tree)
        medicc2_nxtree = relabel_name_to_int(medicc2_nxtree, cell_names)
        medicc2_dpy_tree = convert_networkx_to_dendropy(medicc2_nxtree, edge_length='weight', internal_nodes_label='int', taxon_namespace=gt_dpy_tree.taxon_namespace)
        medicc2_nrf = normalized_rf_distance(gt_dpy_tree, medicc2_dpy_tree)
        print(f"MEDICC2 Tree RF normalized: {medicc2_nrf}")

    dice_nrf = None
    if args.dice_tree is not None:
        # compute DICE tree RF if provided
        dice_tree_path = args.dice_tree
        dice_tree = open(dice_tree_path).read()
        dice_nxtree = newick_to_nx(dice_tree)
        dice_nxtree = relabel_name_to_int(dice_nxtree, cell_names)
        dice_dpy_tree = convert_networkx_to_dendropy(dice_nxtree, edge_length='weight', internal_nodes_label='int', taxon_namespace=gt_dpy_tree.taxon_namespace)
        dice_nrf = normalized_rf_distance(gt_dpy_tree, dice_dpy_tree)
        print(f"DiCE Tree RF normalized: {dice_nrf}")
    # save results
    # for each edge, plot error, edge depth (TODO)
    results = pd.DataFrame({'dat_path': [truth_ad_path], 'dataset': [dataset], 'seed': [seed], 'n_cells': [n_cells], 'n_states': [n_states], 'n_clones': [len(set(clone_assignments))],
                            'n_sites': [ad.n_vars],
                            'lambda': [ad.uns['cnasim-params']['placement_param']],
                            'ru_mse': [err_list[0]],
                            'uv_mse': [err_list[1]], 'uw_mse': [err_list[2]],
                            'rf': [rf], 'urf': [urf], 'nrf': [nrf], 'f1_gt': [f1_gt], 'f1_em': [f1_em], 'wgd': [ad.uns['cnasim-params']['WGD']], 'gt_ties': [gt_ties], 'data_type': [data_type], 'cn_mad': cn_mad,
                            'medicc2_nrf': medicc2_nrf, 'dice_nrf': dice_nrf})
    results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__=="__main__":
    # mock snakemake object for local testing if not executed via snakemake

    #dat_path = '/home/vittorio.zampinetti/Cellmates/reproducibility/workflows/cnasim_makedata/results/A3_0/0'
    #cm_out_dir = 'cm_out'
    #class MockSnakeMake:
    #    input = {
    #        'truth_ad': os.path.join(dat_path, 'anndata.h5ad'),
    #        'cm_dist': os.path.join(dat_path, cm_out_dir, 'distance_matrix.npy'),
    #        'cm_tree': os.path.join(dat_path, cm_out_dir, 'tree.nwk'),
    #        'cm_cells':os.path.join(dat_path, cm_out_dir, 'cell_names.txt'),
    #    }
    #    params = {
    #        'n_states': 7,
    #        'seed': 0,
    #        'dataset': 'A3_0'
    #    }
    #    output = [os.path.join(dat_path, 'eval_tmp.csv')]

    #snakemake = MockSnakeMake()

    # Detect if running inside Snakemake
    try:
        snakemake  # noqa: F821
    except NameError:
        args = parse_args()
    else:
        args = extract_from_snakemake(snakemake)

    main(args)
