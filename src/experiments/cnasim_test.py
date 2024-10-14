"""
Try Cellmates on CNAsim dataset
"""
import os.path

import anndata
import dendropy as dpy

from inference.em import jcb_em_alg, build_tree
from utils.tree_utils import convert_networkx_to_dendropy


def load_cnasim_data(adata_path: str) -> anndata.AnnData:
    # this adata provides the following attributes:
    # X (raw read counts)
    # layers:
    #   - copy (normalized read counts)
    #   - state (copy number states)
    # obs:
    #   - clone (cell type)
    #   - clone-founder (ancestor of cell type)
    # uns:
    #   - clonal-tree-newick (newick string of clonal tree)
    #   - cnasim-params (dict with coverage, bin_length, region_length, window_size)
    adata = anndata.read(adata_path)
    adata.layers['state'] = adata.layers['state'].astype(int)
    return adata


def run(data_path):
    print("Running Cellmates on CNAsim dataset")
    adata = load_cnasim_data(data_path)
    print(f"Loaded CNAsim data: n_cells={adata.n_obs}, n_bins={adata.n_vars}")
    # get true tree dendropy
    print("tree newick:", adata.uns['clonal-tree-newick'])
    true_tree = dpy.Tree.get(data=adata.uns['clonal-tree-newick'], schema='newick',
                             taxon_namespace=dpy.TaxonNamespace())
    true_tree.is_rooted = True

    # get expected number of reads from cnasim params
    params = adata.uns['cnasim-params']
    pois_mean_percopy = (params['coverage'] * params['region_length'] * params['window_size']
                         / (2 * params['bin_length']))
    print(f"Expected mean reads per copy: {pois_mean_percopy:.2f}")

    # run cellmates
    em_out_dict = jcb_em_alg(adata.X.T, n_states=7, max_iter=40, num_processors=4, lam=pois_mean_percopy)
    print("EM done")
    l_hat_path = os.path.join(os.path.basename(data_path), 'em_l_hat.numpy')
    em_out_dict['l_hat'].dump(l_hat_path)
    print(f"l_hat saved: {l_hat_path}")

    # build tree from EM output
    nx_em_tree = build_tree(ctr_table=em_out_dict['l_hat'])
    em_tree = convert_networkx_to_dendropy(nx_em_tree, taxon_namespace=true_tree.taxon_namespace, edge_length='length')
    # compare trees (draw and compute RF distance)
    print("Comparing trees")
    print("True tree")
    true_tree.print_plot(plot_metric='length')
    print("EM tree")
    em_tree.print_plot(plot_metric='length')
    print("RF distance between true and EM tree:", true_tree.robinson_foulds_distance(em_tree))
    print("Done")


if __name__ == '__main__':
    data_path = '/Users/zemp/phd/scilife/cellmates-experiments/data/test.h5ad'
    run(data_path)