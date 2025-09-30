import argparse
import os
import logging

import anndata
import numpy as np

from cellmates.inference.em import EM
from cellmates.inference.neighbor_joining import build_tree
from cellmates.utils.tree_utils import write_newick

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def obs_from_adata(adata, layer_name=None):
    dat = adata.layers[layer_name] if layer_name else adata.X
    return dat.T


def main():
    # read params from argparse
    parser = argparse.ArgumentParser(description="Estimate distance matrix and build phylogenetic tree")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input AnnData file")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path to output AnnData file")
    parser.add_argument('--n-states', '-s', type=int, default=7, help="Number of hidden states")
    parser.add_argument('--max-iter', '-m', type=int, default=30, help="Maximum number of EM iterations")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="Verbosity level (0: silent, 1: progress, 2+: debug)")
    args = parser.parse_args()

    # set paths
    adata_path = args.input
    logger.info(f"Reading AnnData file {adata_path}")
    out_path = args.output if args.output else '.'  # default to current directory if not provided
    if out_path:
        os.makedirs(out_path, exist_ok=True)
        logger.info(f"Created output directory: {out_path}")
    dist_path = out_path + '/distance_matrix.npy'  # numpy format
    tree_path = out_path + '/tree.nwk'

    # load data
    adata = anndata.read_h5ad(adata_path)
    X = obs_from_adata(adata)
    # run inference
    em = EM(n_states=args.n_states, obs_model='normal', verbose=args.verbose)
    em.fit(X, max_iter=args.max_iter,)
    nx_tree = build_tree(em.distances, edge_attr='branch_length')
    # save results
    np.save(dist_path, em.distances)  # save distance matrix
    nwk_str = write_newick(nx_tree, cell_names=adata.obs_names.tolist(), out_path=tree_path, edge_attr='branch_length') # save newick tree

    logger.debug(f"Newick string: {nwk_str} output to {tree_path}")
    logger.info(f"Saved tree to {tree_path}")
    logger.info(f"Saved dist matrix to {dist_path}")

if __name__ == '__main__':
    main()
