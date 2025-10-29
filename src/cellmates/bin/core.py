import argparse
import os
import logging

import anndata
import numpy as np

from cellmates.inference.em import EM
from cellmates.inference.neighbor_joining import build_tree
from cellmates.models.evo import JCBModel
from cellmates.models.obs import NormalModel
from cellmates.utils.tree_utils import write_newick

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def obs_from_adata(adata, layer_name='copy', normal_annotation='normal'):
    """
    Extract observations and chromosome ends from AnnData object.
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    layer_name : str, optional
        Name of the layer to use for observations. If None, use adata.X.
    normal_annotation : str, optional
        Annotation in adata.obs to identify normal cells. (normal cells are not used in distance estimation)
    Returns
    -------
    obs : np.ndarray
        Observation matrix (cells x features).
    chromosome_ends : list
        List of indices where chromosomes end.
    """
    chromosome_ends = np.where(adata.var['chr'].cat.codes.diff() != 0)[0] + 1
    # return dat.T, chromosome_ends.tolist()
    adata_tumor = adata[~adata.obs[normal_annotation]]
    dat = adata_tumor.layers[layer_name] if layer_name else adata_tumor.X
    return dat.T, chromosome_ends.tolist(), adata_tumor.obs_names.tolist()


def main():
    # read params from argparse
    parser = argparse.ArgumentParser(description="Estimate distance matrix and build phylogenetic tree")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input AnnData file")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path to output AnnData file")
    parser.add_argument('--n-states', '-s', type=int, default=7, help="Number of hidden states")
    parser.add_argument('--max-iter', '-m', type=int, default=30, help="Maximum number of EM iterations")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="Verbosity level (0: silent, 1: progress, 2+: debug)")
    parser.add_argument('--num-processors', '-p', type=int, default=1, help="Number of processors to use in parallel")
    parser.add_argument('--alpha', type=float, default=1.0, help="Rate parameter alpha in the Jukes-Cantor model")
    parser.add_argument('--jc-correction', action='store_true', help="Use Jukes-Cantor correction for distance estimation")
    args = parser.parse_args()

    # set paths
    adata_path = args.input
    logger.info(f"Using {args.num_processors} processors for parallel computation")
    # data info
    out_path = args.output if args.output else '.'  # default to current directory if not provided
    if out_path:
        os.makedirs(out_path, exist_ok=True)
        logger.info(f"Created output directory: {out_path}")
    dist_path = os.path.join(out_path, 'distance_matrix.npy')  # numpy format
    tree_path = os.path.join(out_path, 'tree.nwk')
    cell_names_path = os.path.join(out_path, 'cell_names.txt')

    # load data
    logger.info(f"Reading AnnData file {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    logger.info(f"Dataset contains {adata.n_obs} cells and {adata.n_vars} bins")
    obs, chromosome_ends, cell_names = obs_from_adata(adata)
    logger.debug(f"Excluded {adata.n_obs - obs.shape[1]} normal cells from distance estimation")
    # run inference
    evo_model = JCBModel(n_states=args.n_states, chromosome_ends=chromosome_ends, jc_correction=args.jc_correction, alpha=args.alpha)
    obs_model = NormalModel(n_states=args.n_states, mu_v_prior=1., tau_v_prior=100.)
    em = EM(n_states=args.n_states, evo_model=evo_model, obs_model=obs_model, verbose=args.verbose)
    em.fit(obs, max_iter=args.max_iter, num_processors=args.num_processors)
    nx_tree = build_tree(em.distances, edge_attr='branch_length')
    # save results
    np.save(dist_path, em.distances)  # save distance matrix
    nwk_str = write_newick(nx_tree, cell_names=cell_names, out_path=tree_path, edge_attr='branch_length') # save newick tree
    # save cell names
    with open(cell_names_path, 'w') as f:
        for name in cell_names:
            f.write(f"{name}\n")

    logger.debug(f"Newick string: {nwk_str} output to {tree_path}")
    logger.info(f"Saved tree to {tree_path}")
    logger.info(f"Saved dist matrix to {dist_path}")
    logger.info(f"Saved cell names to {cell_names_path}")

if __name__ == '__main__':
    main()
