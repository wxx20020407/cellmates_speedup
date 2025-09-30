import argparse
import os
import logging

import anndata

from cellmates.inference.em import EM
from cellmates.inference.neighbor_joining import build_tree
from cellmates.utils.tree_utils import nxtree_to_newick

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_obs_adata(adata_path):
    adata = anndata.read_h5ad(adata_path)
    return adata.X.values()


def main():
    # read params from argparse
    parser = argparse.ArgumentParser(description="Estimate distance matrix and build phylogenetic tree")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input AnnData file")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path to output AnnData file")
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

    # # load data
    # X = make_obs_adata(adata_path)
    # # run inference
    # em = EM()
    # em.fit(X)
    # # save results
    # em.distances.save(dist_path)
    # nx_tree = build_tree(em.distances)
    # with open(tree_path, 'w') as f:
    #     f.write(nxtree_to_newick(nx_tree))
    # TODO: remove mock
    # MOCK
    import numpy as np
    dist = np.random.rand(10, 10)
    np.save(dist_path, dist)
    with open(tree_path, 'w') as f:
        f.write("(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")
    # MOCK END

    logger.info(f"Saved tree to {tree_path}")
    logger.info(f"Saved dist matrix to {dist_path}")

if __name__ == '__main__':
    main()
