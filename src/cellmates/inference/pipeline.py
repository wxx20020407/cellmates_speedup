import os, time, pickle, logging
import numpy as np
import anndata

from cellmates.common_helpers.cnasim_data import correct_readcounts
from cellmates.inference.neighbor_joining import build_tree
from cellmates.models.obs import NormalModel, JitterCopy
from cellmates.models.evo import JCBModel
from cellmates.inference.em import EM
from cellmates.utils.tree_utils import write_newick

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
    adata_tumor = adata
    if normal_annotation is not None:
        adata_tumor = adata[~adata.obs[normal_annotation]]
    dat = adata_tumor.layers[layer_name] if layer_name else adata_tumor.X

    return dat.T, chromosome_ends.tolist(), adata_tumor.obs_names.tolist()

def load_and_prepare_adata(adata_path, use_copynumbers):
    logger.info(f"Reading AnnData file {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    if 'cnasim-params' in adata.uns:
        logger.info("Found CNAsim parameters; applying correction.")
        correct_readcounts(adata)
    elif 'copy' not in adata.layers and not use_copynumbers:
        raise ValueError("Missing 'copy' layer or CNAsim parameters.")
    elif use_copynumbers and 'state' not in adata.layers:
        raise ValueError("Missing 'state' layer for copy number states.")
    return adata


def prepare_observations(adata, n_states, tau, learn_obs_params, use_copynumbers, normal_annotation):
    if not use_copynumbers:
        obs, chromosome_ends, cell_names = obs_from_adata(adata, normal_annotation=normal_annotation)
        obs_model = NormalModel(
            n_states=n_states, mu_v_prior=1.0, tau_v_prior=tau, train=learn_obs_params
        )
    else:
        logger.info("Using copy number states directly.")
        obs, chromosome_ends, cell_names = obs_from_adata(
            adata, layer_name='state', normal_annotation=None
        )
        obs_model = JitterCopy(n_states=n_states, jitter=1e-3)
    return obs, chromosome_ends, cell_names, obs_model


def run_em_inference(obs,
                     chromosome_ends,
                     n_states,
                     alpha,
                     jc_correction,
                     hmm_alg, max_iter, rtol, num_processors, obs_model, verbose, save_diag, out_path):
    evo_model = JCBModel(n_states=n_states, chromosome_ends=chromosome_ends,
                         jc_correction=jc_correction, alpha=alpha, hmm_alg=hmm_alg)
    em = EM(n_states=n_states, evo_model=evo_model, obs_model=obs_model,
            verbose=verbose, diagnostics=save_diag)

    start = time.time()
    em.fit(obs, max_iter=max_iter, rtol=rtol, num_processors=num_processors, checkpoint_path=out_path)
    elapsed = time.time() - start
    logger.info(f"EM inference completed in {elapsed:.2f}s")

    if save_diag:
        diag_path = os.path.join(out_path, 'em_diagnostics.pkl')
        with open(diag_path, 'wb') as f:
            pickle.dump(em.diagnostic_data, f)
        logger.info(f"Saved EM diagnostics → {diag_path}")
        # wipe checkpoint data to save space (all _checkpoint_*.pkl files)
        for fname in os.listdir(out_path):
            if fname.startswith('_checkpoint_') and fname.endswith('.pkl'):
                os.remove(os.path.join(out_path, fname))
        logger.info("Removed redundant EM checkpoint files")

    return em


def save_results(em, out_path, cell_names):
    os.makedirs(out_path, exist_ok=True)
    dist_path = os.path.join(out_path, 'distance_matrix.npy')
    tree_path = os.path.join(out_path, 'tree.nwk')
    cell_names_path = os.path.join(out_path, 'cell_names.txt')

    np.save(dist_path, em.distances)
    tree = build_tree(em.distances, edge_attr='branch_length')
    nwk_str = write_newick(tree, cell_names=cell_names, out_path=tree_path)

    with open(cell_names_path, 'w') as f:
        f.writelines(f"{n}\n" for n in cell_names)

    logger.info(f"Saved outputs to {out_path}")
    return {"distances": dist_path, "tree": tree_path, "cells": cell_names_path}


def run_inference_pipeline(
    input,
    output=None,
    n_states=7,
    max_iter=30,
    verbose=0,
    num_processors=1,
    alpha=1.0,
    jc_correction=False,
    rtol=1e-5,
    learn_obs_params=False,
    numpy=False,
    use_copynumbers=False,
    tau=50.0,
    save_diagnostics=False,
    normal_annotation='normal',
):
    out_path = output or "."
    hmm_alg = "broadcast" if numpy else "pomegranate"

    adata = load_and_prepare_adata(input, use_copynumbers)
    obs, chrom_ends, cell_names, obs_model = prepare_observations(
        adata, n_states, tau, learn_obs_params, use_copynumbers, normal_annotation
    )

    em = run_em_inference(
        obs=obs,
        chromosome_ends=chrom_ends,
        n_states=n_states,
        alpha=alpha,
        jc_correction=jc_correction,
        hmm_alg=hmm_alg,
        max_iter=max_iter,
        rtol=rtol,
        num_processors=num_processors,
        obs_model=obs_model,
        verbose=verbose,
        save_diag=save_diagnostics,
        out_path=out_path,
    )

    return save_results(em, out_path, cell_names)
