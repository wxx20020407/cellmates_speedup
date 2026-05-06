import os

import anndata
import argparse

import matplotlib.pyplot as plt
import numpy as np
from anndata import read_h5ad
from scipy import stats
import pandas as pd
import scgenome.plotting as pl

from cellmates.inference.pipeline import run_inference_pipeline, run_prediction_from_output


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CellMates inference pipeline on spectrum-OV datasets.")
    parser.add_argument("-i", "--input", type=str, help="Path to input data.", default="/home/vittorio.zampinetti/data/SPECTRUM/signals_SPECTRUM-OV-125.h5")
    parser.add_argument('-f', '--filter-sbm', type=str, help='Path to the adata file containing cells to keep', default="/home/vittorio.zampinetti/data/SPECTRUM/sbmclone_SPECTRUM-OV-125_snv.h5")
    parser.add_argument("-o", "--output", type=str, help="Output directory.", default="/home/vittorio.zampinetti/cellmates_res/real_data/SPECTRUM-OV-125_cm_out_phased")
    parser.add_argument('-p', '--processors', type=int, default=4, help='Number of processors to use. If 0, use all available processors.')
    parser.add_argument('--from-cn', action='store_true', help='Whether to run on CN data or read counts.')
    parser.add_argument('--phased', action='store_true', help='Whether to use allele-specific counts.')
    parser.add_argument('--downscale', type=int, default=1, help='Factor to downscale the data by (e.g., 2 means half the bins).')
    parser.add_argument('--skip-cn-prediction', action='store_true', help='Whether to skip CN prediction step.')
    parser.add_argument('--dry-run', action='store_true', help='If set, do not run the full pipeline, just check arguments and data loading.')
    parser.add_argument('--cleanup', action='store_true', help='If set, clean up intermediate files after running the pipeline.')
    args = parser.parse_args()
    if args.processors == 0:
        import multiprocessing
        args.processors = multiprocessing.cpu_count()
    return args


def downscale_adata(adata, factor, sum_layers, mean_layers, mode_layers):
    # downscale adata by factor
    layers = sum_layers + mean_layers + mode_layers
    chrom_bin_sizes = adata.var['chr'].value_counts() // factor
    # print(f"Chromosome bin sizes after downscaling by factor {factor}:", chrom_bin_sizes.to_dict())
    n_bins = np.sum(chrom_bin_sizes)
    # print("Downscaling adata from", adata.n_vars, "to", n_bins, "bins.")
    lay_X = {layer: np.zeros((adata.n_obs, n_bins)) for layer in layers}
    lay_X['X'] = np.zeros((adata.n_obs, n_bins))
    var_downscaled = []
    bin_idx = 0
    for chrom in adata.var['chr'].unique():
        chrom_mask = adata.var['chr'] == chrom
        chrom_bins = adata.var[chrom_mask]
        n_chrom_bins = chrom_bins.shape[0]
        print(f"Downscaling chromosome {chrom} with {n_chrom_bins} bins to {n_chrom_bins // factor} bins.")
        for new_bin in range(chrom_bin_sizes[chrom]):
            start = new_bin * factor
            end = start + factor
            bin_slice = chrom_bins.index[start:end]
            lay_X['X'][:, bin_idx] = np.nansum(adata[:, bin_slice].X, axis=1)
            for layer in sum_layers:
                lay_X[layer][:, bin_idx] = np.nansum(adata[:, bin_slice].layers[layer], axis=1)
            for layer in mean_layers:
                lay_X[layer][:, bin_idx] = np.mean(adata[:, bin_slice].layers[layer], axis=1)
            for layer in mode_layers:
                lay_X[layer][:, bin_idx] = stats.mode(adata[:, bin_slice].layers[layer], axis=1, nan_policy='omit').mode.flatten()
            var_downscaled.append({
                'chr': chrom,
                'start': chrom_bins.iloc[start]['start'],
                'end': chrom_bins.iloc[end - 1]['end']
            })
            bin_idx += 1
    var_downscaled_df = pd.DataFrame(var_downscaled)
    adata = anndata.AnnData(X=lay_X['X'], var=var_downscaled_df, obs=adata.obs)
    for layer in layers:
        adata.layers[layer] = lay_X[layer]
    return adata

def add_allele_copy(adata):
    ab_counts = ['alleleA', 'alleleB']
    normal = adata[adata.obs['is_normal']]  # only tumor cells
    pseudo_bulk_norm = normal.layers['alleleA'] + normal.layers['alleleB']
    pseudo_bulk_norm = np.nanmean(pseudo_bulk_norm, axis=0) # mean across cells
    tumor = adata[~adata.obs['is_normal']]
    rdrs = []
    for layer in ab_counts:
        # compute depth ratios
        rdr = adata.layers[layer] / np.nansum(adata.layers[layer], axis=1, keepdims=True)  # average depth ratio per cell
        rdr = rdr * np.nansum(pseudo_bulk_norm, keepdims=True) / pseudo_bulk_norm  # scale by pseudo-bulk at each bin
        rdr[adata.obs['is_wgd'], :] = rdr[adata.obs['is_wgd'], :] * 2  # adjust for WGD cells
        rdrs.append(rdr)
    adata.layers['alleleA_copy'] = rdrs[0]
    adata.layers['alleleB_copy'] = rdrs[1]
    return adata

def join_phases(adata):
    # join allele-specific layers into single layer (A then B) and rename chromosomes accordingly (both cn and reads)
    n_bins = adata.n_vars * 2
    # create new var with chromosome names duplicated with suffixes _A and _B
    var_joined = pd.DataFrame({
        'chr': np.concatenate([adata.var['chr'] + '_A', adata.var['chr'] + '_B']),
        'start': np.concatenate([adata.var['start'], adata.var['start']]),
        'end': np.concatenate([adata.var['end'], adata.var['end']])
    })
    lay_X = np.zeros((adata.n_obs, n_bins))
    lay_X[:, :adata.n_vars] = adata.layers['alleleA']
    lay_X[:, adata.n_vars:] = adata.layers['alleleB']
    adata_joined = anndata.AnnData(X=lay_X, var=var_joined, obs=adata.obs)
    # similarly for copy number layers
    lay_A_copy = np.zeros((adata.n_obs, n_bins))
    lay_A_copy[:, :adata.n_vars] = adata.layers['alleleA_copy']
    lay_A_copy[:, adata.n_vars:] = adata.layers['alleleB_copy']
    adata_joined.layers['copy'] = lay_A_copy
    # and state
    lay_state = np.zeros((adata.n_obs, n_bins))
    lay_state[:, :adata.n_vars] = adata.layers['A']
    lay_state[:, adata.n_vars:] = adata.layers['B']
    adata_joined.layers['state'] = lay_state
    return adata_joined


def make_tmp_adata(original_adata_path, downscale_factor, phased, filters=None):
    # load original adata, downscale and join phases if needed, then save to tmp_adata_path
    adata = anndata.read_h5ad(original_adata_path)
    adata.obs['is_normal'] = adata.obs['is_normal'].astype(bool)
    # filters is a list of tuples (keep: bool, obs_name: str)
    if downscale_factor > 1:
        adata = downscale_adata(adata, factor=downscale_factor, sum_layers=['alleleA', 'alleleB'], mean_layers=['copy'], mode_layers=['A', 'B', 'state'])
    if phased:
        # join phases into single layer (one next to the other) - allow for only 'copy' or 'state' layers
        adata = add_allele_copy(adata)
        adata = join_phases(adata)
    if filters is not None:
        for keep, obs_name in filters:
            keep_obs = adata.obs[obs_name].astype(bool)
            if not keep:
                keep_obs = ~keep_obs
            adata = adata[keep_obs].copy()
    return adata


def manual_setup(args):
    args.phased = True
    args.dry_run = False
    args.downscale = 10
    return args

def main():
    args = parse_arguments()
    args = manual_setup(args)
    # prepare data (downscaling, phasing)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # filter cells based on ploidy (only wgd and not noisy)
    normal_anno = 'is_normal'
    datatype = "cn" if args.from_cn else "reads"
    use_copynumbers = datatype == 'cn'
    tau = 3.6
    n_states = 8 if not args.phased else 6  # number of copy number states to consider (8 for total CN, 6 for allele-specific)
    adata = make_tmp_adata(original_adata_path=args.input,
                           downscale_factor=args.downscale,
                           phased=args.phased,
                           filters=[#(False, 'is_s_phase'),
                                    # (False, 'is_outlier'),
                                    #(False, 'is_normal'),
                                    #(False, 'is_wgd')
                               ])

    processed_adata_path = os.path.join(args.output, "processed_adata.h5ad")
    # adata = read_h5ad(processed_adata_path)
    if args.filter_sbm:
        sbm_adata = read_h5ad(args.filter_sbm)
        keep_cells = sbm_adata.obs_names
        adata = adata[adata.obs_names.isin(keep_cells)].copy()
        print(f"Filtered adata to {adata.n_obs} cells based on SBM filter from {args.filter_sbm}")
    adata.write_h5ad(processed_adata_path)
    print(f"Processed adata of shape {adata.shape} saved to {processed_adata_path}")
    # adata = read_h5ad(processed_adata_path)
    print(f"adata of shape {adata.shape} loaded from {args.input}")

    if args.dry_run:
        print("Dry run complete. Exiting without running inference pipeline.")
        return

    print(f"running on {args.processors} processors")
    run_inference_pipeline(input=processed_adata_path, output=args.output,
                           use_copynumbers=use_copynumbers,
                           n_states=n_states, max_iter=30, num_processors=args.processors, rtol=1e-3, learn_obs_params=True,
                           numpy=True, save_diagnostics=True, tau=tau, normal_annotation=normal_anno, init_from_cn=True,
                           predict_cn=not args.skip_cn_prediction, layer_name=None)
    # run_prediction_from_output(args.input, output_path=args.output, tau=tau, n_states=n_states, use_copynumbers=use_copynumbers)

if __name__=="__main__":
    main()