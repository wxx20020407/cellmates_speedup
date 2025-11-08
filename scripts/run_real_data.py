import os

import anndata
import argparse

from cellmates.inference.pipeline import run_inference_pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CellMates inference pipeline on real data.")
    parser.add_argument("-i", "--input", type=str, help="Path to input data.", default="/home/vittorio.zampinetti/data/patient_MM-03.h5ad")
    parser.add_argument("-o", "--output", type=str, help="Output directory.", default="/home/vittorio.zampinetti/cellmates_res/real_data/MM_03_cm_out")
    parser.add_argument('-p', '--processors', type=int, default=4, help='Number of processors to use. If 0, use all available processors.')
    args = parser.parse_args()
    if args.processors == 0:
        import multiprocessing
        args.processors = multiprocessing.cpu_count()
    return args

def main():
    args = parse_arguments()
    # adata_path = "/proj/sc_ml/shared/bahlis_10x/patient_MM-03.h5ad"
    # out_dir = "/proj/sc_ml/users/x_vitza/cellmates_res/real_data/MM_03_cm_out"

    # filter cells based on ploidy (only wgd and not noisy)
    normal_anno = 'normal'
    adata = anndata.read_h5ad(args.input)
    adata_filt = adata[(adata.obs['mean_ploidy'] > 3) & (~adata.obs['is_noisy'])]
    adata_filt.obs[normal_anno] = adata_filt.obs['is_normal'].values.astype(bool) # create a boolean annotation for normal cells
    print("Filtered adata has {} cells and {} bins".format(adata_filt.n_obs, adata_filt.n_vars))

    # save adata for future reference
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    adata_filt_path = args.output + "/adata_filt.h5ad"
    print("Saving filtered adata to {}".format(adata_filt_path))
    adata_filt.write_h5ad(adata_filt_path)
    assert 'copy' in adata_filt.layers.keys()
    print("Using {} processors".format(args.processors))
    run_inference_pipeline(input=adata_filt_path, output=args.output,
                           n_states=8, max_iter=30, num_processors=args.processors, rtol=1e-3, learn_obs_params=False,
                           numpy=True, save_diagnostics=True, tau=2., normal_annotation=normal_anno, init_from_cn=True)


if __name__=="__main__":
    main()