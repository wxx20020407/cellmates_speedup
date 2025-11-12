import os

import anndata
import argparse

from cellmates.inference.pipeline import run_inference_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CellMates inference pipeline on specific clones in spectrum-OV-107.")
    parser.add_argument("-i", "--input", type=str, help="Path to input data.", default="/home/vittorio.zampinetti/data/signals_SPECTRUM-OV-107_small.h5")
    parser.add_argument("-o", "--output", type=str, help="Output directory.", default="/home/vittorio.zampinetti/cellmates_res/real_data/SPECTRUM-OV-107_cm_out")
    parser.add_argument('-p', '--processors', type=int, default=4, help='Number of processors to use. If 0, use all available processors.')
    parser.add_argument('--from-cn', action='store_true', help='Whether to run on CN data or read counts.')
    parser.add_argument('--jc-correction', action='store_true', help='Whether to apply JC correction.')
    parser.add_argument('--skip-cn-prediction', action='store_true', help='Whether to skip CN prediction step.')
    parser.add_argument('--dry-run', action='store_true', help='If set, do not run the full pipeline, just check arguments and data loading.')
    args = parser.parse_args()
    if args.processors == 0:
        import multiprocessing
        args.processors = multiprocessing.cpu_count()
    return args

def main():
    args = parse_arguments()

    # filter cells based on ploidy (only wgd and not noisy)
    normal_anno = None
    datatype = "cn" if args.from_cn else "reads"
    adata = anndata.read_h5ad(args.input)
    print(f"adata of shape {adata.shape} loaded from {args.input}")

    # save adata for future reference
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    assert 'copy' in adata.layers.keys()
    print(f"Using {args.processors} processors")
    tau = 2.
    use_copynumbers = datatype == 'cn'
    n_states = 7
    adata.file.close()
    if args.dry_run:
        print("Dry run complete. Exiting without running inference pipeline.")
        return
    run_inference_pipeline(input=args.input, output=args.output,
                           use_copynumbers=use_copynumbers,
                           n_states=n_states, max_iter=30, num_processors=args.processors, rtol=1e-3, learn_obs_params=True,
                           numpy=True, save_diagnostics=True, tau=tau, normal_annotation=normal_anno, init_from_cn=True,
                           predict_cn=not args.skip_cn_prediction)

if __name__=="__main__":
    main()