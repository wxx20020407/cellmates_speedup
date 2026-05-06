import anndata
import numpy as np
import argparse
import pickle
import scicone

def parse_args():
    parser = argparse.ArgumentParser(description='Run SCICoNE on an AnnData object.')
    parser.add_argument('--input-anndata', type=str, required=True,
                        help='Path to the input AnnData object (.h5ad file).')
    parser.add_argument('--output-prefix', type=str, required=True,
                        help='Prefix for the output files. e.g., /path/to/output_prefix will create /path/to/output_prefix_tree.pkl')
    parser.add_argument('--install-path', type=str, required=True,
                        help='Path to the SCICoNE installation directory.')
    parser.add_argument('--temporary-outpath', type=str, default='./',
                        help='Path for temporary output files.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    return parser.parse_args()


if __name__ == '__main__':
    # install_path = '/cluster/work/bewi/members/pedrof/tupro_code/SCICoNE/build2024/'
    # temporary_outpath = './'
    args = parse_args()
    np.random.seed(args.seed)

    # Create SCICoNE object
    sci = scicone.SCICoNE(args.install_path, args.temporary_outpath, verbose=False)
    adata = anndata.read_h5ad(args.input_anndata)
    print("Loaded AnnData with shape:", adata.shape)
    count_matrix = adata.X
    print("Detecting breakpoints...")
    bps = sci.detect_breakpoints(count_matrix, threshold=3.0)
    print("Inferring tree structure...")
    inferred_tree = sci.learn_tree(count_matrix, bps['segmented_region_sizes'], n_reps=4, seed=args.seed)
    print("Saving inferred tree obj...")
    pickle.dump(inferred_tree, open(args.output_prefix + '_tree.pkl', 'wb'))
    print("Saving other outputs: graphviz Source, cnv profiles, and cell attachments...")
    pickle.dump(inferred_tree.plot_tree(node_labels=True), open(args.output_prefix + '_graphviz_src.pkl', 'wb'))
    np.save(args.output_prefix + '_cnv.npy', inferred_tree.outputs['inferred_cnvs'])
    np.save(args.output_prefix + '_cell_node_labels.npy', inferred_tree.cell_node_labels)
    print("Done.")
