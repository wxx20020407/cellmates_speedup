import argparse
import _utils as utils

def main():
    parser = argparse.ArgumentParser(description="Prepare input for MEDICC2 from .h5ad file")
    parser.add_argument("--input", type=str, required=True, help="Path to the .h5ad input file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output tsv file")
    args = parser.parse_args()

    # Load the .h5ad file
    df = utils.adata_to_df(args.input)
    # df is expected to have columns: 'cell', 'chr', 'start', 'end', 'A', 'B'
    # rename columns to match MEDICC2 input format
    df = df.rename(columns={'cell': 'sample_id', 'chr': 'chrom', 'start': 'start', 'end': 'end', 'A': 'cn_a', 'B': 'cn_b'})
    # select relevant columns
    df['total_cn'] = df['cn_a'] + df['cn_b']
    df = df[['sample_id', 'chrom', 'start', 'end', 'cn_a', 'cn_b', 'total_cn']]
    # save to tsv
    df.to_csv(args.output, sep='\t', index=False, header=True)
    print(f"MEDICC2 input saved to {args.output}")

if __name__ == "__main__":
    main()