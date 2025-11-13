import argparse
import _utils as utils

def main():
    parser = argparse.ArgumentParser(description="Prepare input for DICE from .h5ad file")
    parser.add_argument("--input", type=str, required=True, help="Path to the .h5ad input file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output tsv file")
    args = parser.parse_args()

    # Load the .h5ad file
    df = utils.adata_to_df(args.input)
    # df is expected to have columns: 'cell', 'chr', 'start', 'end', 'A', 'B'
    df['CN states'] = df['A'].astype(str) + ',' + df['B'].astype(str)
    # rename columns to match DICE input format
    df = df.rename(columns={'cell': 'CELL', 'chr': 'chrom'})
    # select relevant columns
    df = df[['CELL', 'chrom', 'start', 'end', 'CN states']]
    # save to tsv
    df.to_csv(args.output, sep='\t', index=False, header=True)
    print(f"DICE input saved to {args.output}")

if __name__ == "__main__":
    main()