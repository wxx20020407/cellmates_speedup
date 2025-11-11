import logging
import subprocess
import warnings

import dendropy as dpy

from cellmates.utils import tree_utils


def load_medicc2_tsv_file(medicc2_tsv_path):
    """
    Load MEDICC2 formatted TSV file into a suitable data structure.
    Parameters
    ----------
    medicc2_tsv_path : str
        Path to the MEDICC2 formatted TSV file.

    Returns
    -------
    dict
        A dictionary where keys are cell IDs and values are lists of tuples (chromosome, start, end, cn_a, cn_b).
    """
    data = {}


def run_medicc2(dataset_path, out_dir_path, topology_only=False, num_proc=1):
    # Prepare command to run MEDICC2
    medicc2_command = f'medicc2 {dataset_path} {out_dir_path} -j "{num_proc}"'
    medicc2_command += '--topology-only ' if topology_only else ''
    medicc2_command += ' --plot auto'
    warnings.filterwarnings(
        action='ignore',
        category=FutureWarning
    )
    # Run MEDICC2
    logging.info(f'Running dice: {medicc2_command}')
    subprocess.run(medicc2_command, shell=True)


def load_medicc2_tree(medicc2_nwk_file_path: str,
                      taxon_namespace: dpy.TaxonNamespace,
                      N_cells,
                      remove_diploid=True) -> dpy.Tree:
    medicc2_tree_nw = open(medicc2_nwk_file_path).read().strip()
    medicc2_tree_dpy: dpy.Tree = dpy.Tree.get(data=medicc2_tree_nw, schema='newick')
    leaves_mapping = {f'cell {i}': str(i) for i in range(N_cells)}
    leaves_mapping['diploid'] = str(N_cells)
    tree_utils.relabel_dendropy(medicc2_tree_dpy, leaves_mapping)
    # Remove healthy root if present
    if remove_diploid and medicc2_tree_dpy.find_node_with_taxon_label(str(N_cells)) is not None:
        medicc2_tree_dpy.prune_subtree(medicc2_tree_dpy.find_node_with_taxon_label(str(N_cells)))

    medicc2_tree_nx = tree_utils.convert_dendropy_to_networkx(medicc2_tree_dpy)
    cell_names = [str(i) for i in range(N_cells)]
    medicc2_tree_nx = tree_utils.relabel_name_to_int(medicc2_tree_nx, cell_names)
    medicc2_tree_dpy2 = tree_utils.convert_networkx_to_dendropy(medicc2_tree_nx,
                                                                taxon_namespace=taxon_namespace)
    tree_utils.label_tree(medicc2_tree_dpy2)
    return medicc2_tree_dpy2


# The user's file adata_filt.h5ad is assumed to be in the same directory for this script to run.

# The user's file adata_filt.h5ad is assumed to be in the same directory for this script to run.

import h5py
import pandas as pd
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix


def anndata_to_medicc2_tsv(input_h5ad_path: str, output_tsv_path: str) -> None:
    """
    Converts a single-cell copy number AnnData (.h5ad) file to the
    long-format TSV file required by MEDICC2.

    The function handles the specific HDF5/AnnData structure found in
    many scRNA-seq and scDNA-seq CNV tools, particularly extracting
    categorical chromosome data.

    Args:
        input_h5ad_path: Path to the input AnnData file (e.g., 'adata_filt.h5ad').
        output_tsv_path: Path to save the final tab-separated file (e.g., 'medicc2_input.tsv').

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If required data fields (layers/state, barcode, chrom, start, end) are missing.
    """
    try:
        with h5py.File(input_h5ad_path, 'r') as f:

            # --- 1. Extract Copy Number Data (from layers/state) ---
            if 'layers' not in f or 'state' not in f['layers']:
                raise ValueError("The 'layers/state' data matrix (copy numbers) is missing from the H5AD file.")

            # Reading the copy number data from the 'state' layer
            # This is a dense dataset, as confirmed by our previous diagnostics
            print("-> Reading 'layers/state' as the copy number matrix...")
            cn_matrix = f['layers']['state'][:]

            # --- 2. Extract Cell IDs (obs names) ---
            # Using 'barcode' as diagnosed in the file structure
            if 'barcode' not in f['obs']:
                raise ValueError("The primary cell ID '/obs/barcode' is missing.")

            cell_ids_bytes = f['obs']['barcode'][:]
            cell_ids = [s.decode('utf-8') for s in cell_ids_bytes]

            # --- 3. Extract Bin Coordinates (var) ---
            var_data = {}

            # a) Read start and end (assumed to be numerical datasets)
            if 'start' not in f['var'] or 'end' not in f['var']:
                raise ValueError("Required bin coordinate columns ('start' or 'end') are missing from /var/.")
            var_data['start'] = f['var']['start'][:]
            var_data['end'] = f['var']['end'][:]

            # b) Read Chromosome (Categorical Data: codes + categories)
            if 'chr' in f['var'] and 'codes' in f['var']['chr'] and 'categories' in f['var']['chr']:
                chrom_codes = f['var']['chr']['codes'][:]
                chrom_categories_bytes = f['var']['chr']['categories'][:]
                chrom_categories = [s.decode('utf-8') for s in chrom_categories_bytes]

                # Map codes to categories to get the chromosome names
                chrom_series = pd.Categorical.from_codes(
                    codes=chrom_codes,
                    categories=chrom_categories
                ).astype(str)
                var_data['chrom'] = chrom_series
            else:
                raise ValueError(
                    "Missing required chromosome data structure in /var/chr/ (expected codes and categories).")

            bin_coords = pd.DataFrame(var_data)

    except FileNotFoundError:
        print(f"Error: The input file '{input_h5ad_path}' was not found.")
        raise
    except Exception as e:
        print(f"A general error occurred during file reading or matrix processing: {e}")
        raise

    # --- 4. Conversion to MEDICC2 Long Format ---

    # Create a DataFrame from the copy number matrix
    cn_df = pd.DataFrame(
        cn_matrix,
        index=cell_ids,
        columns=range(cn_matrix.shape[1])
    )

    # --- FIX for MEDICCIOError ---
    # Check for NaNs before stacking.
    # .stack() drops NaNs by default, which causes the MEDICC2 error.
    nan_count = cn_df.isnull().sum().sum()
    if nan_count > 0:
        print(f"-> WARNING: Found {nan_count} NaN values in the copy number matrix.")
        # We will fill NaNs with 0.
        # You could also use '2' if diploid is a more sensible fill.
        fill_value = 0
        print(f"   Filling NaNs with {fill_value} to prevent MEDICC2 error.")
        cn_df = cn_df.fillna(fill_value)
    # --- END FIX ---

    # 4a. Melt the DataFrame to long format (Cell_ID, Bin_Index, Copy_Number)
    cn_long = cn_df.stack().reset_index()
    cn_long.columns = ['sample_id', 'bin_index', 'total_cn']

    # 4b. Prepare the Bin Coordinate DataFrame for merging
    bin_coords_merge = bin_coords[['chrom', 'start', 'end']].reset_index(drop=False)
    bin_coords_merge.rename(columns={'index': 'bin_index'}, inplace=True)

    # 4c. Merge the long data with the coordinates using the shared 'bin_index'
    final_df = pd.merge(cn_long, bin_coords_merge, on='bin_index', how='left')

    # 4d. Final formatting
    output_columns = ['sample_id', 'chrom', 'start', 'end', 'total_cn']
    medicc2_input = final_df[output_columns].copy()

    # Ensure types are correct for MEDICC2
    medicc2_input['start'] = medicc2_input['start'].astype(int)
    medicc2_input['end'] = medicc2_input['end'].astype(int)
    # MEDICC2 requires integer copy numbers. Round and cast.
    medicc2_input.loc[:, 'total_cn'] = medicc2_input['total_cn'].round().astype(int)

    # 5. Save the final DataFrame as a tab-separated file
    medicc2_input.to_csv(output_tsv_path, sep='\t', index=False)

    print(f"\n[SUCCESS] Conversion complete.")
    print(f"Output saved to: {output_tsv_path}")
    print(f"Total entries (Cell-Bin combinations): {len(medicc2_input):,}")


import pandas as pd
import anndata
from pathlib import Path
import re  # For sorting chromosome names


def load_medicc2_summary(summary_file: Path) -> dict:
    """Loads the MEDICC2 summary.tsv file into a dictionary."""
    summary_dict = {}
    try:
        with open(summary_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    key, value = parts
                    # Try to convert numerical values
                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        pass
                    summary_dict[key] = value
        return summary_dict
    except FileNotFoundError:
        print(f"Warning: Summary file not found at {summary_file}")
        return {}


def sort_bins_biologically(var_df: pd.DataFrame) -> pd.DataFrame:
    """Sorts the bin/var DataFrame by chromosome and start position."""

    # Create a column for natural sorting of chromosome names
    def get_chrom_sort_key(chrom_name):
        # Remove 'chr' prefix if it exists
        name = re.sub(r'^chr', '', chrom_name, flags=re.IGNORECASE)
        if name.isdigit():
            return (1, int(name))
        if name.upper() == 'X':
            return (2, 23)
        if name.upper() == 'Y':
            return (2, 24)
        if name.upper() == 'M' or name.upper() == 'MT':
            return (2, 25)
        # Other contigs
        return (3, name)

    # Apply the sorting key
    sort_keys = var_df['chrom'].apply(get_chrom_sort_key)
    var_df = var_df.assign(sort_key=sort_keys)
    var_df = var_df.sort_values(by=['sort_key', 'start']).drop(columns=['sort_key'])
    return var_df.reset_index(drop=True)


def medicc2_to_anndata(
        medicc2_output_prefix: str,
        output_h5ad_path: str,
        cn_column: str = 'total_cn'
) -> None:
    """
    Loads all MEDICC2 output files associated with a prefix and
    compiles them into a single AnnData (.h5ad) object.

    Args:
        medicc2_output_prefix: The prefix of the MEDICC2 output files.
            (e.g., 'MM03_medicc2_input_final_')
        output_h5ad_path: The file path to save the new .h5ad file.
            (e.g., 'medicc2_output.h5ad')
        cn_column: The name of the copy number column in the profiles_file.
            (default: 'total_cn')
    """

    # --- 1. Define File Paths ---
    base_path = Path(".")
    cn_profile_file = base_path / f"{medicc2_output_prefix}final_cn_profiles.tsv"
    newick_file = base_path / f"{medicc2_output_prefix}final_tree.new"
    branch_len_file = base_path / f"{medicc2_output_prefix}branch_lengths.tsv"
    pairwise_dist_file = base_path / f"{medicc2_output_prefix}pairwise_distances.tsv"
    summary_file = base_path / f"{medicc2_output_prefix}summary.tsv"

    print(f"Loading data from prefix: {medicc2_output_prefix}...")

    # --- 2. Load and Pivot Core CNV Data ---
    try:
        cn_long_df = pd.read_csv(cn_profile_file, sep='\t')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: The main file '{cn_profile_file}' was not found.")
        return

    # --- 3. Create .var (Bins) DataFrame ---
    print("-> Creating .var (bins) DataFrame...")
    # Get unique bins from the long-format file
    var_df = cn_long_df[['chrom', 'start', 'end']].drop_duplicates()

    # Sort bins biologically (e.g., chr1, chr2, ..., chrX)
    var_df = sort_bins_biologically(var_df)

    # Create a unique bin_id for pivoting, and set as index
    var_df['bin_id'] = var_df['chrom'] + ':' + var_df['start'].astype(str) + '-' + var_df['end'].astype(str)
    var_df = var_df.set_index('bin_id')

    # --- 4. Create .obs (Samples/Nodes) DataFrame ---
    print("-> Creating .obs (samples/nodes) DataFrame...")
    obs_names = cn_long_df['sample_id'].unique()
    obs_df = pd.DataFrame(index=obs_names)
    obs_df.index.name = 'sample_id'

    # Add metadata: distinguish original cells from inferred ancestors
    obs_df['is_ancestor'] = obs_df.index.str.startswith('internal_')

    # Load and merge branch lengths
    try:
        branch_df = pd.read_csv(branch_len_file, sep='\t', header=None, names=['sample_id', 'branch_length']).set_index(
            'sample_id')
        obs_df = obs_df.join(branch_df)
        # The root node (diploid) might not have a branch length
        obs_df['branch_length'] = obs_df['branch_length'].fillna(0)
    except FileNotFoundError:
        print(f"Warning: Branch length file not found at {branch_len_file}")
        obs_df['branch_length'] = pd.NA

    # --- 5. Create .X Matrix (Pivoting) ---
    print("-> Pivoting copy number data into .X matrix (this may take a moment)...")

    # Map the unique bin_id back to the main DataFrame for pivoting
    bin_id_map = var_df.reset_index()[['chrom', 'start', 'end', 'bin_id']]
    cn_long_df = cn_long_df.merge(bin_id_map, on=['chrom', 'start', 'end'], how='left')

    # Pivot the long table into a wide (Cells x Bins) matrix
    X_wide = cn_long_df.pivot(
        index='sample_id',
        columns='bin_id',
        values=cn_column
    )

    # IMPORTANT: Ensure matrix rows/columns match obs/var indices perfectly
    X_wide = X_wide.reindex(index=obs_df.index, columns=var_df.index)

    # --- 6. Load Unstructured Data (.uns) ---
    print("-> Loading unstructured data (.uns)...")
    adata_uns = {}

    # Load Newick tree string
    try:
        with open(newick_file, 'r') as f:
            adata_uns['medicc2_tree_newick'] = f.read()
    except FileNotFoundError:
        print(f"Warning: Newick tree file not found at {newick_file}")

    # Load summary dictionary
    adata_uns['medicc2_summary'] = load_medicc2_summary(summary_file)

    # Load pairwise distance matrix
    try:
        pairwise_df = pd.read_csv(pairwise_dist_file, sep='\t', index_col=0)
        adata_uns['medicc2_pairwise_distances'] = pairwise_df
    except FileNotFoundError:
        print(f"Warning: Pairwise distance file not found at {pairwise_dist_file}")

    # --- 7. Assemble and Save AnnData Object ---
    print("-> Assembling and writing AnnData file...")
    adata_out = anndata.AnnData(
        X=X_wide.astype(np.float32),  # Use float for safety
        obs=obs_df,
        var=var_df,
        uns=adata_uns
    )

    # AnnData requires string indices for obs/var
    adata_out.obs.index = adata_out.obs.index.astype(str)
    adata_out.var.index = adata_out.var.index.astype(str)

    adata_out.write_h5ad(output_h5ad_path, compression="gzip")

    print(f"\n[SUCCESS] Conversion complete.")
    print(f"Output saved to: {output_h5ad_path}")


if __name__ == '__main__':
    # --- Example Usage ---

    # This prefix is based on the example files provided by the user
    # (e.g., 'MM03_medicc2_input_final_final_tree.new')
    folder = "../../../output/MM03/medicc2/"
    INPUT_PREFIX = 'MM03_medicc2_input_final_'
    OUTPUT_FILE = 'MM03_medicc2_output.h5ad'

    # Try to run the conversion
    try:
        medicc2_to_anndata(folder + INPUT_PREFIX, folder + OUTPUT_FILE)

        # Optional: Print info about the generated file
        print("\n--- Verification of generated .h5ad file ---")
        adata_check = anndata.read_h5ad(OUTPUT_FILE)
        print(adata_check)

    except Exception as e:
        print(f"\n*** The script failed to complete: {e} ***")
        print("Please ensure all 5 MEDICC2 output files are present in the directory.")


