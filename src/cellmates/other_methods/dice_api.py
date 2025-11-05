import csv
import logging
import os
import subprocess
from typing import Dict, List, Optional

import anndata
import networkx as nx
import numpy as np
from Bio import Phylo
import dendropy as dpy
from dendropy import Tree

from cellmates.utils import tree_utils


def run_dice(dataset_path, out_path=None, method='star', tree_rec='balME'):
    """
    Runs DICE on the .tsv-file specified by dataset_path.
    Requires DICE to be installed in the active virtual environment.
    Parameters
    ----------
    dataset_path: path to DICE input tsv file
    method: star or bar / b
    tree_rec: balME, olsME, NJ or uNJ

    Returns
    -------

    """
    # Prepare command to run DICE
    dice_command = f'dice -i {dataset_path}'
    dice_command += f' -o {out_path}' if out_path else ''
    dice_command += ' -b' if (method == 'bar' or method == 'b') else ''
    dice_command += f' -m {tree_rec}'
    # Run DICE
    logging.info(f'Running dice: {dice_command}')
    subprocess.run(dice_command, shell=True)


def convert_to_dice_tsv(
        cn_array: np.ndarray,
        chromosome_ends: List[int],
        bin_length: int,
        output_filepath: str,
        cell_ids: Optional[List[str]] = None
):
    """
    Converts a NxMx2 numpy array of copy number (CN) states into a
    DICE-compatible TSV file.

    Args:
        cn_array: A numpy array with shape (N, M, 2), where:
                    - N is the number of cells.
                    - M is the number of bins.
                    - 2 is for the two haplotypes (A and B).
                  The array should contain integer CN states.

        chromosome_ends: A list of integers specifying the *end bin index*
                  for each chromosome, in order. Assumes chromosomes are
                  named 'chr1', 'chr2', etc.
                  Example: [1, 3] for M=4 bins means:
                           - 'chr1' = bins 0, 1
                           - 'chr2' = bins 2, 3

        bin_length: The uniform length of each bin (e.g., 10000). The
                    'end' position will be calculated as 'start' + bin_length.

        output_filepath: The path to the .tsv file to be created (e.g., "output.tsv").

        cell_ids: (Optional) A list of string names for the N cells.
                  If None, cells will be named 'cell_0', 'cell_1', ...
    """

    # --- 1. Validate Inputs ---
    if not (len(cn_array.shape) == 3 and cn_array.shape[2] == 2):
        raise ValueError(
            f"Input array must have shape (N, M, 2), but got {cn_array.shape}"
        )

    num_cells_n = cn_array.shape[0]
    num_bins_m = cn_array.shape[1]

    # Validate chromosome_ends
    if not chromosome_ends:
        raise ValueError("chromosome_ends list cannot be empty.")

    if chromosome_ends[-1] != num_bins_m - 1:
        raise ValueError(
            f"The last value in chromosome_ends ({chromosome_ends[-1]}) must "
            f"match the last bin index ({num_bins_m - 1})."
        )

    # Check if list is sorted and has no duplicates
    last_end = -1
    for end_bin in chromosome_ends:
        if end_bin <= last_end:
            raise ValueError(
                f"chromosome_ends must be strictly increasing, but found "
                f"{end_bin} after {last_end}."
            )
        last_end = end_bin

    # Validate or create cell IDs
    if cell_ids is None:
        cell_ids = [f"cell_{i}" for i in range(num_cells_n)]
    elif len(cell_ids) != num_cells_n:
        raise ValueError(
            f"Number of cell_ids ({len(cell_ids)}) does not match "
            f"array's N-dimension ({num_cells_n})"
        )

    # --- 2. Pre-calculate Bin Metadata (chrom, start) ---
    print(f"Calculating genomic positions for {num_bins_m} bins...")
    bin_metadata = []
    current_start = 0
    chr_idx = 0  # 0-based index for chromosome_ends list
    current_chr_name = f"chr{chr_idx + 1}"
    current_chr_end_bin = chromosome_ends[chr_idx]

    for j in range(num_bins_m):
        # Append metadata for the current bin
        bin_metadata.append({"chrom": current_chr_name, "start": current_start})

        # Increment start position for the *next* bin
        current_start += bin_length

        # Check if this bin 'j' is the end of the current chromosome
        if j == current_chr_end_bin:
            # Reset start position for the new chromosome
            current_start = 0
            # Move to the next chromosome
            chr_idx += 1

            # If there are more chromosomes left to process, update names
            if chr_idx < len(chromosome_ends):
                current_chr_name = f"chr{chr_idx + 1}"
                current_chr_end_bin = chromosome_ends[chr_idx]

    if len(bin_metadata) != num_bins_m:
        raise RuntimeError(
            "Internal error: Bin metadata length does not match bin number."
        )

    print(f"Starting conversion for {num_cells_n} cells.")

    # --- 3. Write to TSV File ---
    header = ["CELL", "chrom", "start", "end", "CN states"]

    with open(output_filepath, 'w', newline='') as f:
        # Use csv.writer with tab delimiter for TSV
        tsv_writer = csv.writer(f, delimiter='\t')

        # Write the header row
        tsv_writer.writerow(header)

        # Iterate over every cell (N)
        for i in range(num_cells_n):
            cell_name = cell_ids[i]

            # Iterate over every bin (M)
            for j in range(num_bins_m):
                # Get pre-calculated bin metadata
                meta = bin_metadata[j]
                chrom = meta['chrom']
                start = meta['start']
                end = start + bin_length

                # Get CN states for haplotype A and B from (N, M, 2) array
                # cn_array[i, j, 0] = Haplotype A, Cell i, Bin j
                # cn_array[i, j, 1] = Haplotype B, Cell i, Bin j
                cn_a = cn_array[i, j, 0]
                cn_b = cn_array[i, j, 1]

                # Format the "a,b" string
                cn_state_str = f"{cn_a},{cn_b}"

                # Assemble and write the full row
                row = [cell_name, chrom, start, end, cn_state_str]
                tsv_writer.writerow(row)

    print(f"Successfully wrote DICE input file to: {output_filepath}")

def convert_dice_tsv_to_medicc2(dataset_path, out_path, out_filename=None, totalCN=False):
    """
    Function to convert DICE input tsv file to MEDICC2 input tsv file.
    Loads a DICE formatted tsv file specified by dataset_path and writes a MEDICC2 formatted tsv file to out_path.
    Adapted from DICE codebase: https://github.com/samsonweiner/DICE/blob/main/scripts/utilities.py
    Parameters
    ----------
    out_filename: str
        Name of the output MEDICC2 tsv file. If None, defaults to 'medicc2_input.tsv'.
    dataset_path: str
        Path to the DICE formatted tsv file.
    out_path: str
        Directory where the MEDICC2 formatted tsv file will be saved.
    totalCN: bool
        If True, only total copy number is considered. If False, allele-specific copy numbers are
    """
    out_path += '/' + out_filename if out_filename else '/medicc2_input.tsv'

    data = {}
    if totalCN:
        headers = ['sample_id', 'chrom', 'start', 'end', 'cn_a']
    else:
        headers = ['sample_id', 'chrom', 'start', 'end', 'cn_a', 'cn_b']
    f = open(dataset_path)
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        cell, chrom, start, end, CN = line[:-1].split('\t')
        chrom = chrom[chrom.index('r') + 1:]
        if totalCN:
            row = [cell, 'chrom' + chrom, start, end, CN]
        else:
            cn_a, cn_b = CN[:CN.index(',')], CN[CN.index(',') + 1:]
            row = [cell, 'chrom' + chrom, start, end, cn_a, cn_b]
        if cell not in data:
            data[cell] = [row]
        else:
            data[cell].append(row)

    f = open(out_path, 'w+')
    f.write('\t'.join(headers) + '\n')
    for cell, lines in data.items():
        for line in lines:
            f.write('\t'.join(line) + '\n')
    f.close()


def add_root(dice_tree_nx, healthy_cell_name):
    """Adds a root node to the unrooted DICE-inferred tree networkx object
     by rooting it between the healthy cell and its ancestor as described in the paper."""
    max_idx = dice_tree_nx.number_of_nodes()
    ancestor_healthy = list(dice_tree_nx.predecessors(healthy_cell_name))
    dice_tree_nx.remove_edge(ancestor_healthy[0], healthy_cell_name)
    dice_tree_nx.add_edge(str(max_idx), ancestor_healthy[0])
    dice_tree_nx.add_edge(str(max_idx), healthy_cell_name)
    return dice_tree_nx

def load_dice_tree(dice_output_path: str,
                   taxon_namespace,
                   cell_names,
                   healthy_cell_name='cell_0',
                   ) -> Tree:
    """Loads the DICE-inferred tree from the output file and adds a root node."""
    # Load newick
    dice_nwk_file_path = dice_output_path
    newick_str = open(dice_nwk_file_path).read().strip()
    # Convert to dendropy tree and make integer internal labels
    dice_tree_dpy: dpy.Tree = dpy.Tree.get(data=newick_str, schema='newick', taxon_namespace=taxon_namespace)
    tree_utils.label_tree(dice_tree_dpy)
    # Convert to networkx, add root, relabel cell names to integers
    dice_tree_nx = tree_utils.convert_dendropy_to_networkx(dice_tree_dpy)
    add_root(dice_tree_nx, healthy_cell_name=healthy_cell_name)
    dice_tree_nx = tree_utils.relabel_name_to_int(dice_tree_nx, cell_names)
    # Convert back to dendropy
    dice_tree_dpy2 = tree_utils.convert_networkx_to_dendropy(dice_tree_nx, taxon_namespace=taxon_namespace)
    return dice_tree_dpy2