import logging
import subprocess
import anndata


def anndata_to_dice_tsv(adata, output_prefix):
    """
    Input File Format: DICE** takes as input a single file (specified using the –i command line option)
    containing a tab-separated values (TSV) file describing the copy number profiles of all cells.
    The following headers are required for each file and should be placed on the first line:
        CELL (the cell id of the current row),
        chrom (the chromosome X of the current row in the form of chrX),
        start (the starting location in bp of the copy number bin),
        end (the ending location in bp of the copy number bin),
        CN states (the actual copy number of the bin in the current row).
    If total copy numbers are used, the value of CN states should be a single numerical value.
    If allele-specific copy numbers are used, the value of CN states should be a,b where a is the copy number for haplotype A, and b is the copy number for haplotype B.

    E.g.

    CELL     chrom   start          end       CN states
    leaf1    chr1       0           10000     1,1
    leaf2    chr1       0           10000     1,2
    leaf5    chr3      50000        60000     3,4
    """
    import pandas as pd

    # Extract the necessary data from AnnData
    obs = pd.DataFrame(adata.X, index=[f'cell_{i}' for i in range(adata.n_obs)])
    states = pd.DataFrame(adata.layers['state'], index=[f'cell_{i}' for i in range(adata.n_obs)])

    # Save observation data
    obs.to_csv(f'{output_prefix}_obs.tsv', sep='\t', header=False, index=True)

    # Save state data
    states.to_csv(f'{output_prefix}_states.tsv', sep='\t', header=False, index=True)



def run_dice(dataset_path, out_path=None, method='star', tree_rec='balME'):
    """

    Parameters
    ----------
    dataset_path
    method
    tree_rec: balME, olsME, NJ or uNJ

    Returns
    -------

    """
    # Load dataset


    # Prepare command to run DICE
    dice_command = f'dice -i {dataset_path}'
    dice_command += f' -o {out_path}' if out_path else ''
    dice_command += ' -b' if (method == 'bar' or method == 'b') else ''
    dice_command += f' -m {tree_rec}'
    logging.info(f'Running dice: {dice_command}')
    subprocess.run(dice_command, shell=True)
