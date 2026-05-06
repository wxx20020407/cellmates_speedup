# Cellmates
> a method for single-cell phylogeny reconstruction with proper evolutionary distances from copy numbers

## Installation

- Python 3.10
- NumPy
- DendroPy
- NetworkX
- anndata

```bash
conda create -f environment.yml
conda activate cellmates
pip install .
```

## Execution

You can run Cellmates from the command line as follows:

```bash
cellmates --input <input_file> --output <output_path> [options]
```
For detailed usage instructions and available options, run:

```bash
cellmates --help
```

## Example

An example input file is provided in the `reproducibility` directory.
You can run Cellmates on this example data as follows:

```bash
cellmates --input reproducibility/demo.h5ad --output results/ --n-states 8 --num-processors 4 --predict-cn
```

The above command will produce the following output files in the `results/` directory:
- `distance_matrix.npy`: The computed pairwise triplet-distance matrix between cells (n_cells, n_cells, 3).
- `tree.nwk`: The reconstructed phylogenetic tree in Newick format (with cell names if provided in input).
- `predicted_copy_numbers.npy`: A named list with `data` (predicted copy numbers for each cell and internal node)
- and `labels` (cell and internal node names).
- `cell_names.txt`: A text file containing the names of the cells in the order they appear in the distance matrix.

The input data is a HDF5 AnnData file containing single-cell read counts.
In particular, the file should contain `layers/copy`, the cell-by-bin corrected read counts.
Additional information, such as `obs_names` (cell names) can also be included in the AnnData object.
Cell names will be used in the output tree.

The data should only provide tumor cells. If normal cells are present in the dataset, please remove them before
running Cellmates or add an annotation in `obs/normal` to indicate which cells are normal.
