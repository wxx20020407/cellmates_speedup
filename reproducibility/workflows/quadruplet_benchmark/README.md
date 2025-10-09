# Quadruplet Benchmark

This is a benchmark workflow for evaluating the performance of Cellmates' EM on lengths for single quadruplets made of
two cells, a centroid and the root node.
It consists of two main files:
- `config/config.yaml`: defines the tasks and simulation parameters (e.g. size of sequences, number of replicates, etc.)
- `workflow/Snakefile`: the Snakemake rules to run the benchmark and related scripts. Does not need to be modified.

## Usage
To run the benchmark, just run the following from `quadruplet_benchmark` directory:
```bash
snakemake -c32 --sdm conda
```
This will run the benchmark with 32 CPU cores in parallel and save
the results in the `quadruplet_benchmark/results` directory (use `--directory` to change the output directory).

Note: the current implementation requires to have conda environments 'cellmates' and 'rplot' installed. You can
create them using the provided environment files in `quadruplet_benchmark/workflow/envs`. In future versions, this will be
handled automatically by Snakemake.

## Demo
A demo version of the benchmark with reduced parameters can be ran from the demo configuration file.
This is intended for quick testing of the workflow to ensure dependencies are correctly installed and the workflow runs without errors.
To run the demo, execute the following command in the terminal:
```bash
snakemake -c4 --configfile config/demo.yaml --sdm conda
```
The demo will terminate in a few minutes.
