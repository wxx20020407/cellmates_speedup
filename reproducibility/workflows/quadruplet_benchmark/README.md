# Quadruplet Benchmark

This is a benchmark workflow for evaluating the performance of Cellmates' EM on lengths for single quadruplets made of
two cells, a centroid and the root node.
It consists of two files:
- `config.py`: defines the tasks and simulation parameters (e.g. size of sequences, number of replicates, etc.)
- `run.smk`: the Snakemake rules to run the benchmark. Does not need to be modified.

## Usage
To run the benchmark, run the following from the root directory of the Cellmates repository:
```bash
CELLMATES_PATH=$(pwd) snakemake -c32 -s reproducibility/workflows/quadruplet_benchmark/full/run.smk
```
This will run the benchmark with 32 CPU cores in parallel and save
the results in the `output/reproducibility/workflows/quadruplet_benchmark/full` directory.

Note: setting the `CELLMATES_PATH` environment variable is required for the workflow to find the Cellmates installation.
This will be changed when Cellmates is published on PyPI.

## Demo
A demo version of the benchmark with reduced parameters is available in `reproducibility/workflows/quadruplet_benchmark/demo`
for quick testing of the workflow to ensure dependencies are correctly installed and the workflow runs without errors.
To run the demo, execute the following command in the terminal:
```bash
snakemake -c4 -s reproducibility/workflows/quadruplet_benchmark/demo/run.smk
```
The demo will terminate in a few minutes.
