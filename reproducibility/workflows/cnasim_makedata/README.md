# DICE benchmark CNAsim data workflow
> Reproduce the CNAsim datasets from DICE paper

The snakemake workflow has the purpose of simulating CN profiles and single-cell trees with the parameters
used in the DICE paper. Moreover, it generates further information such as ancestral profiles and bin read counts
that can be used for Cellmates analysis. Every dataset is also packed in an AnnData object for convenience.
The configuration file `config.yaml` contains all the parameters used for the simulations and matches the DICE paper settings,
but it can be modified to generate different datasets.

__NOTE__: currently, only noise-free datasets are simulated.

## Execution
To execute the workflow, run
```bash
snakemake --sdm conda -c32
```
This will create a conda environment for each rule and use up to 32 cores. The datasets will be generated in the `results/` folder
as follows
```
- results/
    - A1_0/                             # dataset with parameters A1 and noise level 0
        - 0/                            # replicate 0
            - anndata.h5ad
            - ancestral_profiles.tsv
            - readcounts.tsv
            - tree.nwk
        - 1/
        - 2/
        ...
    - A1_1/
    ...
```
