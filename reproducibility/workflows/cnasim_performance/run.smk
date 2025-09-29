configfile: config.yaml
workdir: "output/reproducibility/workflows/cnasim_performance/"

# import parameters from config file
N_CELLS_LIST = config["n_cells_list"]
N_CHROM_LIST = config["n_chrom_list"]
N_BINS_LIST = config["n_bins_list"]
N_CLONES_LIST = config["n_clones_list"]
LAMDA_LIST = config["lamda_list"]
E1_LIST = config["e1_list"]
E2_LIST = config["e2_list"]
NUM_REPLICATES = config["num_replicates"]

rule all:
    input: 'results.csv', 'plots.pdf'

# simulate data with cnasim
rule simulate:
    output:
        dir=temp(expand(directory("data/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}/"),
            replicate=range(1,NUM_REPLICATES + 1),
            n_cells=N_CELLS_LIST,
            n_bins=N_BINS_LIST,
            n_clones=N_CLONES_LIST,
            lamda=LAMDA_LIST,
            e1=E1_LIST,
            e2=E2_LIST,
            n_chrom=N_CHROM_LIST
        ))
    params:
        replicate="{replicate}",
        n_cells="{n_cells}",
        n_bins="{n_bins}",
        n_clones="{n_clones}",
        lamda="{lamda}",
        e1="{e1}",
        e2="{e2}",
        n_chrom="{n_chrom}",
        dir="data/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}"
    conda: "envs/cnasim.yml"
    shell: """
    echo "Running: cnasim -m 1 --use-uniform-coverage -n {params.n_cells} -c {params.n_clones} -E1 {params.e1} -E2 {params.e2} \
     -P 8 -o {params.dir} -N {params.n_chrom} -N {params.n_bins} -L 100"
    mkdir -p {params.dir}
    touch {params.dir}/readcounts.tsv}
    """

rule prepare_anndata:
    input: "data/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}/readcounts.tsv"
    output: "data/dat_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.h5ad"
    params:
        replicate="{replicate}",
        n_cells="{n_cells}",
        n_bins="{n_bins}",
        n_clones="{n_clones}",
        lamda="{lamda}",
        e1="{e1}",
        e2="{e2}",
        n_chrom="{n_chrom}"
    conda: "envs/base.yml"
    shell: """
    echo "Converting CNAsim output to anndata"
    {VICTREE_PATH}/src/victree/bin/cnasimh5.py convert -i data/R{params.replicate}_N{params.n_cells}_M{params.n_bins}_K{params.n_clones}_L{params.lamda}_E1{params.e1}_E2{params.e2}_C{params.n_chrom}/ -o {output}
    """

rule run_cellmates:

rule combine_results:

# cnasim -m 1 --use-uniform-coverage -n 10 -n1 0.1 -c 2 --coverage 0.1 -i 1 -E1 0.04 -E2 0.1 -p1 0.01 -P 8 -o data/test   -N 10 -L 10000000
