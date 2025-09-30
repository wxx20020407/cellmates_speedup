configfile: "config.yaml"
workdir: "output/reproducibility/workflows/cnasim_performance/"

import os

# parameters
N_CELLS_LIST = config["N_CELLS"]
N_CHROM_LIST = config["N_CHROM"]
N_BINS_LIST = config["N_BINS"]
LAMDA_LIST = config["N_EVENT_PER_EDGE"]
N_CLONES_LIST = config["N_CLONES"]
E1_LIST = config["BOUNDARY_NOISE"]
E2_LIST = config["JITTER"]
N_REPLICATES = config["N_REPLICATES"]
NORMAL_FRACTION = config["NORMAL_FRACTION"]
BIN_LENGTH = 1000000  # 1Mb bins
CN_LENGTH_MEAN = 10000000  # 10Mb average CNA length

envvars:
    "CELLMATES_PATH",   # e.g. "/path/to/cellmates"
    "CELLMATES_PY"      # e.g. "/path/to/cellmates/venv/bin/python"
CELLMATES_PATH = os.environ["CELLMATES_PATH"]
CELLMATES_PY = os.environ["CELLMATES_PY"]

wildcard_constraints:
    replicate=r"\d+",
    n_cells=r"\d+",
    n_bins=r"\d+",
    n_clones=r"\d+",
    lamda=r"\d+",
    e1=r"[\d.]+",
    e2=r"[\d.]+",
    n_chrom=r"\d+"

rule all:
    input: 'results.csv', 'plots.pdf'

# simulate data with cnasim
rule simulate:
    output:
        # dir=temp(directory("data/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}/")),
        counts=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cnasim_tmp", "readcounts.tsv")
    conda: "envs/cnasim.yml"
    params:
        replicate=lambda wc: int(wc.replicate),
        n_cells=lambda wc: int(wc.n_cells),
        n_bins=lambda wc: int(wc.n_bins),
        n_clones=lambda wc: int(wc.n_clones),
        lamda=lambda wc: int(wc.lamda),
        e1=lambda wc: float(wc.e1),
        e2=lambda wc: float(wc.e2),
        n_chrom=lambda wc: int(wc.n_chrom),
        # chrom_length = n_bins * bin_length // n_chrom
        chrom_length=lambda wc: int(int(wc.n_bins) * BIN_LENGTH / int(wc.n_chrom)),  # assuming bin size of 100kb
        normal_frac=lambda wc: float(NORMAL_FRACTION),
        # use lambda functions to convert to int/float
        out_dir=subpath(output.counts, parent=True)
    shell: """
    cnasim -m 1 --use-uniform-coverage -n {params.n_cells} -c {params.n_clones} -E1 {params.e1} -E2 {params.e2} \
     -o {params.out_dir} -N {params.n_chrom} -L {params.chrom_length} -n1 {params.normal_frac} -B {BIN_LENGTH} --cn-length-mean {CN_LENGTH_MEAN} --WGD --cn-copy-param 0.8
    """

rule prepare_anndata:
    input:
        counts=os.path.join("data","R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cnasim_tmp", "readcounts.tsv")
    output: os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "input.h5ad")
    params:
        cnasim_dir=subpath(input.counts, parent=True)
    conda: "envs/base.yml"
    shell: """
    echo "Converting CNAsim output to anndata"
    python {CELLMATES_PATH}/src/cellmates/common_helpers/cnasim_data.py -i {params.cnasim_dir} -o {output}
    """

rule run_cellmates:
    input: os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "input.h5ad")
    output:
        cm_dist=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cm_out", "distance_matrix.npy"),
        cm_tree=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cm_out", "tree.nwk")
    params:
        directory=subpath(output.cm_dist, parent=True)
    conda: "cellmates"
    shell: """
    echo "Running Cellmates"
    python {CELLMATES_PATH}/src/cellmates/bin/core.py -i {input} -o {params.directory} -v 2
    """

rule evaluate:
    input:
        truth_ad=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "input.h5ad"),
        cm_dist=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cm_out", "distance_matrix.npy"),
        cm_tree=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "cm_out", "tree.nwk")
    output: os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "eval_tmp.csv")
    conda: "envs/base.yml"
    script: "scripts/evaluate_results.py"

rule combine_results:
    input:
        expand(os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "eval_tmp.csv"),
            replicate=range(1,N_REPLICATES + 1), n_cells=N_CELLS_LIST, n_bins=N_BINS_LIST, n_clones=N_CLONES_LIST, lamda=LAMDA_LIST, e1=E1_LIST, e2=E2_LIST, n_chrom=N_CHROM_LIST
        )
    output: "results.csv"
    shell: """
    echo "Combining results"
    head -n 1 {input[0]} > {output}
    for f in {input}; do
        tail -n +2 $f >> {output}
    done
    """
#
# rule plot_input_data:
#     input: os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "input.h5ad")
#     output:
#         cn=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "tree_cn_plot.png"),
#         reads=os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "tree_reads_plot.png")
#     # conda: "cellmates"
#     script: "scripts/plot_input.R"

rule plot_results:
    input: "results.csv"
    output: "plots.pdf"
    conda: "envs/base.yml"
    script: "scripts/plot_results.py"
