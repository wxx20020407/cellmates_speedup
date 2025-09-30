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
        counts=temp(os.path.join("data", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "readcounts.tsv"))
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
        chrom_length=lambda wc: int(int(wc.n_bins) * 100000 / int(wc.n_chrom)),  # assuming bin size of 100kb
        normal_frac=lambda wc: float(NORMAL_FRACTION),
        # use lambda functions to convert to int/float
        out_dir=subpath(output.counts, parent=True)
    log:
        stdout="logs/cnasim/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.log",
        stderr="logs/cnasim/R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.err"
    shell: """
    cnasim -m 1 --use-uniform-coverage -n {params.n_cells} -c {params.n_clones} -E1 {params.e1} -E2 {params.e2} \
     -o {params.out_dir} -N {params.n_chrom} -L {params.chrom_length} -n1 {params.normal_frac} > {log.stdout} 2> {log.stderr}
    """

rule prepare_anndata:
    input: os.path.join("data","R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "readcounts.tsv")
    output: os.path.join("data", "dat_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.h5ad")
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
    python {CELLMATES_PATH}/src/cellmates/common_helpers/cnasim_data.py -i data/R{params.replicate}_N{params.n_cells}_M{params.n_bins}_K{params.n_clones}_L{params.lamda}_E1{params.e1}_E2{params.e2}_C{params.n_chrom}/ -o {output}
    """

rule run_cellmates:
    input: os.path.join("data", "dat_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.h5ad")
    output:
        cm_dist=os.path.join("cm_out","R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}","distance_matrix.npy"),
        cm_tree=os.path.join("cm_out","R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}","tree.nwk")
    params:
        directory=subpath(output.cm_dist, parent=True)
    conda: "cellmates"
    shell: """
    echo "Running Cellmates"
    python {CELLMATES_PATH}/src/cellmates/bin/core.py -i {input} -o {params.directory}
    """

rule evaluate:
    input:
        truth_ad=os.path.join("data", "dat_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.h5ad"),
        # TODO: adjust if file name change
        cm_dist=os.path.join("cm_out", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "distance_matrix.npy"),
        cm_tree=os.path.join("cm_out", "R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}", "tree.nwk")
    output: temp("eval_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.tsv")
    conda: "envs/base.yml"
    script: "scripts/evaluate_results.py"

rule combine_results:
    input:
        expand("eval_R{replicate}_N{n_cells}_M{n_bins}_K{n_clones}_L{lamda}_E1{e1}_E2{e2}_C{n_chrom}.tsv",
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

rule plot_results:
    input: "results.csv"
    output: "plots.pdf"
    conda: "envs/base.yml"
    script: "scripts/plot_results.py"
