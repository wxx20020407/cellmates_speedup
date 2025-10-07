# ==========================================================
# == Common rules for Quadruplet Accuracy Experiment      ==
# ==========================================================
import json
import os

# create all combinations
# COMBOS = list(itertools.product(
#     N_SITES_LIST,
#     range(NUM_REPLICATES),
#     range(len(LENGTH_SIZES_LIST)),
#     range(len(OBS_MODELS))
# ))
CELLMATES_PATH = os.environ.get("CELLMATES_PATH")
CELLMATES_PY = os.environ.get("CELLMATES_PY")
SCRIPTS_PATH = os.path.join(CELLMATES_PATH, "reproducibility/workflows/quadruplet_benchmark/scripts")

# Run a single experiment and save results to a temp file (which will be combined later)
rule run_experiment:
    output:
        temp("tmp/M{n_sites}_S{seed}_L{length}_O{obs}.csv")
    # conda: "envs/cellmates.yml"  # TODO: add env
    params:
        args=lambda wc, output: json.dumps(
            {
                "n_states": N_STATES,
                "max_iter": MAX_ITER,
                "seed": int(wc.seed),
                "n_sites": int(wc.n_sites),
                "sizes_dict": SIZES_DICT,
                "base_variance": BASE_VARIANCE,
                "obs_model": OBS_MODELS[int(wc.obs)],
                "length_size": LENGTH_SIZES_LIST[int(wc.length)],
                "out_file": output[0],
            }
        )

    shell: """
    {CELLMATES_PY} {SCRIPTS_PATH}/run_single_quad.py '{params.args}'
    """

# Combine all results into a single CSV
rule combine:
    input:
        expand(
            "tmp/M{n_sites}_S{seed}_L{length}_O{obs}.csv",
            n_sites=N_SITES_LIST,
            seed=list(range(NUM_REPLICATES)),
            length=list(range(len(LENGTH_SIZES_LIST))),
            obs=list(range(len(OBS_MODELS))),
        )
    output:
        "results.csv"
    shell:
        """
        head -n 1 {input[0]} > {output}
        tail -q -n +2 {input} >> {output}
        """

rule plot:
    input:
        "results.csv"
    output:
        "plots.pdf"
    shell:
        "Rscript {SCRIPTS_PATH}/plot_results.R {input} {output}"
