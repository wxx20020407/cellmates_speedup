# ==================================================
# == Full Quadruplet Accuracy workflow             ==
# ==================================================
from config import (
    N_STATES, MAX_ITER, NUM_REPLICATES, N_SITES_LIST,
    SIZES_DICT, BASE_VARIANCE, OBS_MODELS,
    LENGTH_SIZES_LIST
)

#workdir: "output/reproducibility/workflows/quadruplet_benchmark/demo/"

rule all:
    input: 'results.csv', 'plots.pdf'

include: "../_common_rules.smk"
