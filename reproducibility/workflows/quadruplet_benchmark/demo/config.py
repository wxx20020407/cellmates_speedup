# ===================================================
# == Quadruplet experiment configuration (FULL)   ==
# ===================================================

# ---------------------
# Experiment parameters
# ---------------------
N_STATES       = 7
MAX_ITER       = 60
NUM_REPLICATES = 2
N_SITES_LIST   = [200]
SIZES_DICT     = {
    's': 0.01,
    'm': 0.02,
    'l': 0.08,
}
BASE_VARIANCE  = 0.0001

# observation models
OBS_MODELS = [
    {"type": "normal", "mu_v_prior": 1.0, "tau_v_prior": 50},
    # {"type": "poisson", "lambda_v_prior": 100.},
]

# ---------------------
# Length-size patterns
# ---------------------
LENGTH_SIZES_LIST = (
        [[p] for p in SIZES_DICT] +
        [['s', 'l']] +  # small r->u, large u->v,w
        [['l', 's']]   # large r->u, small u->v,w
)
