# ===================================================
# == Quadruplet experiment configuration (FULL)   ==
# ===================================================

# ---------------------
# Experiment parameters
# ---------------------
N_STATES       = 7
MAX_ITER       = 60
N_REPLICATES = 20
N_SITES_LIST   = [200, 500, 1000]
SIZES_DICT     = {
    'xs': 0.005,
    's': 0.01,
    'm': 0.02,
    'l': 0.08,
    'xl': 0.2
}
BASE_VARIANCE  = 0.0001

# observation models
OBS_MODELS = [
    {"type": "normal", "mu_v_prior": 1.0, "tau_v_prior": 50},
    {"type": "normal", "mu_v_prior": 1.0, "tau_v_prior": 100},
    {"type": "normal", "mu_v_prior": 1.0, "tau_v_prior": 300},
]

# ---------------------
# Length-size patterns
# ---------------------
LENGTH_SIZES_LIST = (
        [[p] for p in SIZES_DICT] +
        [['xs', 'xl'], ['s', 'l']] +  # small r->u, large u->v,w
        [['xl', 'xs'], ['l', 's']] +  # large r->u, small u->v,w
        [['m', 'xs', 'xl'], ['m', 's', 'l']] +  # medium r->u, small u->v, large u->w
        [['xl', 'm', 'xs'], ['l', 'm', 's']] # large r->u, medium u->v, small u->w
)
