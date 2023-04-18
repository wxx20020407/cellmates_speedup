from hmmlearn import hmm

class EM():
    """
    Runs the EM-algorithm for a quadruplet. Requires the copy number sequence of the root and observations of a pair
    of leaves.
    """

    def __init__(self, M, A, C_r, obs_1, obs_2):
        self.M = M
        self.A = A
        self.C_r = C_r
        self.obs_1 = obs_1
        self.obs_2 = obs_2

    def run(self):
        hmm.PoissonHMM(n_components=self.A,
                       startprob_prior=1.0,
                       transmat_prior=1.0,
                       lambdas_prior=0.0,
                       lambdas_weight=0.0,
                       algorithm='viterbi',
                       random_state=None,
                       n_iter=10,
                       tol=0.01,
                       verbose=False,
                       params='stl', init_params='stl', implementation='log')

