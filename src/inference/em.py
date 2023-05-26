import itertools
from typing import Optional, Union

import numpy as np
from hmmlearn import hmm

from models.quadruplet import Quadruplet


class EM():
    """
    Runs the EM-algorithm for a quadruplet. Requires the copy number sequence of the root and observations of a pair
    of leaves.
    """

    def __init__(self, quadruplet: Quadruplet):
        self.quadruplet = quadruplet

    def run_hmmlearn(self):
        yv = self.quadruplet.yv
        yw = self.quadruplet.yv
        A = self.quadruplet.A
        y = np.concatenate([yv, yw])
        lengths = [len(yv), len(yw)]
        eps_trans_matrix_prior = np.ones((A, A, A)) / A * 0.05

        lambdas_prior = np.ones((A, 2))
        init_prior = np.zeros((3, A))
        init_prior[:, 2] = 1.
        model = hmm.PoissonHMM(n_components=(A, A, A),
                               startprob_prior=init_prior,
                               transmat_prior=eps_trans_matrix_prior,
                               lambdas_prior=lambdas_prior,
                               lambdas_weight=0.0,
                               algorithm='viterbi',
                               random_state=None,
                               n_iter=10,
                               tol=0.01,
                               verbose=False,
                               params='stl', init_params='', implementation='log')
        model.startprob_ = init_prior
        model.fit(y, lengths)