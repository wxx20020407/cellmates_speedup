import itertools
import logging
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
from dendropy.calculate.treecompare import (
    robinson_foulds_distance,
    unweighted_robinson_foulds_distance,
    symmetric_difference,
)
from scipy.special import comb

from cellmates.inference.neighbor_joining import build_tree
from cellmates.models.obs import ObsModel, NormalModel, PoissonModel
from cellmates.simulation.datagen import rand_dataset

from cellmates.models.evo import EvoModel, CopyTree, JCBModel
from cellmates.utils.math_utils import l_from_p
from cellmates.utils.tree_utils import convert_networkx_to_dendropy, get_ctr_table


class EM:
    """
    Runs the EM-algorithm for a set of cells. Requires the copy number sequence of the root and observations
    (read counts) leaves.
    """
    def __init__(self, n_states: int = 7, obs_model: ObsModel | str = 'poisson',
                 evo_model: EvoModel | str = 'jcb', tree_build='ctr',
                 alpha=1., verbose: int = 0, diagnostics: bool = False):
        # model variables
        self.min_iter = 3
        if isinstance(evo_model, str):
            self.evo_model: EvoModel = JCBModel(n_states, alpha=alpha) if evo_model == 'jcb' else CopyTree(n_states)
        else:
            self.evo_model: EvoModel = evo_model
        if isinstance(obs_model, str):
            self.obs_model: ObsModel = PoissonModel(n_states=n_states) if obs_model == 'poisson' else NormalModel(n_states=n_states)
        else:
            self.obs_model: ObsModel = obs_model
        self.tree_build = tree_build  # algorithm for tree reconstruction
        self._n_sites = None
        self._n_cells = None
        self._n_states = n_states

        # output variables
        self._distances = None
        self._n_iterations = None
        self._loglikelihoods = None

        # Diagnostics
        self.diagnostics = diagnostics
        if self.diagnostics:
            self.diagnostic_data = {
                'loglikelihoods': [],
                'thetas': [],
                'psis': []
            }

        # set verbose level logger in the style of sklearn
        self.verbose = verbose
        self.logger = logging.getLogger(__class__.__name__)
        if self.verbose == 0:
            self.logger.setLevel(logging.ERROR)
        elif self.verbose == 1:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)


    def fit(self, X: np.ndarray, max_iter: int = 200, rtol: float = 1e-6, num_processors: int = 1,
            theta_init=None,
            psi_init=None,
            **kwargs):
        """
        Run the EM algorithm for the given observations X.
        Parameters
        ----------
        X array of shape (n_sites, n_cells)
        max_iter maximum number of iterations
        rtol relative tolerance for convergence
        num_processors number of processors to use for parallel
        theta_init initial values for the triplet parameters, if None, initialized to an average of 5 changes over the whole length
        kwargs additional keyword arguments
        """
        self.n_sites = X.shape[0]
        self.n_cells = X.shape[1]
        obs = X

        # if correction, change alpha to alpha / (n_states - 1)
        alpha = kwargs.get('alpha', 1.)
        alpha = alpha / (self.n_states - 1) if kwargs.get('jc_correction', False) else alpha
        # init to an average of 5 changes over the whole length if not provided
        p_init_default = 5 / self.n_sites
        theta_init_ = np.empty(3)
        if theta_init is not None:
            theta_init_[:] = theta_init
        else:
            theta_init_[:] = p_init_default if isinstance(self.evo_model, CopyTree) else l_from_p(p_init_default,
                                                                                                   self.n_states)

        l_hat = -np.ones((self.n_cells, self.n_cells, 3))
        zero_tol = kwargs.get('zero_tol', 1e-10)  # saturation level when dp << d (changes are much more prevalent)

        # for each pair of cells
        self.logger.debug(f'starting inference for {int(comb(self.n_cells, 2))} pairs, {self.n_states} states,'
                      f' {self.n_cells} cells, {self.n_sites} sites, {max_iter} max iterations, {rtol} rtol')
        iterations = {}
        loglikelihoods = {}

        # run inference for each pair of cells
        if num_processors > 1:
            self.logger.debug(f'using {num_processors} processors')
            # dispatch jobs to multiple processors using shared memory
            # create shared memory for observations, numpy array backed by shared memory and copy data
            shm_obs = shared_memory.SharedMemory(create=True, size=obs.nbytes)
            shared_obs = np.ndarray(obs.shape, dtype=obs.dtype, buffer=shm_obs.buf)
            np.copyto(shared_obs, obs)
            args = [(s, t, shm_obs.name, theta_init_, psi_init, alpha, max_iter, rtol, zero_tol, self.obs_model)
                    for s, t in itertools.combinations(range(self.n_cells), r=2)]
            with mp.Pool(num_processors) as pool:
                # main loop
                results = pool.starmap(self._fit_quadruplet_shared_mem, args)
            # close shared memory
            shm_obs.close()
        else:
            # single processor
            self.logger.debug(f'using single processor')
            results = []
            for s, t in itertools.combinations(range(self.n_cells), r=2):
                results.append(self._fit_quadruplet(s, t, obs[:, [s, t]], theta_init_, max_iter, rtol, psi_init))

        # collect results
        for (s, t), l_i, loglik, it in results:
            l_hat[s, t, :] = l_i
            iterations[(s, t)] = it
            loglikelihoods[(s, t)] = loglik

        # save result for later use
        self._distances = l_hat
        self._n_iterations = iterations
        self._loglikelihoods = loglikelihoods
        self.logger.info(f'finished in {len(iterations)} iterations')


    def _fit_quadruplet(self, v: int, w: int, obs_vw: np.ndarray,
                        theta_init: np.ndarray,
                        max_iter: int, rtol: float, psi_init: dict = None):
        # define quad logger with cell pair tag adding to the class logger
        logger = self.logger.getChild(f'{v},{w}')
        logger.setLevel(self.logger.level)

        # initialize l = (l_ru, l_uv, l_uw)
        theta_init_ = np.empty(3)  # init desired size
        theta_init_[:] = theta_init  # ...copy instead of referencing and validate input size with assignment
        self.obs_model.initialize(psi_init)
        # (`theta_init_ = theta_init` is wrong, but also `theta_init_ = theta_init.copy()` is prone to error
        quad_model = self.evo_model.new()
        quad_model.theta = theta_init_

        # compute changes is observation and evolution model specific
        d, dp, loglik = quad_model.multi_chr_expected_changes(obs_vw=obs_vw, obs_model=self.obs_model)
        convergence = False
        if self.diagnostics:
            self.diagnostic_data = {'loglikelihoods': [loglik], 'thetas': [theta_init_.copy()], 'psis': [self.obs_model.psi_array()]}

        it = 0
        logger.debug(f'[{it}/{max_iter}] LL = {loglik} d = {d} dp = {dp}')
        while not convergence and it < max_iter:

            # ---------- M-step ----------
            # Evolution model parameter update
            quad_model.update(exp_changes=d, exp_no_changes=dp)

            # Observation model parameter update
            one_slice_marginals_v, one_slice_marginals_w = quad_model.get_one_slice_marginals()
            self.obs_model.update(obs_vw, (one_slice_marginals_v, one_slice_marginals_w))

            # ---------- E-step ----------
            # compute D and D'
            d, dp, new_loglik = quad_model.multi_chr_expected_changes(obs_vw=obs_vw, obs_model=self.obs_model)
            logger.debug(f"[{it + 1}/{max_iter}] LL = {new_loglik}, d = {d}, dp = {dp}")


            if new_loglik < loglik:
                logger.error(f'log likelihood decreased: {new_loglik} < {loglik}')
            elif (new_loglik - loglik) / np.abs(loglik) < rtol and it > self.min_iter:
                convergence = True
            loglik = new_loglik

            if self.diagnostics:
                self.diagnostic_data['loglikelihoods'].append(loglik)
                self.diagnostic_data['thetas'].append(quad_model.theta.copy())
                self.diagnostic_data['psis'].append(self.obs_model.psi_array())

            it += 1

        if it == max_iter and not convergence:
            logger.warning(f'did not converge after {max_iter} iterations')
        else:
            logger.debug(f'converged after {it} iterations')

        return (v, w), quad_model.theta, loglik, it


    def _fit_quadruplet_shared_mem(self, v: int, w: int, shared_obs_mem_name: str,
                                   l_init: np.ndarray, psi_init: dict,
                                   alpha: float, max_iter: int, rtol: float, zero_tol: float,
                                   obs_model: ObsModel) -> (tuple, np.ndarray, float, int):
        """
        Pairwise EM algorithm for a pair of cells v, w with shared observations to be used in multiprocessing
        """
        shm = shared_memory.SharedMemory(name=shared_obs_mem_name)
        obs_vw = np.ndarray((self.n_sites, self.n_cells), dtype=np.float64, buffer=shm.buf)[..., [v, w]]
        return self._fit_quadruplet(v, w, obs_vw, l_init, max_iter, rtol, psi_init)

    def transform(self):
        # alternative method for the distances getter
        return self.distances

    def fit_transform(self, X):
        self.fit(X)
        return self._distances

    def compute_pair_likelihood(self, obs_vw, theta: np.ndarray = None, psi: dict = None) -> float:
        """
        Compute the log likelihood of the observations given the model parameters.
        """
        # run forward algorithm
        psi = self.obs_model.psi if psi is None else psi
        _, _, loglik, _ = self._fit_quadruplet(0, 1, obs_vw, theta, max_iter=0, rtol=0, psi_init=psi)
        return loglik

    @property
    def n_sites(self):
        if self._n_sites is None:
            raise AttributeError("Number of sites is not set.")
        return self._n_sites

    @n_sites.setter
    def n_sites(self, value):
        self._n_sites = value

    @property
    def n_cells(self):
        if self._n_cells is None:
            raise AttributeError("Number of cells is not set.")
        return self._n_cells

    @n_cells.setter
    def n_cells(self, value):
        self._n_cells = value

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, value):
        self._n_states = value

    @property
    def distances(self):
        if self._distances is None:
            raise AttributeError("Distances not set. Run `fit` or `fit_transform` first.")
        return self._distances

    @property
    def n_iterations(self):
        if self._n_iterations is None:
            raise AttributeError("Number of iterations not set. Run `fit` or `fit_transform` first.")
        return self._n_iterations

    @property
    def loglikelihoods(self) -> dict[tuple[int, int], float]:
        if self._loglikelihoods is None:
            raise AttributeError("Loglikelihoods not set. Run `fit` or `fit_transform` first.")
        return self._loglikelihoods

def jcb_em_ctrtable(obs: np.ndarray, n_states: int = 7, alpha=1., l_init=None, max_iter: int = 200, rtol: float = 1e-6,
                    jc_correction: bool = False, num_processors: int = 1) -> np.ndarray:
    """
    Run the JCB EM algorithm to estimate the centroid-to-root distances for each pair of cells. Wrapper function
    that only returns the centroid-to-root distances.
    """
    return jcb_em_alg(obs, n_states, alpha, l_init, max_iter, rtol, jc_correction, num_processors)['l_hat']

def jcb_em_alg(obs: np.ndarray, n_states: int = 7, alpha=1., l_init=None, max_iter: int = 200, rtol: float = 1e-6,
               jc_correction: bool = False, num_processors: int = 1, lam=100) -> dict[str, np.ndarray | dict[tuple[int, int], int | float]]:
    """
Implementation of JCB EM algorithm in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    alpha float, alpha parameter for the JCB model, length scaling factor
    l_init array of shape (3,) with initial values for the triplet parameters, if None, initialized to an average of 5 changes over the whole length
    max_iter int, maximum number of EM iterations (updates)
    rtol float, relative tolerance for convergence
    jc_correction if True, use Jukes-Cantor correction i.e. sets alpha = alpha / (n_states - 1)
    num_processors int, number of processors to use for parallel
    Returns
    -------
    dict with keys 'l_hat', 'iterations', 'loglikelihoods'
    'l_hat' array of shape (n_cells, n_cells, 3), estimated triplet distances (upper triangular, all other entries are -1)
    'iterations' dict with keys (v, w) and values number of iterations until convergence
    'loglikelihoods' dict with keys (v, w) and values log likelihood of the observations
    """
    logging.warning('outdated function, use the new class EM instead')
    em = EM(n_states=n_states, obs_model='poisson', evo_model='jcb', alpha=alpha)
    em.fit(obs, max_iter=max_iter, rtol=rtol, num_processors=num_processors, jc_correction=jc_correction, theta_init=l_init)
    return {
        'l_hat': em.distances,
        'iterations': em.n_iterations,
        'loglikelihoods': em.loglikelihoods
    }

def em_alg(obs: np.ndarray, n_states: int = 7, eps_init=None, max_iter: int = 200, rtol: float = 1e-6,
           num_processors: int = 1) -> dict[str, np.ndarray | dict[tuple[int, int], int | float]]:
    """
    CopyTree
    """
    logging.warning('outdated function, use the new class EM instead')
    if eps_init is None:
        eps_init = np.array([0.01] * 3)
    em = EM(n_states=n_states, obs_model='poisson', evo_model='copytree')
    em.fit(obs, max_iter=max_iter, rtol=rtol, num_processors=num_processors, l_init=eps_init)
    return {
        'l_hat': em.distances,
        'iterations': em.n_iterations,
        'loglikelihoods': em.loglikelihoods
    }


if __name__ == '__main__':
    seed = 42
    logging.basicConfig(level=logging.DEBUG)
    # test EM algorithm
    n_cells = 4
    n_states = 7
    n_sites = 200
    data = rand_dataset(n_states, n_sites, obs_model='poisson', p_change=0.05, n_cells=n_cells, seed=seed)
    # true ctr_table
    true_ctr_table = get_ctr_table(data['tree'])

    start_time = time.time()
    em = EM(n_states,
            obs_model=PoissonModel(n_states=n_states, lambda_v_prior=100),
            evo_model=JCBModel(n_states=n_states, alpha=1.),
            verbose=2)
    em.fit(data['obs'], max_iter=50, num_processors=1, jc_correction=False)
    # jcb_out_dict = jcb_em_alg(data['obs'], n_states=n_states, max_iter=50, jc_correction=False, num_processors=5)
    print(f"Total time: {time.time() - start_time}")
    print(f"Instance: {n_cells} cells, {n_states} states, {n_sites} sites")
    print("True tree")
    data['tree'].print_plot(plot_metric='length')

    ctr_table = em.distances
    print("JCB EM output:")
    loglikelihoods = em.loglikelihoods
    n_iterations = em.n_iterations
    for (v, w) in loglikelihoods.keys():
        print(f"Pair ({v}, {w}): {loglikelihoods[(v, w)]}, {n_iterations[(v, w)]} iterations")

    print("True tree")
    data['tree'].print_plot(plot_metric='length')

    # build tree with em table
    nx_em_tree = build_tree(ctr_table)
    em_tree = convert_networkx_to_dendropy(nx_em_tree, taxon_namespace=data['tree'].taxon_namespace,
                                           edge_length='length')
    print("EM tree")
    em_tree.print_plot(plot_metric='length')
    print("EM tree (unweighted)")
    em_tree.print_plot()

    print(f"Symmetric unweighted difference: {symmetric_difference(data['tree'], em_tree)}")
    print(f"Unweighted Robinson-Foulds distance: {unweighted_robinson_foulds_distance(data['tree'], em_tree)}")
    print(f"Robinson-Foulds distance: {robinson_foulds_distance(data['tree'], em_tree, edge_weight_attr='length')}")
    print(f"CTR table difference: {np.linalg.norm(true_ctr_table - ctr_table)}")
