import os
import unittest

from cellmates.inference.em import EM
from cellmates.inference.pipeline import run_em_inference
from cellmates.models.evo import JCBModel
from cellmates.models.obs import PoissonModel
from cellmates.utils.testing import _generate_obs, create_output_test_folder


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = create_output_test_folder()

    def test_checkpoint_saving(self):
        # run fit quadruplet on dummy obs
        # generate toy data
        out_dir = self.tmp_dir
        n_states = 5
        obs, eps = _generate_obs(noise=10)
        n_cells = obs.shape[1]
        # run em
        evo_model = JCBModel(n_states)
        em = EM(n_states, PoissonModel(n_states, 100, 100), evo_model, tree_build='ctr', verbose=2, diagnostics=True)
        em.fit(obs, max_iter=30, rtol=1e-6, num_processors=1, checkpoint_path=out_dir)
        for v in range(n_cells):
            for w in range(v+1, n_cells):
                if v != w:
                    path = os.path.join(out_dir, f'_checkpoint_{v}_{w}.pkl')
                    self.assertTrue(os.path.exists(path))

    def test_checkpoint_wipeout(self):
        # run fit quadruplet on dummy obs
        # generate toy data
        out_dir = create_output_test_folder()
        n_states = 5
        obs, eps = _generate_obs(noise=10)
        n_cells = obs.shape[1]
        obs_model = PoissonModel(n_states, 100, 100)
        run_em_inference(
            obs=obs,
            chromosome_ends=[],
            n_states=n_states,
            alpha=1.,
            jc_correction=False,
            hmm_alg='broadcast',
            max_iter=10,
            rtol=1e-3,
            num_processors=1,
            obs_model=obs_model,
            verbose=True,
            save_diag=True,
            out_path=out_dir
        )
        # check that checkpoints are removed after inference
        for v in range(n_cells):
            for w in range(v+1, n_cells):
                if v != w:
                    path = os.path.join(out_dir, f'_checkpoint_{v}_{w}.pkl')
                    self.assertFalse(os.path.exists(path))

    def tearDown(self):
        # remove tmp dir and its contents
        for root, dirs, files in os.walk(self.tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))



if __name__ == '__main__':
    unittest.main()
