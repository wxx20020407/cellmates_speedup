import os
import unittest
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
import scgenome.plotting as pl
import anndata

from utils.visual import plot_cn_profile

class VisualTestCase(unittest.TestCase):

    def setUp(self):
        # initialize test folder for saving plots
        self.test_folder = 'output_visual'
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

    def test_plot_cn_profile(self):

        fig, ax = plt.subplots()
        plot_cn_profile(np.random.randint(0, 6, (10, 100)), ax=ax)
        fig.savefig(self.test_folder + '/cn_profile.png')
        plt.close(fig)
        assert os.path.exists(self.test_folder + '/cn_profile.png')

    def test_plot_cn_profile_scgenome(self):

        tree = Phylo.read(StringIO("((0:0.1,1:0.1):0.2,(2:0.15,3:0.15):0.15);"), "newick")
        cn_matrix = np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 3, 3, 3, 3, 4, 4, 3, 3],
            [2, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 3, 3, 2]
        ])
        adata = anndata.AnnData(
            X=cn_matrix,
            var=pd.DataFrame(data={
                'chr': ['1'] * cn_matrix.shape[1],
                'start': list(range(0, cn_matrix.shape[1] * 100, 100)),
                'end': list(range(100, (cn_matrix.shape[1] + 1) * 100, 100))
            }),
        )
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        assert adata.n_obs == cn_matrix.shape[0]
        assert adata.n_vars == cn_matrix.shape[1]
        g = pl.plot_cell_cn_matrix_fig(adata, layer_name=None, tree=tree)
        g['fig'].savefig(self.test_folder + '/cn_profile_scgenome.png')
        plt.close(g['fig'])
        assert os.path.exists(self.test_folder + '/cn_profile_scgenome.png')

    def test_plot_rand_anndata(self):
        from simulation.datagen import rand_ann_dataset
        adata = rand_ann_dataset(n_cells=10, n_states=5, n_sites=50, seed=1234, p_change=0.05)
        tree = Phylo.read(StringIO(adata.uns['tree']), "newick")
        assert adata.n_obs == 10
        assert adata.n_vars == 50
        assert 'state' in adata.layers
        assert 'tree' in adata.uns
        g = pl.plot_cell_cn_matrix_fig(adata, layer_name='state', tree=tree)
        g['fig'].savefig(self.test_folder + '/cn_profile_rand_anndata.png')
        plt.close(g['fig'])
        assert os.path.exists(self.test_folder + '/cn_profile_rand_anndata.png')

        g = pl.plot_cell_cn_matrix_fig(adata, layer_name=None, tree=tree, raw=True)
        g['fig'].savefig(self.test_folder + '/cn_profile_rand_anndata_raw.png')
        plt.close(g['fig'])
        assert os.path.exists(self.test_folder + '/cn_profile_rand_anndata_raw.png')

