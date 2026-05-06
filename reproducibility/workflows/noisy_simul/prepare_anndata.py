import sys

import anndata
import numpy as np
import pandas as pd

from cellmates.common_helpers.cnasim_data import profiles_to_anndata

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    adata = profiles_to_anndata(input, 'state')
    # join
    # make an anndata with concatenated profiles where chr is 1A and 1B
    chr_df = adata.var[['chr', 'start', 'end']].reset_index()
    n_bins = chr_df.shape[0]
    var_df = pd.concat([chr_df, chr_df], axis=0).reset_index(drop=True)
    max_chr = var_df['chr'].astype(int).max()
    var_df['chr'] = (var_df['chr'].astype(int) + ([0] * n_bins + [max_chr] * n_bins)).astype(str)
    var_df['phases'] = var_df['chr'].astype(str) + (['A'] * n_bins + ['B'] * n_bins)
    adata_joint = anndata.AnnData(
        X=np.concatenate([adata.layers['stateA'], adata.layers['stateB']], axis=1),
        var=var_df,
        obs=adata.obs
    )
    adata_joint.layers['state'] = adata_joint.X
    adata_joint.write_h5ad(output)
