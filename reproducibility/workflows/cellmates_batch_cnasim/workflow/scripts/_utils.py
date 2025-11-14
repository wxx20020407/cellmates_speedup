import numpy as np
import pandas as pd
import anndata

def adata_to_df(adata_path, layer_name='noisy-cn') -> pd.DataFrame:
    # create a DataFrame with the required columns
    adata = anndata.read_h5ad(adata_path)
    if layer_name not in adata.layers.keys():
        layer_name = 'state'
    n_obs = adata.n_obs
    n_var = adata.n_vars
    out_df = pd.DataFrame({
        'chr': np.tile(adata.var['chr'].values, n_obs),
        'start': np.tile(adata.var['start'].values, n_obs).astype(int),
        'end': np.tile(adata.var['end'].values, n_obs).astype(int),
        'cell': np.repeat(adata.obs_names.values, n_var),
        'A': np.nan_to_num(adata.layers[layer_name]).flatten().astype(int),
        'B': np.nan_to_num(adata.layers[layer_name]).flatten().astype(int),
    })
    return out_df


