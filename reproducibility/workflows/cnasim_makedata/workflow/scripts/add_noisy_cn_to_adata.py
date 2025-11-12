"""
Only adds a noisy copy number layer to the AnnData object.
"""
import anndata
import os
import sys

from cellmates.common_helpers.cnasim_data import profiles_to_anndata

if __name__ == '__main__':
    path = sys.argv[1]  # path to benchmark datasets directory

    for dataset in os.listdir(path):
        if os.path.isdir(os.path.join(path, dataset)):
            for seed in os.listdir(os.path.join(path, dataset)):
                if os.path.isdir(os.path.join(path, dataset, seed)):
                    if 'clean_profiles.tsv' in os.listdir(os.path.join(path, dataset, seed)):
                        cnasim_path = os.path.join(path, dataset, seed)
                        # adata = anndata.read_h5ad(os.path.join(cnasim_path, 'anndata.h5ad'))
                        # cn_adata = profiles_to_anndata(os.path.join(cnasim_path, "profiles.tsv"), layer_name='noisy-cn')
                        # adata.layers['noisy-cn'] = cn_adata[adata.obs_names].layers['noisy-cn']
                        # adata.write_h5ad(os.path.join(cnasim_path, 'anndata.h5ad'))
                        print(f"done: {os.path.join(cnasim_path, 'anndata.h5ad')}")