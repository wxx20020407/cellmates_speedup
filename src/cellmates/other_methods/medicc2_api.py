import logging
import subprocess
import dendropy as dpy

from cellmates.utils import tree_utils


def load_medicc2_tsv_file(medicc2_tsv_path):
    """
    Load MEDICC2 formatted TSV file into a suitable data structure.
    Parameters
    ----------
    medicc2_tsv_path : str
        Path to the MEDICC2 formatted TSV file.

    Returns
    -------
    dict
        A dictionary where keys are cell IDs and values are lists of tuples (chromosome, start, end, cn_a, cn_b).
    """
    data = {}


def run_medicc2(dataset_path, out_dir_path, topology_only=False, num_proc=1):
    # Prepare command to run MEDICC2
    medicc2_command = f'medicc2 {dataset_path} {out_dir_path} -j "{num_proc}"'
    medicc2_command += '--topology-only ' if topology_only else ''
    medicc2_command += ' --plot auto'
    # Run MEDICC2
    logging.info(f'Running dice: {medicc2_command}')
    subprocess.run(medicc2_command, shell=True)


def load_medicc2_tree(medicc2_nwk_file_path: str,
                      taxon_namespace: dpy.TaxonNamespace,
                      N_cells,
                      remove_diploid=True) -> dpy.Tree:
    medicc2_tree_nw = open(medicc2_nwk_file_path).read().strip()
    medicc2_tree_dpy: dpy.Tree = dpy.Tree.get(data=medicc2_tree_nw,
                                              schema='newick', taxon_namespace=taxon_namespace)
    leaves_mapping = {f'cell {i}': str(i) for i in range(N_cells)}
    leaves_mapping['diploid'] = str(N_cells)
    tree_utils.relabel_dendropy(medicc2_tree_dpy, leaves_mapping)
    # Remove healthy root if present
    if remove_diploid and medicc2_tree_dpy.find_node_with_taxon_label(str(N_cells)) is not None:
        medicc2_tree_dpy.prune_subtree(medicc2_tree_dpy.find_node_with_taxon_label(str(N_cells)))

    medicc2_tree_nx = tree_utils.convert_dendropy_to_networkx(medicc2_tree_dpy)
    medicc2_tree_dpy2 = tree_utils.convert_networkx_to_dendropy(medicc2_tree_nx,
                                                                taxon_namespace=taxon_namespace)
    return medicc2_tree_dpy2