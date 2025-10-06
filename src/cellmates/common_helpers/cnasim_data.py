"""
Helpers for CNASim data processing.
"""
import argparse
import logging
import os
import ast
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from Bio import Phylo
import anndata

# ==================================
# Main functions (read CNAsim files)
# ==================================
def convert_cnasim_output_to_anndata(cnasim_data_dir: str, out_path: str = None):
    """
    Convert CNAsim output files to anndata object and save it to out_path
    """
    adata = load_cnasim_output_files(cnasim_data_dir)
    if out_path is None:
        out_path = os.path.join(cnasim_data_dir, 'adata.h5ad')
    adata.write(Path(out_path))
    return out_path


def get_sc_tree(cnasim_data_path: str, cell_types_df: pd.DataFrame) -> nx.DiGraph:
    """
    Parse tree.nwk file to get single cell tree in networkx format
     add the clone assignment id as node attribute 'clone', and 'is_founder' for founder nodes.
    Clones are either 'normal' or 'founder' (progenitor clone) and 'cloneX' for the rest.
    Parameters:
    - cnasim_data_path: path to CNAsim output directory
    - cell_types_df: DataFrame with ancestor to clone assignment (ancestor, clone, is_cell, is_founder, clone_id, clone_int)
    """
    tree_file = os.path.join(cnasim_data_path, 'tree.nwk')
    tree_nx = _parse_newick(tree_file)
    # set nodes attributes to clone founders
    for i, row in cell_types_df.iterrows():
        n = row['ancestor']
        tree_nx.nodes[n]['clone'] = row['clone_id']
        tree_nx.nodes[n]['is_founder'] = row['is_founder']
    return tree_nx

def read_log_params(cnasim_data_path: str) -> dict:
    params = {}
    with open(os.path.join(cnasim_data_path, 'log.txt'), 'r') as f:
        for line in f:
            if line != '\n':
                k, v = line.strip().split(':')
                try:
                    if k == 'out_path':
                        params[k] = v
                    else:
                        params[k] = ast.literal_eval(v)
                except SyntaxError:
                    params[k] = v.strip()
                except ValueError as e:
                    print(f"Error parsing value {v} for key {k}: {e}")
    return params

def read_cell_types(cnasim_data_path: str) -> (np.ndarray, pd.DataFrame):
    """
    Parse cell_types.tsv file to get cell to clone assignment, and ancestor to clone assignment.
    CNAsim generates a normal, a founder clone and a number of clones (clones1, clone2, ...).
    root is the ancestor associated with the normal clone and the founder is the first clone in the list.
    Cell labeling follows the same pattern: cell1, cell2, ... cellN with idx starting from 0.
    Clones will be sorted by their id number but remapped to a range(num_clones) with
    clone 0 being the normal clone and clone 1 the founder clone.

    Returns:
    - cell_assignment: a 1D array with the clone id for each cell
    - ancestor_clone_map: a dictionary mapping ancestor names to clone ids
    """
    clone_id_map = {'normal': 0, 'founder': 1}

    cell_types = pd.read_csv(os.path.join(cnasim_data_path, 'cell_types.tsv'), delimiter='\t', names=['ancestor', 'clone'])
    cell_types['is_cell'] = cell_types['ancestor'].transform(lambda x: 'cell' in x)
    cell_types['is_founder'] = cell_types['clone'].transform(lambda x: 'founder' in x)
    cell_types['clone'] = cell_types['clone'].transform(lambda x: x.split('_')[0])  # remove '_founder' suffix
    cell_types['clone_int'] = cell_types['clone'].transform(lambda x: int(x[5:]) if 'clone' in x else -1)

    # reassign a clone id to make sure they are in range(num_clones)
    # unique will merge normal with founder (which has id -1 for convenience)
    unique_clones = cell_types['clone_int'].sort_values().unique().tolist()
    # print(unique_clones)
    for i, c in enumerate(unique_clones):
        if c == -1:
            continue
        clone_name = f'clone{c}'
        clone_id_map[clone_name] = i + 2 # normal is 0, founder is 1
    cell_types['clone_id'] = cell_types['clone'].transform(lambda x: clone_id_map[x])
    # map cell to clone
    cell_assignment = np.zeros(cell_types['is_cell'].sum(), dtype=int)
    for i, row in cell_types[cell_types['is_cell']].iterrows():
        cell_id = int(row['ancestor'][4:]) - 1  # cell1 -> 0, cell2 -> 1, ...
        cell_assignment[cell_id] = clone_id_map[row['clone']]

    return cell_assignment, cell_types

def read_cn_profiles(adata: anndata.AnnData, cnasim_data_path: str, n_cells: int, inplace=True) -> np.ndarray:
    """
    If inplace=True, adds layers ['state', 'Astate', 'Bstate'] to the anndata
    Returns the total copy number state
    """
    profiles_file = 'clean_profiles.tsv' if os.path.exists(os.path.join(cnasim_data_path, 'clean_profiles.tsv')) else 'profiles.tsv'
    profiles_df = pd.read_csv(os.path.join(cnasim_data_path, profiles_file), delimiter='\t', header=0)
    # mutate copy number
    profiles_df[['Acn', 'Bcn']] = profiles_df['CN states'].str.split(',', expand=True).astype(int)
    profiles_df['cn'] = profiles_df['Acn'] + profiles_df['Bcn']
    #profiles_df['cn'] = profiles_df['CN states'].transform(lambda x: sum(list(map(int, x.split(',')))))

    wide_cn_df = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='cn', columns='CELL',
                                sort=False)  # already sorted by cell number
    # extract cell columns sorted by cell number
    cells = [f'cell{i+1}' for i in range(n_cells)]
    cn_profiles = wide_cn_df[cells].transpose().to_numpy()

    if inplace:
        adata.layers['state'] = cn_profiles
        adata.layers['Astate'] = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='Acn', columns='CELL',
                                                sort=False)[cells].transpose().to_numpy()
        adata.layers['Bstate'] = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='Bcn', columns='CELL',
                                                sort=False)[cells].transpose().to_numpy()
    return cn_profiles

def correct_readcounts(adata, min_normal_cells=1, inplace=True) -> np.ndarray:
    """
    Correct read counts, normalizing the counts using normal cells as reference
    Requires the `normal` column in adata.obs to be set.
    """
    if 'normal' not in adata.obs:
        raise ValueError("Missing 'normal' column in adata.obs")
    if adata.obs['normal'].sum() < min_normal_cells:
        raise ValueError("Not enough normal cells to normalize read counts")

    normal_cn = 2.
    baseline = adata[adata.obs['normal']].X.mean(axis=0) / normal_cn # counts per copy
    # TODO: implement ploidy correction for WGD cells if any (using e.g. ploidy info from data)

    if 'Acount' in adata.layers and inplace:
        # normalize also phased data
        adata.layers['Acopy'] = adata.layers['Acount'] / baseline
        adata.layers['Bcopy'] = adata.layers['Bcount'] / baseline

    if inplace:
        adata.layers['copy'] = adata.X / baseline

    return adata.layers['copy']

def load_counts_init_anndata(cnasim_data_path: str):
    # manipulate to extract (chr, start, pos) dataframe - vars for anndata
    readcounts_df = pd.read_csv(os.path.join(cnasim_data_path, 'readcounts.tsv'), delimiter='\t', header=0)
    readcounts_df['chr'] = readcounts_df['chrom'].transform(lambda x: x[3:])
    var_df = readcounts_df[readcounts_df['CELL'] == 'cell1'][['chr', 'start', 'end']].reset_index(drop=True)
    var_df['chr'] = pd.Categorical(var_df['chr'], ordered=True)
    var_df.set_index(var_df[['chr', 'start', 'end']].apply(lambda x: f"{x['chr']}:{x['start']}-{x['end']}", axis=1),
                     inplace=True)

    # sum Acount + Bcount = readcount
    is_phased = 'readcount' not in readcounts_df.columns
    if is_phased:
        readcounts_df['readcount'] = readcounts_df['Acount'] + readcounts_df['Bcount']

    # extract readcounts as a n_cells x n_bins matrix
    # to wide format
    wide_counts_df = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='readcount', columns='CELL',
                                    sort=False)
    # extract cell columns
    cell_names = [c for c in wide_counts_df.columns if 'cell' in c]
    counts = wide_counts_df[cell_names].transpose().to_numpy()

    adata = anndata.AnnData(counts)
    adata.obs_names = cell_names
    adata.var_names = var_df.index
    adata.var = var_df
    adata.uns['cnasim-params'] = read_log_params(cnasim_data_path)

    if is_phased:
        # if phasing is available, then make layers for Acount and Bcount
        adata.layers['Acount'] = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='Acount', columns='CELL',
                                                sort=False)[cell_names].transpose().to_numpy()
        adata.layers['Bcount'] = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='Bcount', columns='CELL',
                                                sort=False)[cell_names].transpose().to_numpy()
    return adata

def load_cnasim_output_files(cnasim_data_path: str | Path) -> anndata.AnnData:
    """
    Load CNAsim output files into anndata object. The annData object will contain:
    - X: read counts matrix (n_cells x n_bins)
    - layers['copy']: normalized read counts matrix (n_cells x n_bins)
    - layers['state']: copy number state matrix (n_cells x n_bins)
    - obs: cell metadata (normal: bool, clone: categorical)
    - var: bin metadata (chr, start, end)
    - uns: additional data (cnasim-params, cell-tree-newick, clonal-tree-newick, clone-id-tree-newick)
    Parameters:
    ----------
    cnasim_data_path: str, path to CNAsim output directory
    Returns:
    ----------
    anndata.AnnData object
    """
    # 1. load readcounts.tsv and init anndata
    adata = load_counts_init_anndata(cnasim_data_path)
    n_cells = adata.n_obs
    n_bins = adata.n_vars
    params = adata.uns['cnasim-params']  # load simulation params
    # 2. load clean_profiles.tsv / profiles.tsv
    _ = read_cn_profiles(adata, cnasim_data_path, n_cells=n_cells, inplace=True)
    # 3. load cell assignment cell_type.tsv
    cell_assignment, cell_types_df = read_cell_types(cnasim_data_path)
    assert len(cell_assignment) == n_cells, f"Cell assignment length {len(cell_assignment)} does not match n_cells {n_cells}"

    adata.obs_names = [f'cell{i+1}' for i in range(n_cells)]
    adata.obs['normal'] = cell_assignment == 0  # normal is always 0
    adata.obs['clone'] = pd.Categorical(cell_assignment)
    # 3.1 normalize read counts
    _ = correct_readcounts(adata, inplace=True) # inplace adding layers: ['copy', 'Acopy', 'Bcopy']
    # for each clone, select its founder
    # 4. read and process tree.nwk
    # print(clone_founder)  # ['root', 'ancestor2', 'ancestor19', ...]
    # print(clone_id_map)  # {'normal': 0, 'clone6': 6, 'clone2': 2, ...}
    tree_nx = get_sc_tree(cnasim_data_path, cell_types_df)
    clonal_tree_nx = _collapse_equal_clones(tree_nx, clone_attr='clone')
    remapped_clonal_tree_nx = nx.relabel_nodes(clonal_tree_nx, {n: clonal_tree_nx.nodes[n]['clone'] for n in clonal_tree_nx.nodes()})

    cell_tree_newick = _tree_to_newick(tree_nx, root='root', weight='weight') + ";"
    clonal_tree_newick = _tree_to_newick(clonal_tree_nx, root='root', weight='weight') + ";"
    clone_id_tree_nwk = _tree_to_newick(remapped_clonal_tree_nx, root=0, weight='weight') + ";"

    adata.uns['cell-tree-newick'] = cell_tree_newick
    adata.uns['clonal-tree-newick'] = clonal_tree_newick
    adata.uns['clone-id-tree-newick'] = clone_id_tree_nwk
    return adata

# =======================================
# Tree manipulation (copied from VICTree)
# =======================================
def _collapse_equal_clones(tree: nx.DiGraph, clone_attr: str = 'clone') -> nx.DiGraph:
    """
    Collapse tree nodes to only contain founder clones and normal clone.
    If a node has the same clone as its parent, then the node is removed and its children are connected to the parent.
    """
    def parent_same_clone(x):
        parent = next(tree.predecessors(x))
        return tree.nodes[parent][clone_attr] == tree.nodes[x][clone_attr]
    collapsed_tree = _collapse_tree_if(tree, parent_same_clone)
    return collapsed_tree

def _collapse_tree_if(tree: nx.DiGraph, condition: callable) -> nx.DiGraph:
    """
    Collapse tree nodes to only contain nodes that satisfy the condition on the node.
    If a node satisfies the condition, then the node is removed and its children are connected to the parent.
    """
    collapsed_tree = tree.copy()
    nodes = list(collapsed_tree.nodes())
    for n in nodes:
        for parent in collapsed_tree.predecessors(n):
            if condition(n):
                children = list(collapsed_tree.successors(n))
                for c in children:
                    collapsed_tree.add_edge(parent, c, weight=collapsed_tree[parent][n]['weight'] + collapsed_tree[n][c]['weight'])
                collapsed_tree.remove_node(n)
    return collapsed_tree

def _tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = _tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)

def _parse_newick(tree_file):
    """
    Parameters
    ----------
    tree_file: filepath. if newick string is desired, it's enough to
        pass StringIO(newick_string) instead

    Returns
    -------
    nx.DiGraph tree
    """
    tree = Phylo.read(tree_file, 'newick')
    und_tree_nx = Phylo.to_networkx(tree)
    # Phylo names add unwanted information in unstructured way
    # find node numbers and relabel nx tree
    names_string = list(str(cl.confidence) if cl.name is None else cl.name for cl in und_tree_nx.nodes)
    try:
        names = list(map(int, names_string))
    except ValueError:
        names = names_string
    mapping = dict(zip(und_tree_nx, names))
    relabeled_tree = nx.relabel_nodes(und_tree_nx, mapping)
    tree_nx = nx.DiGraph()
    tree_nx.add_weighted_edges_from(relabeled_tree.edges(data='weight'))
    return tree_nx

# =======================
# Command line interface
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    cli = argparse.ArgumentParser(
        description="Convert CNAsim output files to anndata object."
    )
    cli.add_argument('-i', '--input', type=str, required=True, help="CNAsim files directory (where `readcounts.tsv`, `profiles.tsv`, `cell_types.tsv` and `tree.nwk` are located)")
    cli.add_argument('-o', '--output', type=str, default=None, help="output file path e.g. ./datasets/dat.h5ad")
    args = cli.parse_args()

    convert_cnasim_output_to_anndata(args.input, args.output)
    logger.info(f"Wrote {args.output}")

if __name__ == "__main__":
    main()