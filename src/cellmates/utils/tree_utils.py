import random
from io import StringIO
from itertools import combinations

import dendropy as dpy
from dendropy.calculate import treecompare
import numpy as np
import networkx as nx
import scipy.stats as ss
from Bio import Phylo
from sklearn.metrics import f1_score
from collections import defaultdict

from cellmates.utils import math_utils


def nxtree_to_newick(g: nx.DiGraph, root=None, weight=None, is_internal_call=False):
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
            node_str = nxtree_to_newick(g, root=child, weight=weight, is_internal_call=True)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    newick = "(" + ','.join(subgs) + ")" + str(root)
    if not is_internal_call:
        newick += ';'
    return newick


def convert_networkx_to_dendropy(nx_tree, labels_mapping: dict = None,
                                 taxon_namespace=None, edge_length=None, internal_nodes_label='group') -> dpy.Tree:
    """
    Converts a NetworkX tree to a DendroPy tree through newick string.

    Args:
      nx_tree: The NetworkX tree to convert.
      labels_mapping: dict, mapping of taxa labels to new labels

    Returns:
      A DendroPy tree.
    """
    if labels_mapping is not None:
        nx_tree = nx.relabel_nodes(nx_tree, labels_mapping, copy=True)
    newick = nxtree_to_newick(nx_tree, weight=edge_length)
    dendropy_tree = dpy.Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)
    label_tree(dendropy_tree, method=internal_nodes_label)
    dendropy_tree.is_rooted = True

    return dendropy_tree

def convert_dendropy_to_networkx(dendropy_tree: dendropy.Tree, edge_attr='weight') -> nx.DiGraph:
    """
    Converts a DendroPy tree to a NetworkX tree through newick string.

    Args:
      dendropy_tree: The DendroPy tree to convert.

    Returns:
      A NetworkX tree.
    """
    newick = dendropy_tree.as_string(schema='newick')
    nx_tree = newick_to_nx(newick, edge_attr=edge_attr)
    return nx_tree


def random_binary_tree(n: int, length_mean: float, seed=None)-> dendropy.Tree:
    """
    Generate a random binary tree with n leaves using Dendropy.
    ref: https://dendropy.org/primer/treesims.html
    Args:
        n: Number of leaves.
        seed: Random seed.

    Returns:
        A Dendropy tree.
    """
    # set dendropy and scipy seed for reproducibility
    if seed is not None:
        dpy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    tns = dpy.TaxonNamespace([dpy.Taxon(str(i)) for i in range(n)], label='taxa')
    tree = dpy.treesim.treesim.pure_kingman_tree(taxon_namespace=tns)
    tree.is_rooted = True
    label_tree(tree)
    # traverse the tree and assign _lengths
    for edge in tree.preorder_edge_iter():
        # scale = 1 / lambda
        edge.length = ss.expon(scale=length_mean).rvs()
    return tree


def get_node2node_distance(tree: dpy.Tree, node1_label: str, node2_label: str):
    tree.calc_node_root_distances()
    node1 = tree.find_node_with_label(node1_label)
    node2 = tree.find_node_with_label(node2_label)
    if node1.root_distance < node2.root_distance:
        node1, node2 = node2, node1
    return node1.root_distance - node2.root_distance


def label_tree(tree, method='int'):
    """
    Assigns int labels to tree nodes. Leaves are assigned with cell ids, internal nodes with decremental numbers
    different from cell ids.
    """
    if method == 'int':
        rev_node_idx = len(tree.nodes()) - 1
        for n in tree.nodes():
            if n.is_leaf():
                n.label = n.taxon.label
            else:
                n.label = str(rev_node_idx)
                rev_node_idx -= 1
    elif method == 'group':
        for n in tree.postorder_node_iter():
            if n.is_leaf():
                n.label = str(n.taxon.label)
            else:
                # group the taxa in the subtree in a sorted string
                taxa = []
                for c in n.child_node_iter():
                    for t in c.label.split('_'):
                        taxa.append(t)
                n.label = '_'.join(sorted(taxa, key=lambda x: int(x)))
    else:
        raise ValueError(f"Unknown method {method}")

def newick_to_nx(nwk_str, edge_attr='weight'):
    """
    Parameters
    ----------
    nwk_str: str, newick string
    edge_attr: str, edge attribute in which to store the weights

    Returns
    -------
    nx.DiGraph tree with nodes and weighted edges
    """
    tree = Phylo.read(StringIO(nwk_str), 'newick')
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
    tree_nx.add_weighted_edges_from(relabeled_tree.edges(data='weight'), weight=edge_attr)
    return tree_nx

def write_newick(nx_tree, cell_names, out_path=None, edge_attr='weight') -> str:
    # relabel leaf nodes with cell ids from adata
    # add ancestor nodes names with breadth-first search
    assert nx.is_arborescence(nx_tree)
    root_node = list(filter(lambda p: p[1] == 0, nx_tree.in_degree()))[0][0]
    mapping = {root_node: "root"}
    count = 1
    for u, v in nx.bfs_edges(nx_tree, source=root_node):
        if nx_tree.out_degree(v) == 0:
            mapping[v] = cell_names[int(v)] # leaf node
        else:
            mapping[v] = "ancestor" + str(count) # internal node
            count += 1

    nx.relabel_nodes(nx_tree, mapping, copy=False)
    # save and plot inferred tree
    nwk_str = nxtree_to_newick(nx_tree, weight=edge_attr)
    if out_path is not None:
        with open(out_path, 'w') as f:
            f.write(nwk_str)
    return nwk_str

def make_gt_tree_dist(ad, n_states, cell_names: list) -> tuple[dpy.Tree, np.ndarray]:
    # traverse the tree, write lengths to branches and, for each pair, sum lengths between them
    n_sites = ad.n_vars
    nxtree = newick_to_nx(ad.uns['cell-tree-newick'])
    nxtree = nx.dfs_tree(nxtree, source='founder')  # make sure it's rooted
    # get copy number (ancestors) at each node and compute the length based on changes
    ancestor_idx = {a: i for i,a in enumerate(ad.uns['ancestral-names'])}  # names as in the tree (index for ancestral-cn)
    ancestor_cn = ad.uns['ancestral-cn'] # shape (n_ancestors, n_bins)
    for u, v in nxtree.edges():
        if not v.startswith('cell'):
            i,j = ancestor_idx[u], ancestor_idx[v]
            p = math_utils.compute_cn_changes(np.vstack([ancestor_cn[i], ancestor_cn[j]]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        else:
            v_cn = ad[v].layers['state']
            u_cn = ancestor_cn[ancestor_idx[u]]
            p = math_utils.compute_cn_changes(np.vstack([u_cn, v_cn]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        # print(f"Edge {u}->{v} p {p}, length {nxtree[u][v]['length']}")
    # relabel nodes to integers
    nxtree = relabel_name_to_int(nxtree, cell_names)
    dpy_tree = convert_networkx_to_dendropy(nxtree, edge_length='length', internal_nodes_label='int')
    # print("DPY tree with lengths:", dpy_tree.as_string(schema='newick'))
    dist_matrix = get_ctr_table(dpy_tree)
    return dpy_tree, dist_matrix


def relabel_name_to_int(nxtree: nx.DiGraph, cell_names: list) -> nx.DiGraph:
    """
    Give integer labels to nodes in the tree. Cell names (leaves) are labeled from 0 to n-1 in the order of cell_names
    and ancestors are labeled from n to n+m-1 where m is the number of ancestors.
    """
    cells_mapping = {name: i for i, name in enumerate(cell_names)}
    ancestors_mapping = {n: i + len(cell_names) for i, n in enumerate(nxtree.nodes()) if n not in cells_mapping}
    full_mapping = {**cells_mapping, **ancestors_mapping}
    # check that all cell names are in the tree and they are leaves
    for name in cell_names:
        assert name in nxtree.nodes(), f"Cell name {name} not in tree nodes"
        assert nxtree.out_degree(name) == 0, f"Cell name {name} is not a leaf node"
    return nx.relabel_nodes(nxtree, full_mapping, copy=True)


def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table(tree: dpy.Tree) -> np.ndarray:
    """
    Get the centroid table for a given tree.
    The centroid table is a 3D numpy array of shape (n_cells, n_cells, 3) where n_cells is the number of leaves in the tree.
    For each pair of cells (r, s) with r < s, the entry ctr_table[r, s] is a vector of 3 values:
        - ctr_table[r, s, 0]: distance from the centroid of r and s to the root
        - ctr_table[r, s, 1]: distance from the centroid of r and s to r
        - ctr_table[r, s, 2]: distance from the centroid of r and s to s
    The entries for r >= s are set to -1.
    The tree must be rooted and all leaves must be labeled with unique integers from 0 to n_cells - 1.
    Parameters
    ----------
    tree: dpy.Tree, the input tree with edge _lengths

    Returns
    -------
    ctr_table: np.ndarray, the centroid table
    """
    n_cells = len(tree.leaf_nodes())
    assert n_cells == len(tree.taxon_namespace)
    ctr_table = - np.ones((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        centroid = tree.mrca(taxon_labels=[str(r), str(s)])
        ctr_table[r, s, 0] = get_root_distance(centroid)
        ctr_table[r, s, 1] = get_node2node_distance(tree, centroid.label, str(r))
        ctr_table[r, s, 2] = get_node2node_distance(tree, centroid.label, str(s))

    return ctr_table

def f1_score_clades(tree: dpy.Tree, clone_assignment: list) -> float:
    """
    Metrics defined in DICE: for each clone, find the clade that maximizes the F1 score
    Return the average F1 score across all clones.
    """
    # get all clades in the tree
    clades = []
    for n in tree.postorder_node_iter():
        if not n.is_leaf():
            clade = set()
            for leaf in n.leaf_iter():
                clade.add(int(leaf.label))
            clades.append(clade)

    # group cells by clone assignment
    clone_to_cells = defaultdict(set)
    for cell, clone in enumerate(clone_assignment):
        clone_to_cells[clone].add(cell)

    f1_scores = []
    for clone, cells in clone_to_cells.items():
        best_f1 = 0
        for clade in clades:
            y_true = [1 if i in cells else 0 for i in range(len(clone_assignment))]
            y_pred = [1 if i in clade else 0 for i in range(len(clone_assignment))]
            score = f1_score(y_true, y_pred)
            if score > best_f1:
                best_f1 = score
        f1_scores.append(best_f1)

    return np.mean(f1_scores).item()

def normalized_rf_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> float:
    """
    Compute the normalized Robinson-Foulds distance between two (rooted) trees using DendroPy.
    The trees must have the same set of leaf labels.
    The normalized RF distance is the RF distance divided by the maximum possible RF distance.
    """
    rf = treecompare.symmetric_difference(tree1, tree2)
    n_leaves = len(tree1.leaf_nodes())
    max_rf = 2 * (n_leaves - 2)
    return rf / max_rf

if __name__ == '__main__':
    # try tree conversion
    nwk = "((0:0.1,1:0.2):0.3,(2:0.4,3:0.5):0.6);"
    print(nwk)
    dpy_tree: dpy.Tree = dpy.Tree.get(data=nwk, schema='newick')
    dpy_tree.is_rooted = True
    label_tree(dpy_tree)
    full_nwk = dpy_tree.as_string(schema='newick')
    print(full_nwk)

    # read networkx
    # FIXME: nodes are none
    nx_tree = newick_to_nx(full_nwk)
    print(nx_tree.edges(data=True))
    nx_tree_newick = nxtree_to_newick(nx_tree, weight='weight')
    print(nx_tree_newick)

    # convert to Phylo tree
    phylo_tree = Phylo.read(StringIO(nx_tree_newick), 'newick', rooted=True)

    # write newick string
    print(phylo_tree.format('newick'))
