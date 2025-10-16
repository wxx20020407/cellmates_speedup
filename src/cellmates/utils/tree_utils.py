import random
from io import StringIO

import dendropy
import numpy as np
import skbio
from dendropy import Tree
import networkx as nx
import scipy.stats as ss
from Bio import Phylo


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
                                 taxon_namespace=None, edge_length=None) -> dendropy.Tree:
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
    dendropy_tree = Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)
    label_tree(dendropy_tree, method='group')
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
        dendropy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    tns = dendropy.TaxonNamespace([dendropy.Taxon(str(i)) for i in range(n)], label='taxa')
    tree = dendropy.treesim.treesim.pure_kingman_tree(taxon_namespace=tns)
    tree.is_rooted = True
    label_tree(tree)
    # traverse the tree and assign _lengths
    for edge in tree.preorder_edge_iter():
        # scale = 1 / lambda
        edge.length = ss.expon(scale=length_mean).rvs()
    return tree


def get_node2node_distance(tree: dendropy.Tree, node1_label: str, node2_label: str):
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

def newick_to_nx(nwk_str, edge_attr='weight', interior_node_names=None) -> nx.DiGraph:
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
    if interior_node_names is not None:
        for cl in und_tree_nx.nodes:
            if cl.name is None:
                cl.name = interior_node_names.pop(0)
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
            mapping[v] = cell_names[int(v) - 1] # leaf node
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

if __name__ == '__main__':
    # try tree conversion
    nwk = "((0:0.1,1:0.2):0.3,(2:0.4,3:0.5):0.6);"
    print(nwk)
    dpy_tree: Tree = Tree.get(data=nwk, schema='newick')
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


def get_lowest_common_ancestor(tree_nx, node1, node2):
    """
    Get the index of the least common ancestor of two nodes in a directed tree.
    """
    return nx.lowest_common_ancestor(tree_nx, node1, node2)


def convert_skbio_to_networkx(tree_nj_skbio: skbio.TreeNode, interior_node_names=None)-> nx.DiGraph:
    """
    Convert a skbio TreeNode to a networkx DiGraph.
    Parameters
    ----------
    tree_nj_skbio

    Returns tree_nx: nx.DiGraph
    -------

    """
    newick = str(tree_nj_skbio)
    tree_nx = newick_to_nx(newick, edge_attr='weight', interior_node_names=interior_node_names)
    return tree_nx