import dendropy
from dendropy import Tree
import networkx as nx

def tree_to_newick(g: nx.DiGraph, root=None, weight=None, is_internal_call=False):
    """
    Copied from VICTree project
    """
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
            node_str = tree_to_newick(g, root=child, weight=weight, is_internal_call=True)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    newick = "(" + ','.join(subgs) + ")" + str(root)
    if not is_internal_call:
        newick += ';'
    return newick


def _copy_subtree(nxtree: nx.DiGraph, nxroot, droot: dendropy.Node):
    children = [s for s in nxtree.successors(nxroot)]
    if len(children) > 0:
        for nxchild in children:
            dchild = dendropy.Node()
            dchild.label = nxchild
            _copy_subtree(nxtree, nxchild, dchild)
            droot.add_child(dchild)
    else:
        droot.taxon = dendropy.Taxon(droot)


# FIXME: not working because of different taxon namespace.
#  Use `convert_networkx_to_dendropy` instead. Fix might be required later on for
#  deeper copies of trees (e.g. preserving labels/metadata)
def _convert_nx_tree_to_dendropy_tree(nx_tree, nxroot='r'):

    dtree = dendropy.Tree()
    droot = dtree.seed_node
    # set root label
    droot.label = nxroot
    # create taxa
    tns = dendropy.TaxonNamespace([n for n, d in nx_tree.out_degree if d == 0], label='taxa')
    _copy_subtree(nx_tree, nxroot, droot)

    return dtree


def convert_networkx_to_dendropy(nx_tree, taxon_namespace=None) -> dendropy.Tree:
    """
    Converts a NetworkX tree to a DendroPy tree through newick string.

    Args:
      nx_tree: The NetworkX tree to convert.

    Returns:
      A DendroPy tree.
    """
    newick = tree_to_newick(nx_tree)
    dendropy_tree = Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)

    return dendropy_tree
