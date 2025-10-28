import unittest

import dendropy as dpy
from dendropy.calculate.treecompare import symmetric_difference
import networkx as nx

from cellmates.utils import tree_utils
from cellmates.utils.tree_utils import convert_networkx_to_dendropy, label_tree, convert_dendropy_to_networkx


class TreeUtilsTestCase(unittest.TestCase):
    def test_convert_nx_to_dendropy(self):
        # build a tree on nx
        nxtree = nx.DiGraph()
        # add edges to make tree: ((3,4)2,(5,6)1);
        tnamespace = dpy.TaxonNamespace(['5', '6', '3', '4'], label='taxa')
        nxtree.add_edges_from([(2, 3), (2, 4), (1, 5), (1, 6), (0, 1), (0, 2)])
        nx.write_network_text(nxtree, sources=[0])

        # convert
        dtree_converted = convert_networkx_to_dendropy(nxtree, taxon_namespace=tnamespace)

        # check root label
        self.assertEqual(dtree_converted.seed_node.label, '3_4_5_6')

        # build same tree with dendropy
        dtree = dpy.Tree.get(data="((3,4)2,(5,6)1);", schema='newick', taxon_namespace=tnamespace)
        # must have same internal labels and be rooted
        label_tree(dtree, method='group')
        dtree.is_rooted = True

        # compute rf dist
        rfdist = symmetric_difference(dtree, dtree_converted)
        self.assertAlmostEqual(rfdist, 0)

    def test_convert_dendropy_to_nx(self):
        # build a tree with dendropy
        dtree = dpy.Tree.get(data="((3,4)2,(5,6)1)0;", schema='newick')
        dtree.is_rooted = True

        # convert
        nxtree_converted = convert_dendropy_to_networkx(dtree)

        # build same tree on nx
        nxtree = nx.DiGraph()
        # add edges to make tree: ((3,4)2,(5,6)1))0;
        nxtree.add_edges_from([(2, 3), (2, 4), (1, 5), (1, 6), (0, 1), (0, 2)])
        nx.write_network_text(nxtree, sources=[0])

        # Check edges
        self.assertEqual(set(nxtree.edges()), set(nxtree_converted.edges()))

    def test_random_binary_tree(self):
        n_cells = 5
        length_mean = 0.1

        # generate random binary tree
        tree = tree_utils.random_binary_tree(n_cells, length_mean=length_mean)
        self.assertEqual(len(tree.leaf_nodes()), n_cells)
        self.assertTrue(tree.is_rooted)
        print(tree.as_string(schema='newick'))
        print([edge.label for edge in tree.edges()])
        tree.print_plot()

if __name__ == '__main__':
    unittest.main()
