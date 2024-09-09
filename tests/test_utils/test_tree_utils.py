import unittest

import dendropy as dpy
from dendropy.calculate.treecompare import symmetric_difference
import networkx as nx

from utils.tree_utils import convert_networkx_to_dendropy


class MyTestCase(unittest.TestCase):
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
        self.assertEqual(dtree_converted.seed_node.label, '0')

        # build same tree with dendropy
        dtree = dpy.Tree.get(data="((3,4)2,(5,6)1);", schema='newick', taxon_namespace=tnamespace)
        # compute rf dist
        rfdist = symmetric_difference(dtree, dtree_converted)
        self.assertAlmostEqual(rfdist, 0.)



if __name__ == '__main__':
    unittest.main()
