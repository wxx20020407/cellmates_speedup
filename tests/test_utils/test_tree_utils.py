import unittest

import dendropy as dpy
from dendropy.calculate.treecompare import symmetric_difference
import networkx as nx

from utils.tree_utils import convert_networkx_to_dendropy


class MyTestCase(unittest.TestCase):
    def test_convert_nx_to_dendropy(self):
        # build a tree on nx
        nxtree = nx.DiGraph()
        nxtree.add_edges_from([(0, 1), (0, 2), (2, 3), (2, 4)])

        # convert
        dtree_converted = convert_networkx_to_dendropy(nxtree)

        # check root label
        self.assertEqual(dtree_converted.seed_node.label, '0')

        # build same tree with dendropy
        dtree = dpy.Tree.get(data="((3,4),1);", schema='newick', taxon_namespace=dtree_converted.taxon_namespace)
        # compute rf dist
        rfdist = symmetric_difference(dtree, dtree_converted)
        self.assertAlmostEqual(rfdist, 0.)


if __name__ == '__main__':
    unittest.main()
