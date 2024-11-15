import logging
import operator

import networkx as nx
import numpy as np


def reconstruct_tree(distance_matrix):
    return None


def _build_tree_rec(ctr: dict, ntc: dict, ntr: dict, otus: set, edges: set[tuple]) -> set[tuple]:
    if len(otus) == 2:
        for c in otus:
            # add edge with length
            edges.add(('r', c, ntr[c]))
    else:
        vw, l = max(ctr.items(), key=operator.itemgetter(1))
        # remove pair and add common ancestor with averaged distance

        # Computing edge _lengths after merge
        # save node-to-root distance for edge computation later
        # removing the pair from the centroid to rood distances as they are merged
        v, w = vw

        # remove node-to-root for merged nodes
        ntr.pop(v)
        ntr.pop(w)

        # Update distances merging vw in one OTU
        vsw = v + '_' + w  # node with string showing merges v_w
        ntr[vsw] = ctr.pop(vw)  # save centroid to root as the new node-to-root distance (new OTU)
        new_otus = otus.difference({w, v})
        for c in new_otus:
            vc = frozenset({v, c})
            wc = frozenset({w, c})
            # new pairwise distances
            vsw_c = frozenset({vsw, c})
            # update ctr distance for new node
            ctr[vsw_c] = .5 * (ctr[vc] + ctr[wc])
            # update ntc distances for new node
            ntc[vsw, c] = ntr[vsw] - ctr[vsw_c]
            ntc[c, vsw] = .5 * (ntc[c, v] + ntc[c, w])
            # remove already merged nodes
            ctr.pop(vc)
            ctr.pop(wc)
            ntc.pop(c, v)
            ntc.pop(c, w)
            ntc.pop(v, c)
            ntc.pop(w, c)

        # add node/subtree as OTU
        new_otus.add(vsw)

        v_edge_length = ntc.pop((v, w))
        w_edge_length = ntc.pop((w, v))

        edges = _build_tree_rec(ctr, ntc, ntr, new_otus, edges)
        # find edge with merged node and add subtrees
        for x, v_, l in edges:
            if v_ == vsw:
                # add edge with length checking for negative values
                if v_edge_length < 0:
                    v_edge_length = 0
                    logging.warning(f'negative edge length for {v} <- {vsw}')
                if w_edge_length < 0:
                    w_edge_length = 0
                    logging.warning(f'negative edge length for {w} <- {vsw}')
                edges = edges.union([(v_, v, v_edge_length), (v_, w, w_edge_length)])
                break

    return edges


def build_tree(ctr_table: np.ndarray) -> nx.DiGraph:
    # operational taxonomic units, OTUs, init with cells
    otus = set(map(str, range(ctr_table.shape[0])))
    # at each iteration, contains the centroid to root distance for each pair of OTUs
    # OTU is a set of cells (frozenset) which consist of a non-modifiable subtree
    ctr = {}
    # node-to-centroid distances for each OTU (initially single-cells) wrt to each other (index order is important here
    # as opposed to ctr that is symmetric)
    ntc = {}  # dict (str,str) -> float
    # node-to-root distances for each OTU as average of node-to-centroid distances over all other OTUs
    ntr = {str(v): 0 for v in range(len(otus))}  # dict str -> float
    for v in range(len(otus)):
        v_str = str(v)
        for w in range(v + 1, len(otus)):
            w_str = str(w)
            vsw = frozenset({v_str, w_str})
            # init ctr distances
            ctr[vsw] = ctr_table[v, w, 0]

            # compute node to root distance of v wrt w
            ntc[v_str, w_str] = ctr_table[v, w, 1]
            # compute node to root distance of w wrt v
            ntc[w_str, v_str] = ctr_table[v, w, 2]

            # compute node to root distance of v
            ntr[v_str] += ntc[v_str, w_str] + ctr_table[v, w, 1]
            # compute node to root distance of w
            ntr[w_str] += ntc[w_str, v_str] + ctr_table[v, w, 2]

    # normalize node-to-root distances to get the average
    ntr = {str(v): ntr[str(v)] / (len(otus) - 1) for v in range(len(otus))}

    # build tree only using ctr distances
    edges = _build_tree_rec(ctr, ntc, ntr, otus, set())
    em_tree = nx.DiGraph()
    # add edges with _lengths
    em_tree.add_weighted_edges_from(edges, weight='length')
    # add_lengths(em_tree, ctr_table)

    return em_tree
