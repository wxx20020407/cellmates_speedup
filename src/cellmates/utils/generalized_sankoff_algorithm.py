import networkx as nx
import numpy as np
from collections import defaultdict
import itertools
import time


# --- Helper Functions for Mutations ---

def get_transition_cost(cnp_from, cnp_to, max_k):
    """
    Greedily calculates the minimum number of block mutations to transform
    cnp_from to cnp_to.
    Returns the cost (integer) or float('inf') if it exceeds max_k.
    """
    if np.array_equal(cnp_from, cnp_to):
        return 0

    diff = cnp_to - cnp_from
    cost = 0

    temp_diff = np.copy(diff)

    while np.any(temp_diff) and cost <= max_k:
        cost += 1
        first_diff_idx = np.nonzero(temp_diff)[0][0]
        change_val = temp_diff[first_diff_idx]

        end_idx = first_diff_idx
        while (end_idx + 1 < len(temp_diff) and
               temp_diff[end_idx + 1] == change_val):
            end_idx += 1

        temp_diff[first_diff_idx: end_idx + 1] = 0

    return cost if not np.any(temp_diff) else float('inf')


def apply_mutation(cnp, mutation):
    """Applies a single mutation event to a CNP."""
    start, length, change = mutation
    if length == 0:
        return cnp
    new_cnp = np.copy(cnp)
    new_cnp[start: start + length] += change
    return new_cnp


# --- New Helper for Relabeling and Data Preparation ---

def relabel_tree_and_create_leaf_dict(tree, leaf_cnps_array, leaf_order):
    """
    Relabels a tree to integer labels and prepares the leaf CNP dictionary.

    - Leaf nodes are labeled 0 to N-1 based on leaf_order.
    - Internal nodes are labeled N onwards in a consistent order.

    Args:
        tree (nx.DiGraph): The original tree with arbitrary node names.
        leaf_cnps_array (np.ndarray): NxM array of leaf CNPs.
        leaf_order (list): A list of original leaf node names, where the index
                           corresponds to the row in leaf_cnps_array.

    Returns:
        (nx.DiGraph, dict, dict): A tuple containing:
            - The new relabeled tree.
            - The leaf_cnps dictionary for the algorithm.
            - The full mapping from old labels to new integer labels.
    """
    original_leaves = {node for node in tree.nodes() if tree.out_degree(node) == 0}
    original_internals = {node for node in tree.nodes() if tree.out_degree(node) > 0}

    if set(leaf_order) != original_leaves:
        raise ValueError("leaf_order must contain all and only the leaf nodes from the tree.")

    N = len(original_leaves)
    if leaf_cnps_array.shape[0] != N:
        raise ValueError("Number of rows in leaf_cnps_array must match the number of leaves.")

    # Sort original internals for a consistent assignment order
    sorted_internals = sorted(list(original_internals), key=str)

    # Create mapping from original names to new integer IDs
    leaf_map = {name: i for i, name in enumerate(leaf_order)}
    internal_map = {name: i + N for i, name in enumerate(sorted_internals)}

    mapping = {**leaf_map, **internal_map}

    relabeled_tree = nx.relabel_nodes(tree, mapping, copy=True)

    # Create the new leaf_cnps dictionary with integer keys for the algorithm
    new_leaf_cnps = {i: leaf_cnps_array[i] for i in range(N)}

    return relabeled_tree, new_leaf_cnps, mapping


# --- Main Reconstruction Algorithm ---

def reconstruct_cnps_with_block_mutations(tree, leaf_cnps, max_mutations_per_edge=1):
    """
    Reconstructs the CNPs of internal nodes in a directed binary tree using a
    generalized Sankoff algorithm for block mutations.

    Args:
        tree (nx.DiGraph): A directed binary tree with integer node labels.
        leaf_cnps (dict): A dictionary mapping integer leaf node names to their CNP arrays.
        max_mutations_per_edge (int): The maximum number of block mutations on an edge.

    Returns:
        dict: A dictionary mapping all integer node names to their reconstructed CNP arrays.
    """
    root = [n for n, d in tree.in_degree() if d == 0][0]

    for node in nx.dfs_postorder_nodes(tree, source=root):
        if tree.out_degree(node) == 0:
            cnp = leaf_cnps[node]
            tree.nodes[node]['costs'] = {tuple(cnp): 0}
            continue

        children = list(tree.successors(node))
        c1, c2 = children[0], children[1]

        costs1 = tree.nodes[c1]['costs']
        costs2 = tree.nodes[c2]['costs']

        node_costs = defaultdict(lambda: float('inf'))
        node_traceback = {}

        for s1, cost1 in costs1.items():
            s1_arr = np.array(s1)
            for s2, cost2 in costs2.items():
                s2_arr = np.array(s2)

                # Test parent state = s1
                s_p1_arr = s1_arr
                cost_trans_1_2 = get_transition_cost(s_p1_arr, s2_arr, max_mutations_per_edge)
                total_cost1 = cost1 + cost2 + cost_trans_1_2
                s_p1_tuple = tuple(s_p1_arr)
                if total_cost1 < node_costs[s_p1_tuple]:
                    node_costs[s_p1_tuple] = total_cost1
                    node_traceback[s_p1_tuple] = {c1: s1, c2: s2}

                # Test parent state = s2
                s_p2_arr = s2_arr
                cost_trans_2_1 = get_transition_cost(s_p2_arr, s1_arr, max_mutations_per_edge)
                total_cost2 = cost1 + cost2 + cost_trans_2_1
                s_p2_tuple = tuple(s_p2_arr)
                if total_cost2 < node_costs[s_p2_tuple]:
                    node_costs[s_p2_tuple] = total_cost2
                    node_traceback[s_p2_tuple] = {c1: s1, c2: s2}

        tree.nodes[node]['costs'] = dict(node_costs)
        tree.nodes[node]['traceback'] = node_traceback

    reconstructed_cnps = {}
    root_costs = tree.nodes[root]['costs']
    best_root_cnp_tuple = min(root_costs, key=root_costs.get)
    reconstructed_cnps[root] = np.array(best_root_cnp_tuple)

    for node in nx.dfs_preorder_nodes(tree, source=root):
        if tree.out_degree(node) == 0:
            reconstructed_cnps[node] = leaf_cnps[node]
            continue

        parent_cnp_tuple = tuple(reconstructed_cnps[node])
        children = list(tree.successors(node))
        c1, c2 = children[0], children[1]

        best_child_states = tree.nodes[node]['traceback'][parent_cnp_tuple]

        reconstructed_cnps[c1] = np.array(best_child_states[c1])
        reconstructed_cnps[c2] = np.array(best_child_states[c2])

    return reconstructed_cnps


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define the Tree Structure with arbitrary (e.g., string) names
    G_orig = nx.DiGraph()
    G_orig.add_edges_from([
        ('root', 'A'), ('root', 'B'),
        ('A', 'C'), ('A', 'leaf1'),
        ('B', 'leaf2'), ('B', 'leaf3'),
        ('C', 'leaf4'), ('C', 'leaf5')
    ])

    # 2. Define Known Leaf CNPs as a single NxM numpy array
    # N=5 leaves, M=10 sites
    # The order of rows must match the 'leaf_order' list below.
    leaf_order = ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5']
    leaf_cnps_array = np.array([
        [2, 2, 2, 2, 2, 5, 5, 5, 5, 5],  # Corresponds to leaf1
        [4, 4, 4, 4, 4, 5, 5, 5, 5, 5],  # Corresponds to leaf2
        [4, 4, 4, 4, 4, 7, 7, 5, 5, 5],  # Corresponds to leaf3
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Corresponds to leaf4
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Corresponds to leaf5
    ])

    print("--- Input Data ---")
    print(f"Original tree edges: {G_orig.edges()}")
    print(f"Leaf order for numpy array: {leaf_order}")
    print("-" * 20)

    # 3. Relabel the tree and prepare data for the algorithm
    G_relabeled, leaf_cnps_dict, mapping = relabel_tree_and_create_leaf_dict(
        G_orig, leaf_cnps_array, leaf_order
    )
    print("\n--- Relabeling ---")
    print("Mapping from original names to new integer labels:")
    print(mapping)
    print(f"New tree edges: {G_relabeled.edges()}")
    print("-" * 20)

    # 4. Run the Reconstruction
    print("\nRunning reconstruction with max_mutations_per_edge = 1...")
    start_time = time.time()
    reconstructed = reconstruct_cnps_with_block_mutations(G_relabeled, leaf_cnps_dict, max_mutations_per_edge=1)
    end_time = time.time()

    # 5. Print the Results using new integer labels
    print("\n--- Reconstruction Results (with integer labels) ---")
    total_cost = 0
    root = [n for n, d in G_relabeled.in_degree() if d == 0][0]

    # Sort nodes for clear, ordered output
    sorted_nodes = sorted(reconstructed.keys())

    for node in sorted_nodes:
        cnp = reconstructed[node]
        print(f"Node {node}: {cnp}")
        if G_relabeled.out_degree(node) > 0:
            for child in sorted(G_relabeled.successors(node)):
                child_cnp = reconstructed[child]
                cost = get_transition_cost(cnp, child_cnp, max_k=10)
                total_cost += cost
                print(f"  Edge {node}->{child} | Mutations: {cost}")

    print("-" * 20)
    print(f"\nTotal Parsimony Score (number of mutations): {total_cost}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

