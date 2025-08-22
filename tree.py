import numpy as np
from typing import List
from node import Node

class Tree:
    """
    Represents one of the trees in the bidirectional RRT search.
    
    This class manages a collection of nodes and provides methods to add nodes
    and find the nearest node to a given configuration.
    """
    def __init__(self, root_node: Node):
        """Initialises the tree with a root node."""
        self.nodes: List[Node] = [root_node]

    def add_node(self, node: Node):
        """Adds a new node to the tree."""
        self.nodes.append(node)

    def get_nearest_neighbor(self, q_sample: np.ndarray) -> Node:
        """Finds the node in the tree closest to a given configuration."""
        # Flatten configurations for Euclidean distance calculation in 3N-dimensional space
        configurations = np.array([node.configuration.flatten() for node in self.nodes])
        sample_flat = q_sample.flatten()
        
        # Calculate squared Euclidean distances and find the index of the minimum
        distances_sq = np.sum((configurations - sample_flat)**2, axis=1)
        nearest_index = np.argmin(distances_sq)
        
        return self.nodes[nearest_index]

    @staticmethod
    def reconstruct_path(leaf_node: Node) -> List[np.ndarray]:
        """Reconstructs the path from the root of the tree to a given leaf node."""
        path = []
        current = leaf_node
        while current is not None:
            path.append(current.configuration)
            current = current.parent
        # The path is reconstructed from leaf to root, so we reverse it
        return path[::-1]