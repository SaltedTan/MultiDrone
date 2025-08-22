import numpy as np
from typing import Optional

class Node:
    """
    Represents a node in the RRT search tree.
    
    Attributes:
        configuration (np.ndarray): The configuration of the multi-drone system,
                                    represented as an array of shape (N, 3).
        parent (Optional[Node]): The parent node in the tree. The root node
                                 has a parent of None.
    """
    def __init__(self, configuration: np.ndarray, parent: Optional['Node'] = None):
        """Initialises a new Node."""
        self.configuration = configuration
        self.parent = parent