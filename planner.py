import numpy as np
import time
from typing import Optional, List, Tuple

from multi_drone import MultiDrone
from node import Node
from tree import Tree

class BiasedRRTConnect:
    """
    Implements the Biased RRT-Connect motion planner for multiple drones.
    """
    def __init__(self, sim: MultiDrone, max_iter: int, step_size: float, p_obs: float):
        """
        Initialises the planner with simulation and algorithm parameters.

        Args:
            sim (MultiDrone): The simulation environment.
            max_iter (int): The maximum number of iterations to run the planner.
            step_size (float): The step size (epsilon) for extending the trees.
            p_obs (float): The probability of using obstacle-biased sampling.
        """
        self.sim = sim
        self.max_iter = max_iter
        self.step_size = step_size
        self.p_obs = p_obs

    def _sample(self) -> np.ndarray:
        """Generates a random configuration using a hybrid sampling strategy."""
        if np.random.rand() < self.p_obs:
            # Obstacle-biased sampling
            return self._sample_near_obstacle()
        else:
            # Uniform random sampling
            return self._sample_uniform()

    def _sample_uniform(self) -> np.ndarray:
        """Generates a valid configuration by uniform random sampling."""
        while True:
            # Generate a random point for each drone within the environment bounds
            low = self.sim._bounds[:, 0]
            high = self.sim._bounds[:, 1]
            q_sample = np.random.uniform(low, high, size=(self.sim.N, 3))
            if self.sim.is_valid(q_sample):
                return q_sample

    def _sample_near_obstacle(self) -> np.ndarray:
        """
        Generates a sample near an obstacle boundary by interpolating
        between a known invalid and a known valid configuration.
        """
        q_free = self._sample_uniform() # Guaranteed to be valid

        # Find a random invalid configuration
        while True:
            low = self.sim._bounds[:, 0]
            high = self.sim._bounds[:, 1]
            q_obs = np.random.uniform(low, high, size=(self.sim.N, 3))
            if not self.sim.is_valid(q_obs):
                break
        
        alpha = np.random.rand()
        q_biased = (1 - alpha) * q_obs + alpha * q_free
        return q_biased

    @staticmethod
    def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
        """Takes a step of a fixed size from q_from towards q_to."""
        direction_vector = q_to - q_from
        dist = np.linalg.norm(direction_vector)
        
        if dist < step_size:
            return q_to
        else:
            # Move along the normalised direction vector
            unit_vector = direction_vector / dist
            return q_from + unit_vector * step_size

    def _extend(self, tree: Tree, q_rand: np.ndarray) -> Optional[Node]:
        """Extends the tree towards a random configuration."""
        q_near_node = tree.get_nearest_neighbor(q_rand)
        q_near = q_near_node.configuration
        
        q_new = self._steer(q_near, q_rand, self.step_size)
        
        if self.sim.motion_valid(q_near, q_new):
            new_node = Node(q_new, parent=q_near_node)
            tree.add_node(new_node)
            return new_node
            
        return None

    def plan(self) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Executes the main RRT-Connect planning algorithm.
        
        Returns:
            A tuple containing:
            - The solution path as a list of configurations, or None if not found.
            - A dictionary with planning statistics.
        """
        start_time = time.time()
        
        q_start = self.sim.initial_configuration
        q_goal = self.sim.goal_positions
        
        tree_a = Tree(Node(q_start))
        tree_b = Tree(Node(q_goal))
        
        for k in range(self.max_iter):
            q_rand = self._sample()
            
            # Extend Tree A and try to connect to Tree B
            q_new_node = self._extend(tree_a, q_rand)
            
            if q_new_node:
                q_new = q_new_node.configuration
                q_connect_node = tree_b.get_nearest_neighbor(q_new)
                q_connect = q_connect_node.configuration

                if self.sim.motion_valid(q_new, q_connect):
                    # Solution found!
                    path_a = Tree.reconstruct_path(q_new_node)
                    path_b = Tree.reconstruct_path(q_connect_node)
                    
                    # Path from B is goal->connect, so reverse it
                    solution_path = path_a + path_b[::-1]
                    
                    stats = {
                        "planning_time": time.time() - start_time,
                        "iterations": k + 1,
                        "path_length": len(solution_path),
                        "nodes_in_tree_a": len(tree_a.nodes),
                        "nodes_in_tree_b": len(tree_b.nodes)
                    }
                    print("Solution found!")
                    return solution_path, stats

            # Swap trees for the next iteration to grow the other one
            tree_a, tree_b = tree_b, tree_a

        # If loop finishes, no solution was found
        stats = {
            "planning_time": time.time() - start_time,
            "iterations": self.max_iter,
            "path_length": 0,
            "nodes_in_tree_a": len(tree_a.nodes),
            "nodes_in_tree_b": len(tree_b.nodes)
        }
        print("Failed to find a solution within the iteration limit.")
        return None, stats