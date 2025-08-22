import numpy as np
from multi_drone import MultiDrone
from planner import BiasedRRTConnect

def main():
    """
    Main function to run the multi-drone motion planner.
    """
    print("Initialising Multi-Drone environment...")
    # Initialise the simulation environment for K drones
    sim = MultiDrone(num_drones=2, environment_file="environment.yaml")
    
    # --- Planner Parameters ---
    MAX_ITERATIONS = 5000   # Maximum number of iterations
    STEP_SIZE = 2.0         # Step size for tree extension (epsilon)
    P_OBS = 0.2             # Probability of obstacle-biased sampling (20%)
    
    print("Setting up Biased RRT-Connect planner...")
    planner = BiasedRRTConnect(
        sim=sim,
        max_iter=MAX_ITERATIONS,
        step_size=STEP_SIZE,
        p_obs=P_OBS
    )
    
    print("Starting motion planning...")
    solution_path, stats = planner.plan()
    
    # --- Print Statistics ---
    print("\n--- Planning Statistics ---")
    print(f"Planning Time: {stats['planning_time']:.4f} seconds")
    print(f"Iterations: {stats['iterations']}/{MAX_ITERATIONS}")
    print(f"Path Length: {stats['path_length']} waypoints")
    print(f"Nodes in Tree A: {stats['nodes_in_tree_a']}")
    print(f"Nodes in Tree B: {stats['nodes_in_tree_b']}")
    print("---------------------------\n")

    if solution_path:
        print("Visualising the found path... Close the window to exit.")
        sim.visualize_paths(solution_path)
    else:
        print("No solution path to visualise.")

if __name__ == "__main__":
    main()