import os
import numpy as np
import pandas as pd
from scipy import stats
import time
import argparse
import yaml

from multi_drone import MultiDrone
from planner import BiasedRRTConnect

# --- Experiment Configuration ---
NUM_TRIALS = 10
PLANNER_PARAMS = {
    'max_iter': 50000,
    'step_size': 3.0,
    'p_obs': 0.2,
}

def run_single_trial(env_file: str) -> dict:
    """Runs a single motion planning trial and returns the results."""
    # Determine the number of drones from the environment file
    with open(env_file, 'r') as f:
        config = yaml.safe_load(f)
        try:
            num_drones = len(config['initial_configuration'])
        except (KeyError, TypeError):
            print(f"\nError: Could not determine number of drones from {env_file}.")
            return {'success': False, 'planning_time': -1, 'iterations': -1, 'path_length': np.nan, 'total_nodes': -1}

    # Initialise simulation in headless mode with the correct number of drones
    sim = MultiDrone(num_drones=num_drones, environment_file=env_file, headless=True)
    
    planner = BiasedRRTConnect(
        sim=sim,
        max_iter=PLANNER_PARAMS['max_iter'],
        step_size=PLANNER_PARAMS['step_size'],
        p_obs=PLANNER_PARAMS['p_obs']
    )
    
    solution_path, stats = planner.plan()
    
    result = {
        'success': solution_path is not None,
        'planning_time': stats['planning_time'],
        'iterations': stats['iterations'],
        'path_length': stats['path_length'] if solution_path else np.nan,
        'total_nodes': stats['nodes_in_tree_a'] + stats['nodes_in_tree_b'],
    }
    return result

def calculate_ci(data: pd.Series) -> float:
    """Calculates the half-width of the 95% confidence interval."""
    if len(data) < 2:
        return 0.0
    mean, sem = data.mean(), stats.sem(data)
    return sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)

def generate_latex_table(summary_stats: pd.DataFrame, caption: str):
    """Generates a LaTeX formatted table from the summary statistics."""
    header = f"""
\\begin{{table}}[h!]
\\centering
\\caption{{{caption} (N=10 trials)}}
\\label{{tab:performance_scaling}}
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
\\textbf{{Environment}} & \\textbf{{Success}} & \\textbf{{Time (s)}} & \\textbf{{Path Len.}} & \\textbf{{Iters.}} & \\textbf{{Nodes}} \\\\
\\hline"""
    footer = r"""\hline
\end{tabular}
\end{table}"""
    
    body = ""
    for index, row in summary_stats.iterrows():
        env_name = index.replace('_', '\\_')
        sr = f"{row['success_rate']:.0f}\\%"
        
        if row['success_rate'] > 0:
            time_str = f"${row['planning_time_mean']:.2f} \\pm {row['planning_time_ci']:.2f}$"
            len_str = f"${row['path_length_mean']:.1f} \\pm {row['path_length_ci']:.1f}$"
            iter_str = f"${int(row['iterations_mean'])} \\pm {int(row['iterations_ci'])}$"
            nodes_str = f"${int(row['total_nodes_mean'])} \\pm {int(row['total_nodes_ci'])}$"
        else:
            time_str = len_str = iter_str = nodes_str = "N/A"
            
        body += f"{env_name} & {sr} & {time_str} & {len_str} & {iter_str} & {nodes_str} \\\\\n"

    print("\n--- LaTeX Table for Report ---\n")
    print(header + body + footer)
    print("\n" + "-"*30 + "\n")

def main():
    """Main function to run all experiments and report results."""
    parser = argparse.ArgumentParser(description="Run motion planning experiments.")
    parser.add_argument('environments_dir', type=str, help="Directory containing the environment YAML files.")
    args = parser.parse_args()

    if not os.path.exists(args.environments_dir):
        print(f"Error: Directory '{args.environments_dir}' not found.")
        return

    env_files = sorted([f for f in os.listdir(args.environments_dir) if f.endswith('.yaml')])
    all_results = {}

    for env_file in env_files:
        env_path = os.path.join(args.environments_dir, env_file)
        print(f"\n===== Running Trials for: {env_file} =====")
        
        trial_results = []
        for i in range(NUM_TRIALS):
            print(f"  > Trial {i + 1}/{NUM_TRIALS}...", end='', flush=True)
            start_time = time.time()
            result = run_single_trial(env_path)
            end_time = time.time()
            print(f" {'Success' if result['success'] else 'Failure'} ({end_time - start_time:.2f}s)")
            trial_results.append(result)
            
        all_results[env_file.replace('.yaml', '')] = pd.DataFrame(trial_results)

    summary_data = []
    print("\n\n" + "="*40)
    print("        EXPERIMENT SUMMARY")
    print("="*40)

    for env_name, results_df in all_results.items():
        successful_trials = results_df[results_df['success'] == True]
        success_rate = (len(successful_trials) / len(results_df)) * 100
        stats_summary = {'environment': env_name, 'success_rate': success_rate}
        
        if not successful_trials.empty:
            metrics = ['planning_time', 'path_length', 'iterations', 'total_nodes']
            for metric in metrics:
                stats_summary[f'{metric}_mean'] = successful_trials[metric].mean()
                stats_summary[f'{metric}_ci'] = calculate_ci(successful_trials[metric])
        
        summary_data.append(stats_summary)
        
        print(f"\n--- Environment: {env_name} ---")
        print(f"  Success Rate: {success_rate:.1f}% ({len(successful_trials)}/{len(results_df)})")
        if not successful_trials.empty:
            m, ci = stats_summary['planning_time_mean'], stats_summary['planning_time_ci']
            print(f"  Avg. Planning Time: {m:.3f} +/- {ci:.3f} s")
            m, ci = stats_summary['path_length_mean'], stats_summary['path_length_ci']
            print(f"  Avg. Path Length:   {m:.1f} +/- {ci:.1f} waypoints")
            m, ci = stats_summary['iterations_mean'], stats_summary['iterations_ci']
            print(f"  Avg. Iterations:    {m:.0f} +/- {ci:.0f}")
            m, ci = stats_summary['total_nodes_mean'], stats_summary['total_nodes_ci']
            print(f"  Avg. Total Nodes:   {m:.0f} +/- {ci:.0f}")
        else:
            print("  No successful trials to report statistics.")

    summary_df = pd.DataFrame(summary_data).set_index('environment')
    table_caption = f"Planner Performance vs. {'Complexity' if 'q4' in args.environments_dir else 'Number of Drones'}"
    generate_latex_table(summary_df, table_caption)

if __name__ == '__main__':
    main()