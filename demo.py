import numpy as np
import networkx as nx
import random
from env.outbreak_env import OutbreakEnv
from dataset_utils import load_snap_livejournal
import json # ADDED JSON TRACKING

def run_baseline(env, strategy="random"):
    obs, info = env.reset()
    done = False
    
    tested_nodes = set()
    num_nodes = env.num_nodes
    
    log_data = {"steps": []}
    
    positive_nodes = set()  # Track confirmed positive nodes for contact-tracing

    while not done:
        untested = [i for i in range(num_nodes) if i not in tested_nodes]
        if strategy == "random":
            action = random.choice(untested)
        elif strategy == "degree":
            action = max(untested, key=lambda x: env.degrees[x])
        elif strategy == "contact_tracing":
            # Prioritize untested neighbors of known positives
            priority = []
            for pos_node in positive_nodes:
                for neighbor in env.adj_list[pos_node]:
                    if neighbor not in tested_nodes:
                        priority.append(neighbor)
            if priority:
                # Among priority neighbors, pick the highest degree
                action = max(priority, key=lambda x: env.degrees[x])
            else:
                # Fallback: degree heuristic when no positive neighbors
                action = max(untested, key=lambda x: env.degrees[x])
        
        if env.tests_used == env.max_tests - 1:
            positives = np.where(env.tested_nodes == 1)[0]
            if len(positives) > 0:
                action = positives[0] + num_nodes
            else:
                action = random.choice(range(num_nodes)) + num_nodes
                
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if action < num_nodes:
            tested_nodes.add(action)
            # Track positive results for contact tracing
            if env.tested_nodes[action] == 1:
                positive_nodes.add(action)
            action_desc = f"Test Node {action}"
        else:
            action_desc = f"Guess Node {action - num_nodes}"
            
        print(f"Action: {action_desc} | Reward: {reward:.2f}")
        log_data["steps"].append({"action": action_desc, "reward": float(reward)})
    
    final_result = reward > 0
    print(f"[{strategy.upper()}] Episode Finished. Final Guess Correct? {final_result}. True Patient Zero: {env.patient_zero}")
    
    log_data["final_correct"] = bool(final_result)
    log_data["true_patient_zero"] = int(env.patient_zero)
    return log_data

if __name__ == "__main__":
    print("Loading 50,000 node subgraph for Demo...")
    graph = load_snap_livejournal(subgraph_size=50000)
    
    env_config = {
        "graph": graph,
        "max_tests": 15,
        "infection_prob": 0.3,
        "simulation_steps": 5
    }
    env = OutbreakEnv(env_config)
    
    all_results = {}
    print("\n--- Running Random Testing Baseline ---")
    all_results["random"] = run_baseline(env, strategy="random")
    
    print("\n--- Running Degree Heuristic Baseline ---")
    all_results["degree"] = run_baseline(env, strategy="degree")
    
    print("\n--- Running Contact-Tracing Baseline ---")
    all_results["contact_tracing"] = run_baseline(env, strategy="contact_tracing")
    
    # Save results to JSON for UI
    results_path = "results_demo.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Also sync to ui/ folder for portable dashboard access
    import shutil
    import os
    if os.path.exists("ui"):
        shutil.copy(results_path, os.path.join("ui", results_path))
        
    print(f"Demo script completed. Results saved to '{results_path}'.")
