import ray
from ray.rllib.algorithms.algorithm import Algorithm
from dataset_utils import load_snap_livejournal
from env.outbreak_env import OutbreakEnv
from ray.tune.registry import register_env
import os
import json # ADDED JSON EXPORT API

# Load 50,000 node subgraph for evaluation
print("Loading 50,000 node subgraph for Evaluation...")
GRAPH = load_snap_livejournal(subgraph_size=50000)

def env_creator(env_config):
    return OutbreakEnv(env_config)

register_env("outbreak_env", env_creator)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    checkpoint_path = "checkpoints"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at '{checkpoint_path}'. Make sure you run train.py first!")
        exit(1)
        
    print(f"Loading trained AI from checkpoint: {checkpoint_path}")
    # Load the trained agent
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Initialize the environment for testing
    env_config = {
        "graph": GRAPH,
        "max_tests": 15,
        "infection_prob": 0.3,
        "simulation_steps": 5
    }
    env = OutbreakEnv(env_config)
    
    print("\n--- Running Trained Agent Evaluation ---")
    obs, info = env.reset()
    done = False
    
    num_nodes = env.num_nodes
    log_data = {"steps": []}
    
    while not done:
        # Ask the trained brain for the best action based on the current observation
        action = algo.compute_single_action(obs, explore=False)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if action < num_nodes:
            action_desc = f"Tested Node {action}"
        else:
            action_desc = f"Guessed Node {action - num_nodes}"
            
        print(f"Agent Action: {action_desc} | Step Reward: {reward:.2f}")
        log_data["steps"].append({"action": action_desc, "reward": float(reward)})
            
    final_result = reward > 0
    print(f"\n[EVALUATION] Episode Finished.")
    print(f"Total Tests Used: {env.tests_used}")
    print(f"Final Guess Correct? {final_result}")
    print(f"True Patient Zero: {env.patient_zero}")
    
    log_data["total_tests"] = int(env.tests_used)
    log_data["final_correct"] = bool(final_result)
    log_data["true_patient_zero"] = int(env.patient_zero)
    
    # Overwrite prior logs natively
    with open('results_evaluate.json', 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print("\nEvaluation completed. Results saved to 'results_evaluate.json'.")
    ray.shutdown()
