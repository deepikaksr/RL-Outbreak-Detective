import ray
from ray.rllib.algorithms.ppo import PPOConfig
from dataset_utils import load_snap_livejournal
from env.outbreak_env import OutbreakEnv
from ray.tune.registry import register_env
import os
import json

# Load 50,000 node subgraph for training
print("Loading 50,000 node subgraph...")
GRAPH = load_snap_livejournal(subgraph_size=50000)

def env_creator(env_config):
    return OutbreakEnv(env_config)

register_env("outbreak_env", env_creator)

if __name__ == "__main__":
    # Initialize Ray. In a real HPC, you'd connect to the existing Ray cluster here.
    ray.init(ignore_reinit_error=True)
    
    config = (
        PPOConfig()
        .environment(
            env="outbreak_env",
            env_config={
                "graph": GRAPH,
                "max_tests": 15,
                "infection_prob": 0.3,
                "simulation_steps": 5
            }
        )
        .framework("torch")
        # Reduced workers for stability in limited environments
        .env_runners(num_env_runners=4, num_envs_per_env_runner=1) 
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=500,
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            }
        )
        .resources(num_gpus=0)
    )
    
    print("Starting PPO Training using Ray RLlib...")
    algo = config.build()
    
    training_metrics = []
    
    # Train for 5 iterations exactly as requested
    for i in range(5): 
        result = algo.train()
        mean_reward = result.get('env_runners', {}).get('episode_return_mean', 'N/A')
        print(f"Iteration {i+1} completed. Env Runners metrics: {mean_reward}")
        
        training_metrics.append({
            "iteration": i + 1,
            "mean_reward": float(mean_reward) if mean_reward != 'N/A' else None
        })
        
    checkpoint_dir = algo.save(checkpoint_dir=os.path.abspath("checkpoints"))
    print(f"Training completed. Checkpoint saved to: {checkpoint_dir}")
    
    # Save the UI outputs to match demo and evaluation!
    with open('results_train.json', 'w') as f:
        json.dump({"training_metrics": training_metrics}, f, indent=4)
    print("Training metrics saved to 'results_train.json'.")
    
    print("\nTraining finished! Securing shutdown...")
    
    ray.shutdown()
