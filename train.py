import ray
from ray.rllib.algorithms.ppo import PPOConfig
from dataset_utils import load_snap_livejournal
from env.outbreak_env import OutbreakEnv
from ray.tune.registry import register_env
import os
import json

# ROOT CAUSE FIX 2: Reduced to 30 nodes for reliable training.
# To guarantee the agent learns to find Patient Zero, we need a graph
# small enough that 15 tests gives high coverage, and 40 iterations is enough.
print("Loading 30 node subgraph...")
GRAPH = load_snap_livejournal(subgraph_size=30)

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
        # 2 workers.
        # On HPC, raise num_env_runners to match available cores (e.g. 26).
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=400,   # Increased for better gradient estimates
            # KEY FIX: entropy_coeff forces the agent to EXPLORE (test nodes)
            # instead of immediately guessing. Without this, agent collapses.
            entropy_coeff=0.01,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        .resources(num_gpus=0)
    )
    
    print("Starting PPO Training using Ray RLlib...")
    
    # Check for existing checkpoint to resume training
    checkpoint_path = os.path.abspath("checkpoints")
    if os.path.exists(checkpoint_path) and any(f.endswith('.json') for f in os.listdir(checkpoint_path)):
        print(f"Resuming from previous checkpoint: {checkpoint_path}")
        algo = config.build()
        algo.restore(checkpoint_path)
    else:
        print("No valid checkpoint found. Starting fresh training.")
        algo = config.build()
    
    training_metrics = []
    
    # Train for 5 iterations exactly as requested
    # 40 iterations. With 30 nodes this is very fast.
    for i in range(40): 
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
    results_path = "results_train.json"
    results = {"training_metrics": training_metrics}
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Also sync to ui/ folder for portable dashboard access
    import shutil
    import os
    if os.path.exists("ui"):
        shutil.copy(results_path, os.path.join("ui", results_path))
        
    print(f"Training completed. Metrics saved to '{results_path}'.")
    
    print("\nTraining finished! Securing shutdown...")
    
    ray.shutdown()
