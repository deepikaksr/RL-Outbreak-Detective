# RL Outbreak Detective: Implementation Plan

## Goal Description
The objective is to build a Reinforcement Learning (RL) agent capable of identifying a hidden "patient zero" in a contact network under a strictly limited testing budget. Given the computational intensity of simulating outbreaks over complex graphs, we will leverage High-Performance Computing (HPC) for large-scale parallel training and dataset evaluation.

## Proposed Architecture and Tools

### 1. Environment & Simulation (The Outbreak Simulator)
- **Graph Framework:** Use `NetworkX` for graph operations, and `PyTorch Geometric` if Graph Neural Networks (GNNs) are used for state representation.
- **Disease Spreading Model:** Implement a vectorized SIR (Susceptible-Infected-Recovered) simulation. Vectorizing the adjacency matrix operations using `SciPy`/`NumPy` will ensure the simulator runs exceptionally fast on HPC CPUs, which is critical for RL rollout generation.
- **RL Environment:** Wrap the simulation into a standard `gymnasium.Env`.
  - *State:* Encoded features of the graph (e.g., node degrees, whether nodes have been tested, test results).
  - *Action Space:* Discrete choices of which node to test next.
  - *Reward:* +1 for finding patient zero at the end of the budget K, -1 otherwise, with a slight negative penalty (e.g., -0.01) per test to encourage early stopping if the patient is found.

### 2. Reinforcement Learning Algorithm & HPC
- **Library Choice:** Due to your requirement to run on an **HPC (High-Performance Computer)**, **Ray RLlib** is strongly recommended over Stable-Baselines3. RLlib natively handles distributed computing, allowing you to easily scale training across multiple nodes and thousands of cores without modifying the core logic.
- **Algorithm:** Proximal Policy Optimization (PPO), as proposed, due to its robustness.
- **HPC Scaling Strategy:**
  - *Parallel Rollouts:* RLlib will deploy multiple workers across the HPC cluster to simulate thousands of outbreaks simultaneously.
  - *Hyperparameter Sweeps:* Use `Ray Tune` via a SLURM batch script to sweep learning rates, testing budgets ($K$), and network sizes.

---

## Recommended Large-Scale Datasets (For HPC)
Testing on arbitrary random graphs is useful, but HPC allows us to train on massive, realistic empirical contact networks. Here are highly recommended dataset sources:

1. **SocioPatterns (High-Resolution Real Contact Networks)**
   - *URL:* http://www.sociopatterns.org/datasets/
   - *Why:* Contains real-world, face-to-face contact data over time in schools, hospitals, and exhibitions. This is the gold standard for disease spread.
   - *Scale:* Thousands of nodes. Great for foundational training and realistic generalization.

2. **SNAP: Location-based Social Networks (`loc-Gowalla` / `loc-Brightkite`)**
   - *URL:* http://snap.stanford.edu/data/
   - *Why:* Represents physical check-ins and encounters, heavily mirroring physical transmission vectors.
   - *Scale:* ~200,000 nodes, 1-2 million edges. Perfect for a mid-tier HPC training run.

3. **SNAP: `com-LiveJournal` or `com-Orkut`**
   - *Why:* Massive social networks to truly push the boundaries of the RL agent and HPC cluster.
   - *Scale:* 3 to 4 million nodes, 34 to 117 million edges. 
   - *Usage:* Train locally generated subgraphs, then deploy the RL agent on the entire LiveJournal network to trace patient zero.

4. **EpiFast / NDSSL Synthetic Populations**
   - *Why:* Highly realistic synthetic populations designed exactly for epidemiological studies by Virginia Tech. 

---

## Verification Plan

### Automated Tests
- **Simulator Constraints:** Unit test the SIR transition matrix to ensure total population $S + I + R$ remains constant across timesteps.
- **Environment Interface:** Run `gymnasium.utils.env_checker` to validate the environment API.
- **Mock Training:** Execute a local RLlib PPO run (with 1 worker and 5 iterations) to catch any tensor shape mismatches or memory leaks before dispatching to the HPC queue.

### Manual Verification
- **Visual Trace:** Extract a single episode on a small 50-node graph and visualize the step-by-step testing actions using `NetworkX` and `Matplotlib` to visualize the agent's logic.
- **Baseline Comparison:** Plot the reward and accuracy curves of the RL agent against the Degree-Heuristic and Random-Testing baselines to empirically prove that RL provides an advantage.
