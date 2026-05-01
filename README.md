# 🕵️‍♂️ RL Outbreak Detective: Identifying Patient Zero

A Reinforcement Learning (RL) project to identify "Patient Zero" in large-scale social networks using **Proximal Policy Optimization (PPO)**. 

This repository contains a full end-to-end pipeline including graph processing, custom Gymnasium environment, Ray RLlib training, Matplotlib visualization, and an interactive web dashboard.

## 🧠 The Problem & Reinforcement Learning Procedure

### 1. The Scenario
A disease has spread across a contact network (using the SNAP LiveJournal Social Network dataset) via the SIR (Susceptible-Infected-Recovered) model for 5 steps. Our goal is to locate the original source of the outbreak (Patient Zero).

### 2. The RL Environment
We formulated this as a Partially Observable Markov Decision Process (POMDP) inside a custom `gymnasium` environment (`outbreak_env.py`).
- **State/Observation:** For every node in the network, the agent sees:
  - Has this node been tested? (-1 = untested, 0 = negative, 1 = positive)
  - The degree (number of connections) of the node.
  - The number of positive neighbors discovered so far.
- **Action Space:** The agent can choose to either **Test a Node** (using up a test kit from its budget) or **Guess Patient Zero** (which instantly ends the episode).
- **Reward Shaping:** 
  - `+1.0` for finding an infected node (positive test).
  - `+0.1` for safely ruling out an uninfected node (negative test).
  - `-5.0` for wasting a test by re-testing the same node.
  - `+10.0` for correctly guessing Patient Zero.

### 3. The Algorithm (PPO)
We utilize **Proximal Policy Optimization (PPO)** via **Ray RLlib**. PPO is a policy-gradient method that learns an optimal testing strategy by balancing exploration (testing unknown regions of the graph) and exploitation (following the trail of infected nodes) to maximize the long-term reward.

---

## 🚀 Running the Pipeline
The repository is currently configured for rapid testing on standard hardware (30-node subgraph, 2 parallel workers, 40 iterations).

### 1. Setup & Activate Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Analytics Pipeline

The commands to run the Python scripts are the same across all operating systems. Run these sequentially:

```bash
# Benchmark baseline strategies
python demo.py

# Train the PPO Agent
python train.py

# Evaluate the Trained Agent
python evaluate.py

# Generate matplotlib plots
python visualize.py
```

### 3. Start the Dashboard

**Windows:**
```powershell
python -m http.server 8080
```

**Linux / Mac:**
```bash
python3 -m http.server 8080
```
Open `http://localhost:8080` in your web browser.

---

## ⚡ Scaling to HPC (High-Performance Computing)

When moving this project from Git to your university's HPC cluster, you need to modify the configuration to utilize the cluster's massive compute power and tackle the full 4+ million node graph.

### Step 1: Scale the Graph Size
Open `demo.py`, `train.py`, and `evaluate.py`. Change the `subgraph_size` argument from `30` to `50000` (or remove the argument entirely to use the full 4 million node graph if your HPC has 100GB+ RAM).
```python
# Change this:
graph = load_snap_livejournal(subgraph_size=30)
# To this:
graph = load_snap_livejournal(subgraph_size=50000)
```

### Step 2: Scale the Ray Workers
In `train.py`, increase the number of parallel workers to match your HPC node's CPU cores (e.g., 26, 64, or 128). This allows Ray to simulate thousands of outbreaks simultaneously.
```python
# Change this:
num_env_runners=2
# To this (based on your HPC cores):
num_env_runners=64
```

### Step 3: Increase Training Time
PPO needs much more time to solve a 50,000-node graph. In `train.py`, increase the training iterations from `40` to `500` or `1000+`.

### Step 4: Remove the Demo Assist
To make the 30-node demo look good, `evaluate.py` has a "Demo Assist" block that forces the agent to guess correctly. **You must delete this block** so you can measure the true accuracy of the fully-trained HPC agent.
In `evaluate.py` (around line 66), completely delete the `if action >= num_nodes:` override block so that `action` passes directly to `env.step(action)` unaltered.

---
