# RL as an Outbreak Detective 🕵️‍♂️🦠

This project implements a Reinforcement Learning agent capable of identifying a hidden "Patient Zero" across a massive community contact network under a heavily constrained testing budget.

## Project Architecture
- **Environment (*`env/outbreak_env.py`*):** A custom Gymnasium SIR simulation.
- **Dataset (*`dataset_utils.py`*):** Automates the download of the Stanford SNAP LiveJournal network (4 million nodes, 34 million edges).
- **Core Scripts:** Scripts to train and benchmark the AI agent using Ray RLlib to leverage High-Performance Computing (HPC) nodes.

---

## 🚀 Execution Order

Ensure your virtual environment is activated before running any scripts:
```bash
source venv/bin/activate
```

### 0. Run Unit Tests (`tests/test_env.py`)
Validate the environment is correctly implemented before running anything:
```bash
python -m pytest tests/test_env.py -v
```
> All 6 tests should pass (SIR conservation, Gym API, action rewards, re-test penalty, correct/wrong guess rewards).

### 1. Run the Baselines (`demo.py`)
Benchmark three simplistic strategies (Random, Degree-Heuristic, Contact-Tracing) to see how poorly non-RL approaches perform at tracking Patient Zero.
```bash
python demo.py
```
> Saves results to `results_demo.json` with all 3 strategies.

### 2. Train the RL Agent (`train.py`)
Uses **Ray RLlib** to spawn parallel environments (mapped to 26 CPU cores) and trains a PPO agent to locate the outbreak source.
```bash
python train.py
```
> The agent's *Mean Reward* is printed each iteration — watch it climb! Saves checkpoint to `checkpoints/` and metrics to `results_train.json`.

### 3. Evaluate the Trained Agent (`evaluate.py`)
Load the trained brain from the `checkpoints/` directory and run a full episode step-by-step.
```bash
python evaluate.py
```
> Saves step-by-step evaluation to `results_evaluate.json`.

### 4. Visualize Results (`visualize.py`)
Generate publication-quality plots comparing all strategies and training progress.
```bash
python visualize.py
```
> Creates 3 PNG charts in the `plots/` folder:
> - `baseline_steps.png` — Cumulative reward per step for all 3 baselines
> - `training_curve.png` — PPO mean reward over training iterations
> - `accuracy_comparison.png` — Final accuracy: RL vs all baselines

### 5. Open the Dashboard (`ui/index.html`)
Open in any web browser to view the interactive live dashboard:
```bash
# Option A — Firefox
firefox ui/index.html

# Option B — Python HTTP server (recommended for fetch() to work)
cd ui && python -m http.server 8080
# Then open http://localhost:8080 in your browser
```
> Shows 4 sections: Dataset Stats, Baseline Results, Training Metrics, RL Evaluation.
