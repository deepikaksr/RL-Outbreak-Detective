import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx

class OutbreakEnv(gym.Env):
    """
    RL Environment for identifying Patient Zero in an outbreak network.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, env_config=None):
        super().__init__()
        
        if env_config is None:
            env_config = {}
            
        # Determine the graph
        self.graph = env_config.get("graph", nx.barabasi_albert_graph(100, 3))
        self.num_nodes = self.graph.number_of_nodes()
        self.max_tests = env_config.get("max_tests", 15)
        self.infection_prob = env_config.get("infection_prob", 0.3)
        self.simulation_steps = env_config.get("simulation_steps", 5)
        
        # State: for each node -> [test_result, degree, positive_tested_neighbors_count]
        # test_result: -1 (untested), 0 (negative), 1 (positive)
        self.observation_space = spaces.Box(
            low=-1, high=np.inf, 
            shape=(self.num_nodes * 3,), dtype=np.float32
        )
        
        # Action space: 
        # 0 to N-1: Test node i
        # N to 2N-1: Guess node (i-N) is patient zero and terminate episode
        self.action_space = spaces.Discrete(self.num_nodes * 2)
        
        # Precompute degrees and adjacency list for fast simulation
        self.degrees = np.array([self.graph.degree(i) for i in range(self.num_nodes)])
        self.adj_list = {i: list(self.graph.neighbors(i)) for i in range(self.num_nodes)}
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Select patient zero
        self.patient_zero = random.randint(0, self.num_nodes - 1)
        
        # SIR States: 0 = Susceptible, 1 = Infected, 2 = Recovered
        self.node_states = np.zeros(self.num_nodes, dtype=int)
        self.node_states[self.patient_zero] = 1
        
        # Simulate outbreak before agent starts interacting
        for _ in range(self.simulation_steps):
            new_infected = []
            for node in range(self.num_nodes):
                if self.node_states[node] == 1:
                    # Infect neighbors
                    for neighbor in self.adj_list[node]:
                        if self.node_states[neighbor] == 0:
                            if random.random() < self.infection_prob:
                                new_infected.append(neighbor)
                    # Node recovers automatically (or has a chance). Let's say 100% chance for simple 5-step SIR.
                    self.node_states[node] = 2
            for n in new_infected:
                self.node_states[n] = 1
                
        # Agent's testing tracking
        self.tested_nodes = np.full(self.num_nodes, -1) # -1 means untested
        self.positive_neighbors = np.zeros(self.num_nodes)
        self.tests_used = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.column_stack((self.tested_nodes, self.degrees, self.positive_neighbors))
        return obs.flatten().astype(np.float32)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        
        # Action is a guess
        if action >= self.num_nodes:
            guess = action - self.num_nodes
            terminated = True
            if guess == self.patient_zero:
                reward = 10.0 # Huge reward for finding patient zero
            else:
                reward = -5.0 # Penalty for wrong guess
        else:
            # Action is a test
            node_to_test = action
            
            if self.tested_nodes[node_to_test] != -1:
                # Already tested, illegal or wasteful move
                reward = -1.0 # high penalty to discourage testing same node
            else:
                self.tests_used += 1
                # If Infected or Recovered, test is positive
                is_positive = 1 if self.node_states[node_to_test] in [1, 2] else 0
                self.tested_nodes[node_to_test] = is_positive
                
                # Update neighbor knowledge 
                if is_positive == 1:
                    for neighbor in self.adj_list[node_to_test]:
                        self.positive_neighbors[neighbor] += 1
                        
                reward = -0.05 # Cost per test
                
            if self.tests_used >= self.max_tests:
                # Ran out of budget
                truncated = True
                reward -= 5.0 # Penalty for failing to guess in time
                
        return self._get_obs(), reward, terminated, truncated, {}
