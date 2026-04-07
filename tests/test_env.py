import unittest
import sys
import os
import numpy as np
import networkx as nx

# Ensure the parent directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.outbreak_env import OutbreakEnv


def make_env(num_nodes=50, max_tests=10, seed=42):
    """Helper: create a small test environment with a reproducible BA graph."""
    graph = nx.barabasi_albert_graph(num_nodes, 3, seed=seed)
    graph = nx.convert_node_labels_to_integers(graph)
    config = {
        "graph": graph,
        "max_tests": max_tests,
        "infection_prob": 0.5,
        "simulation_steps": 3,
    }
    return OutbreakEnv(config)


class TestSIRConservation(unittest.TestCase):
    """S + I + R must always equal N throughout the simulation reset."""

    def test_sir_conservation(self):
        env = make_env()
        for trial in range(5):
            env.reset(seed=trial)
            N = env.num_nodes
            states = env.node_states
            S = np.sum(states == 0)
            I = np.sum(states == 1)
            R = np.sum(states == 2)
            self.assertEqual(S + I + R, N,
                msg=f"Trial {trial}: S+I+R={S+I+R} ≠ N={N}")


class TestGymAPI(unittest.TestCase):
    """Environment must pass the official Gymnasium checker."""

    def test_gym_api(self):
        from gymnasium.utils.env_checker import check_env
        env = make_env()
        # check_env raises AssertionError on failures
        check_env(env, warn=True, skip_render_check=True)


class TestTestAction(unittest.TestCase):
    """Testing a NEW node must give -0.05 reward."""

    def test_test_action_reward(self):
        env = make_env()
        env.reset(seed=0)
        # Test node 0 — guaranteed untested after reset
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertAlmostEqual(reward, -0.05, places=5,
            msg="Testing a new node should cost -0.05")


class TestRetestPenalty(unittest.TestCase):
    """Re-testing the same node must give -1.0 reward."""

    def test_retest_penalty(self):
        env = make_env()
        env.reset(seed=0)
        env.step(0)           # First test — valid
        _, reward, _, _, _ = env.step(0)  # Re-test — should penalise
        self.assertAlmostEqual(reward, -1.0, places=5,
            msg="Re-testing the same node should give -1.0 penalty")


class TestCorrectGuessReward(unittest.TestCase):
    """Guessing the true patient_zero must give +10.0 reward."""

    def test_correct_guess_reward(self):
        env = make_env()
        env.reset(seed=0)
        patient_zero = env.patient_zero
        num_nodes = env.num_nodes
        # Guess action = patient_zero + num_nodes
        _, reward, terminated, _, _ = env.step(patient_zero + num_nodes)
        self.assertTrue(terminated, "Guess action should terminate the episode")
        self.assertAlmostEqual(reward, 10.0, places=5,
            msg="Correct guess should yield +10 reward")


class TestWrongGuessReward(unittest.TestCase):
    """Guessing the wrong node must give -5.0 reward."""

    def test_wrong_guess_reward(self):
        env = make_env()
        env.reset(seed=0)
        patient_zero = env.patient_zero
        num_nodes = env.num_nodes
        # Pick any node that is NOT patient zero
        wrong_guess = (patient_zero + 1) % num_nodes
        _, reward, terminated, _, _ = env.step(wrong_guess + num_nodes)
        self.assertTrue(terminated, "Guess action should terminate the episode")
        self.assertAlmostEqual(reward, -5.0, places=5,
            msg="Wrong guess should yield -5 reward")


if __name__ == "__main__":
    unittest.main(verbosity=2)
