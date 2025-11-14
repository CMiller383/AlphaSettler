#!/usr/bin/env python3
"""
Test AlphaZero MCTS with a dummy neural network.
Verifies that the AlphaZero MCTS implementation works correctly.
"""

import numpy as np
from catan_engine import (
    GameState, AlphaZeroMCTS, AlphaZeroConfig, NNEvaluation,
    generate_legal_actions
)


def dummy_nn_evaluator(game_state):
    """
    Dummy neural network that returns uniform policy and zero value.
    Replace this with actual NN inference later.
    """
    # Get legal actions count
    legal_actions = generate_legal_actions(game_state)
    num_actions = len(legal_actions)
    
    # Return uniform policy and zero value
    eval_result = NNEvaluation()
    eval_result.policy = [1.0 / num_actions] * num_actions
    eval_result.value = 0.0
    
    return eval_result


def test_alphazero_search():
    """Test AlphaZero MCTS search with dummy NN."""
    print("Testing AlphaZero MCTS with dummy NN...\n")
    
    # Create game
    game = GameState.create_new_game(4, 42)
    
    # Configure AlphaZero MCTS
    config = AlphaZeroConfig()
    config.num_simulations = 50  # Low for testing
    config.cpuct = 1.5
    config.add_exploration_noise = True
    
    print(f"Config: {config}")
    
    # Create MCTS with dummy evaluator
    mcts = AlphaZeroMCTS(config, dummy_nn_evaluator)
    
    # Run search
    print("\nRunning MCTS search...")
    action = mcts.search(game)
    
    print(f"Selected action: {action}")
    
    # Get action probabilities
    probs = mcts.get_action_probabilities()
    print(f"\nAction probabilities from visit counts:")
    print(f"  Num actions: {len(probs)}")
    print(f"  Top 5 probs: {sorted(probs, reverse=True)[:5]}")
    print(f"  Sum of probs: {sum(probs):.4f}")
    
    print("\n✓ AlphaZero MCTS test passed!")


if __name__ == "__main__":
    test_alphazero_search()
