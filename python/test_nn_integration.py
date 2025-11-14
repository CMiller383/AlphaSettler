#!/usr/bin/env python3
"""
Integration test for neural network + AlphaZero MCTS.
Verifies that all components work together correctly.
"""

import numpy as np
import torch

from catan_engine import (
    GameState, AlphaZeroMCTS, AlphaZeroConfig, NNEvaluation,
    generate_legal_actions, apply_action
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork, create_network


def test_state_encoder():
    """Test state encoder produces correct shape."""
    print("Testing state encoder...")
    
    encoder = StateEncoder()
    state = GameState.create_new_game(4, 42)
    
    # Encode state
    features = encoder.encode_state(state, 0)
    
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature size: {encoder.get_feature_size()}")
    print(f"  All finite: {np.all(np.isfinite(features))}")
    
    assert features.shape[0] == encoder.get_feature_size(), "Feature size mismatch"
    assert np.all(np.isfinite(features)), "Non-finite values in features"
    
    print("  ✓ State encoder working\n")


def test_network_forward():
    """Test network forward pass."""
    print("Testing network forward pass...")
    
    encoder = StateEncoder()
    network = create_network(encoder.get_feature_size(), hidden_size=256, num_blocks=2)
    
    # Create dummy input
    batch_size = 4
    state_features = torch.randn(batch_size, encoder.get_feature_size())
    
    # Forward pass without mask
    policy_logits, value = network(state_features)
    
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    assert policy_logits.shape == (batch_size, 300), "Policy shape mismatch"
    assert value.shape == (batch_size, 1), "Value shape mismatch"
    assert torch.all(torch.isfinite(policy_logits)), "Non-finite policy logits"
    assert torch.all(torch.isfinite(value)), "Non-finite values"
    
    # Test with action mask
    action_mask = torch.rand(batch_size, 300) > 0.5
    policy_logits_masked, _ = network(state_features, action_mask)
    
    # Check that masked actions have very negative logits
    masked_out = ~action_mask
    assert torch.all(policy_logits_masked[masked_out] < -1e8), "Masking not working"
    
    print("  ✓ Network forward pass working\n")


def test_network_predict():
    """Test network prediction interface."""
    print("Testing network prediction...")
    
    encoder = StateEncoder()
    network = create_network(encoder.get_feature_size(), hidden_size=256, num_blocks=2)
    
    # Create game state
    state = GameState.create_new_game(4, 42)
    features = encoder.encode_state(state, 0)
    
    # Get legal actions
    legal_actions = generate_legal_actions(state)
    legal_indices = list(range(len(legal_actions)))
    
    print(f"  Number of legal actions: {len(legal_actions)}")
    
    # Predict
    policy, value = network.predict(features, legal_indices)
    
    print(f"  Policy shape: {policy.shape}")
    print(f"  Policy sum: {policy.sum():.6f}")
    print(f"  Value: {value:.3f}")
    
    assert policy.shape[0] == len(legal_actions), "Policy size mismatch"
    assert abs(policy.sum() - 1.0) < 1e-5, "Policy doesn't sum to 1"
    assert -1 <= value <= 1, "Value out of range"
    
    print("  ✓ Network prediction working\n")


def test_alphazero_with_nn():
    """Test AlphaZero MCTS with real neural network."""
    print("Testing AlphaZero MCTS with neural network...")
    
    encoder = StateEncoder()
    network = create_network(encoder.get_feature_size(), hidden_size=128, num_blocks=2)
    
    # Create evaluator function
    def nn_evaluator(state):
        player = state.current_player
        features = encoder.encode_state(state, player)
        legal_actions = generate_legal_actions(state)
        
        if len(legal_actions) == 0:
            result = NNEvaluation()
            result.policy = []
            result.value = 0.0
            return result
        
        legal_indices = list(range(len(legal_actions)))
        policy, value = network.predict(features, legal_indices)
        
        result = NNEvaluation()
        result.policy = policy.tolist()
        result.value = float(value)
        return result
    
    # Create MCTS
    config = AlphaZeroConfig()
    config.num_simulations = 50  # Small number for testing
    
    mcts = AlphaZeroMCTS(config)
    
    # Run search
    state = GameState.create_new_game(4, 42)
    action_idx = mcts.search(state, nn_evaluator)
    
    print(f"  Selected action index: {action_idx}")
    
    # Get action probabilities
    probs = mcts.get_action_probabilities(state, temperature=1.0)
    print(f"  Number of actions with probabilities: {len(probs)}")
    print(f"  Probability sum: {sum(probs):.6f}")
    
    assert action_idx >= 0, "Invalid action index"
    assert abs(sum(probs) - 1.0) < 1e-5, "Probabilities don't sum to 1"
    
    print("  ✓ AlphaZero MCTS with NN working\n")


def test_play_one_game():
    """Test playing a complete game with the network."""
    print("Testing complete game playthrough...")
    
    encoder = StateEncoder()
    network = create_network(encoder.get_feature_size(), hidden_size=128, num_blocks=2)
    
    def nn_evaluator(state):
        player = state.current_player
        features = encoder.encode_state(state, player)
        legal_actions = generate_legal_actions(state)
        
        if len(legal_actions) == 0:
            result = NNEvaluation()
            result.policy = []
            result.value = 0.0
            return result
        
        legal_indices = list(range(len(legal_actions)))
        policy, value = network.predict(features, legal_indices)
        
        result = NNEvaluation()
        result.policy = policy.tolist()
        result.value = float(value)
        return result
    
    # Create game
    state = GameState.create_new_game(4, 123)
    
    config = AlphaZeroConfig()
    config.num_simulations = 25  # Very small for speed
    
    turn_count = 0
    max_turns = 100  # Limit for testing
    
    print("  Playing game...")
    while not state.is_game_over() and turn_count < max_turns:
        legal_actions = generate_legal_actions(state)
        if len(legal_actions) == 0:
            break
        
        # Run MCTS
        mcts = AlphaZeroMCTS(config)
        action_idx = mcts.search(state, nn_evaluator)
        
        # Apply action
        if action_idx < len(legal_actions):
            action = legal_actions[action_idx]
            apply_action(state, action, np.random.randint(0, 1000000))
        
        turn_count += 1
        
        if turn_count % 20 == 0:
            print(f"    Turn {turn_count}...")
    
    print(f"  Game finished in {turn_count} turns")
    print(f"  Game over: {state.is_game_over()}")
    
    if state.is_game_over():
        winner = state.get_winner()
        print(f"  Winner: Player {winner}")
    
    print("  ✓ Complete game playthrough working\n")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Neural Network Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_state_encoder()
        test_network_forward()
        test_network_predict()
        test_alphazero_with_nn()
        test_play_one_game()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
