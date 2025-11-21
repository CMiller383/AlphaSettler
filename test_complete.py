#!/usr/bin/env python3
"""
Complete end-to-end test for AlphaSettler.
Tests full pipeline: game engine → MCTS → neural network → training.
"""

import sys
import os
import torch
import numpy as np

# Add python directory to path to import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# Import catan_engine once at module level to avoid double registration
import catan_engine
from catan_engine import (
    GameState, generate_legal_actions, apply_action,
    AlphaZeroMCTS, AlphaZeroConfig, NNEvaluation,
    BatchedEvaluator, BatchedEvaluatorConfig
)

def test_game_engine():
    """Test game state and move generation."""
    
    print("Testing game engine...")
    state = GameState.create_new_game(4, 42)
    assert not state.is_game_over()
    
    actions = generate_legal_actions(state)
    assert len(actions) > 0, "Should have legal actions"
    
    # Apply an action
    apply_action(state, actions[0], 123)
    print("✓ Game engine works")

def test_mcts():
    """Test MCTS search."""
    
    print("Testing MCTS...")
    state = GameState.create_new_game(4, 42)
    
    # Simple evaluator
    def evaluator(s):
        actions = generate_legal_actions(s)
        n = len(actions)
        eval_result = NNEvaluation()
        eval_result.policy = [1.0/n] * n if n > 0 else []
        eval_result.value = 0.0
        return eval_result
    
    config = AlphaZeroConfig()
    config.num_simulations = 25
    mcts = AlphaZeroMCTS(config, evaluator)
    
    action = mcts.search(state)
    probs = mcts.get_action_probabilities()
    
    assert len(probs) > 0, "Should return action probabilities"
    assert abs(sum(probs) - 1.0) < 0.01, "Probabilities should sum to 1"
    print("✓ MCTS works")

def test_neural_network():
    """Test neural network forward pass."""
    from catan_network import CatanNetwork
    from state_encoder import StateEncoder
    
    print("Testing neural network...")
    encoder = StateEncoder()
    network = CatanNetwork(
        input_size=encoder.total_features,
        hidden_size=64,
        num_residual_blocks=2,
        max_action_space=300
    )
    network.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
    
    state = GameState.create_new_game(4, 42)
    features = encoder.encode_state(state, 0)
    
    # Forward pass
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        policy, value = network(features_tensor)
    
    assert policy.shape[0] == 1
    assert value.shape == (1, 1)
    print("✓ Neural network works")

def test_batched_evaluation():
    """Test batched NN evaluation using make_batched_nn_evaluator."""
    from catan_network import CatanNetwork
    from state_encoder import StateEncoder
    from catan_engine import make_batched_nn_evaluator
    
    print("Testing batched evaluation...")
    encoder = StateEncoder()
    network = CatanNetwork(
        input_size=encoder.total_features,
        hidden_size=64,
        num_residual_blocks=2,
        max_action_space=300
    )
    network.eval()  # Set to eval mode for inference
    
    def batch_callback(states_data):
        results = []
        for features, num_actions in states_data:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                policy, value = network(features_tensor)
                probs = torch.softmax(policy[0, :num_actions], dim=0).numpy().tolist()
                results.append((probs, value.item()))
        return results
    
    config = BatchedEvaluatorConfig()
    config.max_batch_size = 4
    config.timeout_ms = 10
    
    evaluator = BatchedEvaluator(config, batch_callback)
    evaluator.start()
    
    # Create NN evaluator function compatible with MCTS
    nn_eval_func = make_batched_nn_evaluator(evaluator)
    
    # Test evaluation with a game state
    state = GameState.create_new_game(4, 42)
    result = nn_eval_func(state)
    
    # Verify result structure
    assert hasattr(result, 'policy'), "Result should have policy attribute"
    assert hasattr(result, 'value'), "Result should have value attribute"
    assert len(result.policy) > 0, "Policy should have elements"
    assert isinstance(result.value, float), "Value should be float"
    
    evaluator.stop()
    print("✓ Batched evaluation works")

def test_training_step():
    """Test one training iteration."""
    from config import QuickTestConfig
    from catan_network import CatanNetwork
    from state_encoder import StateEncoder
    import torch.optim as optim
    import torch.nn.functional as F
    
    print("Testing training step...")
    config = QuickTestConfig()
    encoder = StateEncoder()
    max_action_space = 300
    
    network = CatanNetwork(
        input_size=encoder.total_features,
        hidden_size=64,
        num_residual_blocks=2,
        max_action_space=max_action_space
    )
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    # Create dummy training data with batch_size=2 (BatchNorm requires > 1)
    states = [GameState.create_new_game(4, 42), GameState.create_new_game(4, 123)]
    features_list = [encoder.encode_state(s, 0) for s in states]
    num_actions_list = [len(generate_legal_actions(s)) for s in states]
    
    # Stack into batch (convert to numpy first to avoid warning)
    features_batch = torch.from_numpy(np.array(features_list, dtype=np.float32))
    
    # Dummy targets for batch
    policy_targets = torch.zeros(2, max_action_space)
    for i, num_actions in enumerate(num_actions_list):
        policy_targets[i, :num_actions] = 1.0 / num_actions
    value_targets = torch.tensor([[0.5], [0.3]])
    
    # Forward pass
    network.train()  # Set to train mode
    policy_logits, values = network(features_batch)
    
    # Compute losses
    policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum() / 2
    value_loss = F.mse_loss(values, value_targets)
    total_loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    assert total_loss.item() > 0
    print("✓ Training step works")

def main():
    """Run all tests."""
    print("="*60)
    print("AlphaSettler Complete End-to-End Test")
    print("="*60)
    print()
    
    try:
        test_game_engine()
        test_mcts()
        test_neural_network()
        test_batched_evaluation()
        test_training_step()
        
        print()
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
