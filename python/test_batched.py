"""Simple test of batched evaluator without threading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from catan_engine import (
    GameState,
    AlphaZeroMCTS,
    AlphaZeroConfig,
    BatchedEvaluator,
    BatchedEvaluatorConfig,
    make_batched_nn_evaluator,
    generate_legal_actions,
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import QuickTestConfig


def simple_test():
    """Test batched evaluator with single threaded execution."""
    print("Testing BatchedEvaluator...")
    
    # Setup
    config = QuickTestConfig()
    encoder = StateEncoder()
    
    network = CatanNetwork(
        input_size=encoder.get_feature_size(),
        hidden_size=config.hidden_size,
        num_residual_blocks=config.num_residual_blocks,
    ).to(config.device)
    network.eval()
    
    print(f"✓ Network created ({config.device})")
    
    # Batch callback
    call_count = [0]
    
    def batch_callback(batch):
        print(f"  [PYTHON] Batch callback ENTERED (batch_size={len(batch)})")
        call_count[0] += 1
        
        # Extract encoded states
        encoded_states = [item[0] for item in batch]
        num_actions_list = [item[1] for item in batch]
        
        print(f"  [PYTHON] Extracted {len(encoded_states)} states")
        
        # Convert to tensor
        states_tensor = torch.tensor(
            np.array(encoded_states),
            dtype=torch.float32,
            device=config.device
        )
        
        print(f"  [PYTHON] Tensor created, shape: {states_tensor.shape}")
        
        # Inference
        with torch.no_grad():
            policy_logits, values = network(states_tensor)
        
        print(f"  [PYTHON] NN inference done")
        
        # Convert results
        results = []
        for i in range(len(batch)):
            num_actions = num_actions_list[i]
            if num_actions == 0:
                results.append(([], 0.0))
                continue
            
            logits = policy_logits[i, :num_actions]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            value = torch.tanh(values[i]).item()
            
            results.append((probs.tolist(), value))
        
        print(f"  [PYTHON] Batch callback RETURNING {len(results)} results")
        return results
    
    # Test 1: Disable batching (immediate evaluation)
    print("\nTest 1: Batching disabled (immediate evaluation)")
    batch_config = BatchedEvaluatorConfig()
    batch_config.enable_batching = False
    
    evaluator = BatchedEvaluator(batch_config, batch_callback)
    print("✓ BatchedEvaluator created")
    
    # Create test state
    state = GameState.create_new_game(4, 0)
    legal_actions = generate_legal_actions(state)
    print(f"✓ Test state created ({len(legal_actions)} legal actions)")
    
    # Single evaluation
    call_count[0] = 0
    policy, value = evaluator.evaluate(state, 0, len(legal_actions))
    print(f"✓ Evaluation completed: {len(policy)} probs, value={value:.3f}")
    print(f"  Callback called {call_count[0]} times")
    
    # Test 2: With batching enabled but single request
    print("\nTest 2: Batching enabled (single request)")
    batch_config.enable_batching = True
    batch_config.min_batch_size = 1  # Process immediately
    batch_config.max_batch_size = 8
    batch_config.timeout_ms = 100
    
    evaluator2 = BatchedEvaluator(batch_config, batch_callback)
    evaluator2.start()
    print("✓ BatchedEvaluator started")
    
    call_count[0] = 0
    policy, value = evaluator2.evaluate(state, 0, len(legal_actions))
    print(f"✓ Evaluation completed: {len(policy)} probs, value={value:.3f}")
    print(f"  Callback called {call_count[0]} times")
    
    evaluator2.stop()
    print("✓ BatchedEvaluator stopped")
    
    # Test 3: MCTS with batched evaluator
    print("\nTest 3: MCTS with batched evaluator")
    mcts_config = AlphaZeroConfig()
    mcts_config.num_simulations = 10  # Small number for testing
    mcts_config.add_exploration_noise = True
    
    evaluator3 = BatchedEvaluator(batch_config, batch_callback)
    evaluator3.start()
    
    nn_evaluator = make_batched_nn_evaluator(evaluator3)
    mcts = AlphaZeroMCTS(mcts_config, nn_evaluator)
    
    print("✓ MCTS created")
    
    call_count[0] = 0
    action = mcts.search(state)
    print(f"✓ MCTS search completed")
    print(f"  Callback called {call_count[0]} times")
    print(f"  Selected action: {action.type}")
    
    evaluator3.stop()
    print("✓ BatchedEvaluator stopped")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    simple_test()
