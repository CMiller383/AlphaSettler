"""
Standalone AlphaZero agent for Catan.
Uses trained neural network with optimized batched MCTS evaluation.
"""

import torch
import numpy as np

from catan_engine import (
    GameState, AlphaZeroMCTS, AlphaZeroConfig,
    BatchedEvaluator, BatchedEvaluatorConfig,
    make_batched_nn_evaluator
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork


class AlphaZeroAgent:
    """
    AlphaZero agent that selects actions using MCTS with neural network guidance.
    Uses batched evaluation for optimal GPU performance.
    """
    
    def __init__(self, network, mcts_simulations=100, device='cuda', 
                 enable_batching=True, add_noise=False):
        """
        Initialize AlphaZero agent.
        
        Args:
            network: Trained CatanNetwork
            mcts_simulations: Number of MCTS simulations per move
            device: 'cuda' or 'cpu'
            enable_batching: Enable batched NN evaluation (much faster)
            add_noise: Add Dirichlet noise for exploration (training only)
        """
        self.network = network
        self.device = device
        self.encoder = StateEncoder()
        self.network.eval()
        
        # MCTS configuration
        self.mcts_config = AlphaZeroConfig()
        self.mcts_config.num_simulations = mcts_simulations
        self.mcts_config.num_parallel_sims = mcts_simulations  # Parallel tree descent
        self.mcts_config.cpuct = 1.5
        self.mcts_config.virtual_loss_penalty = 1.0
        self.mcts_config.add_exploration_noise = add_noise
        self.mcts_config.dirichlet_alpha = 0.3
        self.mcts_config.dirichlet_weight = 0.25
        
        # Setup batched evaluator if enabled
        self.batched_evaluator = None
        self.nn_evaluator = None
        
        if enable_batching:
            self._setup_batched_evaluator()
    
    def _setup_batched_evaluator(self):
        """Setup optimized batched evaluator."""
        batch_config = BatchedEvaluatorConfig()
        batch_config.max_batch_size = 64
        batch_config.timeout_ms = 10
        batch_config.enable_batching = True
        
        def batch_callback(stacked_states_flat, num_legal_actions, batch_size, feature_size):
            """Optimized batch callback with pre-stacked arrays."""
            if batch_size == 0:
                return []
            
            # Reshape flat array to 2D (zero-copy view)
            states_np = stacked_states_flat.reshape(batch_size, feature_size)
            states_tensor = torch.from_numpy(states_np).to(self.device, non_blocking=True)
            
            with torch.no_grad():
                policy_logits, values = self.network(states_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1)
                policy_np = policy_probs.cpu().numpy()
                values_np = torch.tanh(values).squeeze(1).cpu().numpy()
            
            results = []
            for i in range(batch_size):
                num_actions = num_legal_actions[i]
                if num_actions == 0:
                    results.append(([], 0.0))
                else:
                    policy = policy_np[i, :num_actions].tolist()
                    value = float(values_np[i])
                    results.append((policy, value))
            
            return results
        
        self.batched_evaluator = BatchedEvaluator(batch_config, batch_callback)
        self.batched_evaluator.start()
        self.nn_evaluator = make_batched_nn_evaluator(self.batched_evaluator)
    
    def select_action(self, state: GameState):
        """
        Select best action for current game state using AlphaZero MCTS.
        
        Args:
            state: Current game state
            
        Returns:
            Action object (best move according to MCTS)
        """
        if self.batched_evaluator is None:
            raise RuntimeError("Batched evaluator not initialized. Create agent with enable_batching=True")
        
        mcts = AlphaZeroMCTS(self.mcts_config, self.nn_evaluator, self.batched_evaluator)
        return mcts.search(state)
    
    def cleanup(self):
        """Stop batched evaluator thread. Call when done using agent."""
        if self.batched_evaluator is not None:
            self.batched_evaluator.stop()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, mcts_simulations=100, device='cuda', 
                       add_noise=False):
        """
        Create AlphaZero agent from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            mcts_simulations: Number of MCTS simulations per move
            device: 'cuda' or 'cpu'
            add_noise: Add exploration noise (for training)
            
        Returns:
            AlphaZeroAgent instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create network
        encoder = StateEncoder()
        network = CatanNetwork(
            input_size=encoder.get_feature_size(),
            hidden_size=256,
            num_residual_blocks=3
        ).to(device)
        
        network.load_state_dict(checkpoint['network_state'])
        network.eval()
        
        return cls(network, mcts_simulations, device, enable_batching=True, add_noise=add_noise)


def create_alphazero_agent(checkpoint_path, simulations=100, device='cuda'):
    """
    Convenience function to create AlphaZero agent from checkpoint.
    
    Args:
        checkpoint_path: Path to trained model
        simulations: MCTS simulations per move (default 100)
        device: 'cuda' or 'cpu'
        
    Returns:
        AlphaZeroAgent ready to play
        
    Example:
        agent = create_alphazero_agent('checkpoints/final_model.pt')
        action = agent.select_action(game_state)
        agent.cleanup()  # When done
    """
    return AlphaZeroAgent.from_checkpoint(checkpoint_path, simulations, device)
