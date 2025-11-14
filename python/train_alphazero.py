"""
AlphaZero training loop for Catan.
Self-play → collect training data → train network → evaluate → repeat.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import pickle
import os
from tqdm import tqdm

from catan_engine import (
    GameState, AlphaZeroMCTS, AlphaZeroConfig, NNEvaluation,
    generate_legal_actions, apply_action
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import TrainingConfig, SmallTrainingConfig, QuickTestConfig


class AlphaZeroTrainer:
    """
    Manages the AlphaZero training loop for Catan.
    
    Training cycle:
    1. Self-play: Generate games using current network
    2. Collect: Store (state, policy, value) tuples
    3. Train: Update network on collected data
    4. Evaluate: Test against baselines
    5. Checkpoint: Save best models
    """
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: TrainingConfig object with hyperparameters
        """
        self.config = config
        
        # Initialize components
        self.encoder = StateEncoder()
        self.network = CatanNetwork(
            input_size=self.encoder.get_feature_size(),
            hidden_size=config.hidden_size,
            num_residual_blocks=config.num_residual_blocks
        ).to(config.device)
        
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Statistics
        self.training_step = 0
        self.self_play_games = 0
        
    def nn_evaluator(self, state: GameState):
        """
        Neural network evaluator for AlphaZero MCTS.
        Wraps the NN to match the C++ callback signature.
        """
        # Encode state
        player = state.current_player
        features = self.encoder.encode_state(state, player)

        # Get legal actions and map to local indices 0..N-1
        legal_actions = generate_legal_actions(state)
        if not legal_actions:
            result = NNEvaluation()
            result.policy = []
            result.value = 0.0
            return result

        legal_indices = list(range(len(legal_actions)))

        # NN prediction in full action space, then restrict to legal actions
        policy_probs, value = self.network.predict(features, legal_indices)
        
        # Create result
        result = NNEvaluation()
        result.policy = policy_probs.tolist()
        result.value = float(value)
        
        return result
    
    def self_play_game(self, mcts_config: AlphaZeroConfig):
        """
        Play one self-play game and collect training data.
        
        Returns:
            List of (state_features, policy_target, value_target) tuples
        """
        # Create new game
        state = GameState.create_new_game(self.config.num_players, np.random.randint(0, 1000000))
        
        # Training examples from this game
        training_examples = []
        
        # Play until game over
        turn_count = 0
        max_turns = 500
        
        while not state.is_game_over() and turn_count < max_turns:
            player = state.current_player
            
            # Encode current state
            features = self.encoder.encode_state(state, player)
            
            # Get legal actions
            legal_actions = generate_legal_actions(state)
            if len(legal_actions) == 0:
                break
            
            # Run MCTS to get policy (returns best Action directly)
            mcts = AlphaZeroMCTS(mcts_config, self.nn_evaluator)
            selected_action = mcts.search(state)

            # Get action visit-count probabilities aligned with root legal actions
            # C++ binding exposes get_action_probabilities(self) -> list[float]
            # of length len(root.legal_actions).
            action_probs_full = mcts.get_action_probabilities()

            # Local indices 0..N-1 correspond directly to legal_actions order
            num_legal = len(legal_actions)
            if len(action_probs_full) != num_legal:
                # Safety check: resize or renormalize if needed
                action_probs_full = action_probs_full[:num_legal]
            policy_for_legal = np.array(action_probs_full, dtype=np.float32)
            legal_indices_arr = np.arange(num_legal, dtype=np.int32)

            # Store training example (we'll set value after game ends)
            training_examples.append({
                'state': features,
                'policy': policy_for_legal,
                'legal_indices': legal_indices_arr,
                'player': player
            })
            
            # Apply action chosen by MCTS
            apply_action(state, selected_action, np.random.randint(0, 1000000))
            
            turn_count += 1
        
        # Assign values based on game outcome
        if state.is_game_over():
            winner = state.get_winner()
            for example in training_examples:
                # Value is 1 for winner, 0 for losers (from each player's perspective)
                example['value'] = 1.0 if example['player'] == winner else 0.0
        else:
            # Game didn't finish - use current VP as proxy
            for example in training_examples:
                player_vp = state.players[example['player']].total_victory_points()
                example['value'] = player_vp / 10.0  # Normalize by win condition
        
        return training_examples
    
    def generate_self_play_data(self, num_games):
        """Generate self-play games and collect training data."""
        print(f"Generating {num_games} self-play games...")
        
        mcts_config = AlphaZeroConfig()
        mcts_config.num_simulations = self.config.mcts_simulations
        mcts_config.cpuct = self.config.cpuct
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_weight = self.config.dirichlet_weight
        
        for game_idx in tqdm(range(num_games)):
            examples = self.self_play_game(mcts_config)
            self.replay_buffer.extend(examples)
            self.self_play_games += 1
    
    def train_step(self, batch_size):
        """
        Perform one training step on a batch from replay buffer.
        
        Returns:
            Dictionary with loss statistics
        """
        if len(self.replay_buffer) < batch_size:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        # Sample batch
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Prepare tensors
        states = torch.stack([torch.from_numpy(ex['state']) for ex in batch]).float().to(self.config.device)
        
        # Policy targets: full local action space up to the network's
        # max_action_space, using exact stored legal indices (0..N-1).
        action_space_size = self.network.max_action_space
        policy_targets = []
        for ex in batch:
            target = np.zeros(action_space_size, dtype=np.float32)
            probs = ex['policy']          # shape (num_legal,)
            indices = ex['legal_indices'] # shape (num_legal,)
            for idx, p in zip(indices, probs):
                if 0 <= idx < action_space_size:
                    target[idx] = p
            policy_targets.append(target)
        policy_targets = torch.from_numpy(np.stack(policy_targets, axis=0)).float().to(self.config.device)
        
        value_targets = torch.tensor([ex['value'] for ex in batch]).float().unsqueeze(1).to(self.config.device)
        
        # Forward pass
        self.network.train()
        policy_logits, values = self.network(states)
        
        # Compute losses
        policy_loss = F.cross_entropy(
            policy_logits, 
            policy_targets,
            reduction='mean'
        )
        
        value_loss = F.mse_loss(values, value_targets)
        
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_on_buffer(self, num_epochs, batch_size):
        """Train network on current replay buffer."""
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            losses = []
            
            # Multiple passes through data
            num_batches = max(1, len(self.replay_buffer) // batch_size)
            
            for _ in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss_dict = self.train_step(batch_size)
                losses.append(loss_dict)
            
            # Print average losses
            avg_policy_loss = np.mean([l['policy_loss'] for l in losses])
            avg_value_loss = np.mean([l['value_loss'] for l in losses])
            avg_total_loss = np.mean([l['total_loss'] for l in losses])
            
            print(f"Epoch {epoch+1}: Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}, Total={avg_total_loss:.4f}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'self_play_games': self.self_play_games,
            'replay_buffer': list(self.replay_buffer)
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_step = checkpoint['training_step']
        self.self_play_games = checkpoint['self_play_games']
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.config.replay_buffer_size)
        print(f"Loaded checkpoint from {path}")


def main():
    """Example training run."""
    # Choose configuration preset
    # config = QuickTestConfig()     # ~15 games for quick testing
    config = SmallTrainingConfig()   # ~1000 games for initial trainin/g
    # config = MediumTrainingConfig()  # ~10k games for longer experiments
    # config = LargeTrainingConfig()   # ~100k games for full training
    
    # Print configuration
    config.summary()
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = AlphaZeroTrainer(config)
    
    # Training loop
    for iteration in range(config.num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{config.num_iterations} ===")
        
        # Self-play
        trainer.generate_self_play_data(config.games_per_iteration)
        
        # Train
        trainer.train_on_buffer(config.training_epochs_per_iteration, config.batch_size)
        
        # Save checkpoint
        if (iteration + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_iter_{iteration+1}.pt')
            trainer.save_checkpoint(checkpoint_path)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
