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
import json
import time
from datetime import datetime
from tqdm import tqdm

from catan_engine import (
    GameState, AlphaZeroMCTS, AlphaZeroConfig, NNEvaluation,
    generate_legal_actions, apply_action, BatchedEvaluatorConfig,
    ActionType
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import (
    TrainingConfig, QuickTestConfig, SmallTrainingConfig, 
    MediumTrainingConfig, LargeTrainingConfig, 
    PACEGPUConfig, H100Config, H200Config
)
from parallel_selfplay import ParallelSelfPlayEngine


# ============================================================================
# TRAINING CONFIGURATION - CHANGE THIS TO SELECT DIFFERENT CONFIGS
# ============================================================================

# Uncomment ONE of the following configurations:

# TRAINING_CONFIG = QuickTestConfig()      # 15 games - quick test (5-10 min)
# TRAINING_CONFIG = SmallTrainingConfig()   # 500 games - initial training (~2-4 hours)
# TRAINING_CONFIG = MediumTrainingConfig()  # 10k games - serious training (~10-20 hours)
# TRAINING_CONFIG = LargeTrainingConfig()   # 100k games - full training (~100+ hours CPU)

# For PACE GPU Cluster:
# TRAINING_CONFIG = PACEGPUConfig()        # A100 optimized (100k games, ~40-80 hours)
# TRAINING_CONFIG = H100Config()           # H100 optimized (ready for full training)
# TRAINING_CONFIG = H200Config()           # H200 optimized (100k games, ~20-40 hours)

# For testing:
TRAINING_CONFIG = QuickTestConfig()      # 1000 games benchmark test

# ============================================================================


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
        
        # Training history for logging
        self.training_history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'games_played': [],
            'replay_buffer_size': [],
            'timestamps': [],
            'self_play_times': [],
            'training_times': []
        }
        
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
        
        # HEURISTIC FIX: Adjust policy to encourage building, discourage EndTurn
        # This helps early training when network outputs are nearly uniform
        policy_probs = self._adjust_policy_heuristic(policy_probs, legal_actions)
        
        # Create result
        result = NNEvaluation()
        result.policy = policy_probs.tolist()
        result.value = float(value)
        
        return result
    
    def _adjust_policy_heuristic(self, policy, actions):
        """
        Apply heuristic adjustments to policy to break deadlocks.
        
        Strongly encourages building actions, discourages EndTurn when buildings available.
        This is critical for early training when the network is essentially random.
        """
        import numpy as np
        adjusted = policy.copy()
        
        # Check if any building actions are available
        has_buildings = any(a.type in [ActionType.PlaceSettlement, ActionType.PlaceRoad, 
                                       ActionType.UpgradeToCity] for a in actions)
        
        for i, action in enumerate(actions):
            if action.type == ActionType.EndTurn:
                # Discourage EndTurn, especially when buildings available
                if has_buildings:
                    adjusted[i] *= 0.05  # Strong suppression
                else:
                    adjusted[i] *= 0.3  # Mild suppression
            elif action.type in [ActionType.PlaceSettlement, ActionType.PlaceRoad, 
                                ActionType.UpgradeToCity]:
                # Strongly encourage building
                adjusted[i] *= 10.0
            elif action.type in [ActionType.BankTrade, ActionType.PortTrade]:
                # Mildly encourage trading
                adjusted[i] *= 2.0
        
        # Renormalize
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
        else:
            # Fallback to uniform if something went wrong
            adjusted = np.ones_like(policy) / len(policy)
        
        return adjusted
    
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
        """Generate self-play games using parallel batched evaluation."""
        print(f"Generating {num_games} self-play games with parallel batched evaluation...")
        
        # MCTS config
        mcts_config = AlphaZeroConfig()
        mcts_config.num_simulations = self.config.mcts_simulations
        mcts_config.cpuct = self.config.cpuct
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_weight = self.config.dirichlet_weight
        mcts_config.add_exploration_noise = True
        
        # Virtual loss settings for parallel tree descent (GPU optimization)
        mcts_config.num_parallel_sims = self.config.mcts_simulations  # Run all sims in parallel
        mcts_config.virtual_loss_penalty = 1.0  # Standard virtual loss penalty
        
        # Batched evaluator config  
        batch_config = BatchedEvaluatorConfig()
        batch_config.max_batch_size = self.config.max_batch_size
        batch_config.min_batch_size = 1
        batch_config.timeout_ms = 10  # 10ms optimal timeout for batching
        batch_config.enable_batching = True  # Re-enable batching

        
        # Determine number of workers (parallel games)
        if self.config.selfplay_workers is not None:
            num_workers = min(num_games, self.config.selfplay_workers)
        else:
            # Auto-detect: use 2x CPU cores since MCTS threads wait on NN
            import multiprocessing
            num_workers = min(num_games, multiprocessing.cpu_count() * 2, 16)
        
        # Create parallel self-play engine
        engine = ParallelSelfPlayEngine(
            network=self.network,
            config=mcts_config,
            batch_config=batch_config,
            num_workers=num_workers,
            device=self.config.device
        )
        
        # Run parallel self-play
        results = engine.play_games(
            num_games=num_games,
            num_players=self.config.num_players,
            seed_offset=self.self_play_games
        )
        
        # Convert results to training examples
        total_examples = 0
        completed_games = sum(1 for r in results if r['winner'] is not None)
        total_turns = [r['turns'] for r in results]
        avg_turns = sum(total_turns) / len(total_turns) if total_turns else 0
        
        print(f"  Game completion: {completed_games}/{len(results)} ({100*completed_games/len(results):.1f}%)")
        print(f"  Avg turns/game: {avg_turns:.1f}")
        
        for game_result in results:
            # Convert each game to training examples
            states = game_result['states']
            policies = game_result['policies']
            winner = game_result['winner']
            
            # CRITICAL FIX: Only train on completed games!
            # Training on incomplete games teaches the network that stuck states are normal
            if winner is None:
                # Game didn't finish - discard these examples
                self.self_play_games += 1
                continue
            
            for i, (state, policy) in enumerate(zip(states, policies)):
                player = state.current_player
                
                # Encode state
                features = self.encoder.encode_state(state, player)
                
                # Convert to numpy array immediately to ensure proper type
                features = np.array(features, dtype=np.float32)
                
                # Sanity check for NaN/Inf during encoding
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"WARNING: NaN/Inf detected during state encoding at game {i}!")
                    print(f"  Player: {player}, Turn: {state.turn_number}")
                    continue  # Skip this corrupted example
                
                # Get legal actions for this state
                legal_actions = generate_legal_actions(state)
                num_legal = len(legal_actions)
                
                # Policy is already in correct format from MCTS
                policy_arr = np.array(policy[:num_legal], dtype=np.float32)
                legal_indices = np.arange(num_legal, dtype=np.int32)
                
                # Value: 1 for winner, 0 for losers (only completed games reach here)
                value = 1.0 if player == winner else 0.0
                
                # Add to replay buffer
                self.replay_buffer.append({
                    'state': features,
                    'policy': policy_arr,
                    'legal_indices': legal_indices,
                    'player': player,
                    'value': value
                })
                total_examples += 1
            
            self.self_play_games += 1
        
        # Print statistics
        print(f"  Parallel workers: {num_workers}")
        print(f"  Total games: {len(results)}")
        print(f"  Avg turns/game: {sum(r['turns'] for r in results) / len(results):.1f}")
        print(f"  Training examples: {total_examples} (from {completed_games} completed games)")
        
        if completed_games == 0:
            print(f"  ⚠️  WARNING: No completed games! Cannot train without successful game data.")
    
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
            
            # Only use indices within the network's action space
            for idx, p in zip(indices, probs):
                if 0 <= idx < action_space_size:
                    target[idx] = p
            
            # Renormalize in case some indices were out of bounds
            target_sum = target.sum()
            if target_sum > 0:
                target = target / target_sum
            else:
                # All indices were out of bounds - use uniform over first action
                target[0] = 1.0
            
            policy_targets.append(target)
        policy_targets = torch.from_numpy(np.stack(policy_targets, axis=0)).float().to(self.config.device)
        
        value_targets = torch.tensor([ex['value'] for ex in batch]).float().unsqueeze(1).to(self.config.device)
        
        # Forward pass
        self.network.train()
        policy_logits, values = self.network(states)
        
        # Compute losses
        # Policy loss: KL divergence between MCTS policy and network policy
        # Cross-entropy with soft targets (MCTS visit distribution)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        
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
        all_losses = []
        
        for epoch in range(num_epochs):
            losses = []
            
            # Multiple passes through data
            num_batches = max(1, len(self.replay_buffer) // batch_size)
            
            # Progress bar for batches
            batch_pbar = tqdm(range(num_batches), 
                            desc=f"    Epoch {epoch+1}/{num_epochs}", 
                            leave=False, 
                            ncols=100, 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for _ in batch_pbar:
                loss_dict = self.train_step(batch_size)
                losses.append(loss_dict)
                # Update progress bar with current losses
                batch_pbar.set_postfix({'p_loss': f'{loss_dict["policy_loss"]:.3f}', 
                                       'v_loss': f'{loss_dict["value_loss"]:.3f}'})
            
            all_losses.extend(losses)
        
        # Return average losses across all epochs
        return {
            'policy_loss': np.mean([l['policy_loss'] for l in all_losses]),
            'value_loss': np.mean([l['value_loss'] for l in all_losses]),
            'total_loss': np.mean([l['total_loss'] for l in all_losses])
        }
    
    def log_iteration(self, iteration, losses, self_play_time, training_time):
        """Log iteration metrics to history."""
        self.training_history['iterations'].append(iteration)
        self.training_history['policy_losses'].append(losses['policy_loss'])
        self.training_history['value_losses'].append(losses['value_loss'])
        self.training_history['total_losses'].append(losses['total_loss'])
        self.training_history['games_played'].append(self.self_play_games)
        self.training_history['replay_buffer_size'].append(len(self.replay_buffer))
        self.training_history['timestamps'].append(datetime.now().isoformat())
        self.training_history['self_play_times'].append(self_play_time)
        self.training_history['training_times'].append(training_time)
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'self_play_games': self.self_play_games,
            'replay_buffer': list(self.replay_buffer),
            'training_history': self.training_history
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
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        print(f"Loaded checkpoint from {path}")
    
    def save_training_log(self, run_dir):
        """Save training history to JSON (fast, machine-readable)."""
        log_path = os.path.join(run_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_history, f)
    
    def save_run_info(self, run_dir, start_time, end_time):
        """Save run metadata and config."""
        info = {
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': (end_time - start_time),
            'config': {
                'num_players': self.config.num_players,
                'mcts_simulations': self.config.mcts_simulations,
                'cpuct': self.config.cpuct,
                'hidden_size': self.config.hidden_size,
                'num_residual_blocks': self.config.num_residual_blocks,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'games_per_iteration': self.config.games_per_iteration,
                'num_iterations': self.config.num_iterations,
                'total_games': self.self_play_games,
                'device': str(self.config.device)
            },
            'final_stats': {
                'total_training_steps': self.training_step,
                'replay_buffer_size': len(self.replay_buffer),
                'final_policy_loss': self.training_history['policy_losses'][-1] if self.training_history['policy_losses'] else None,
                'final_value_loss': self.training_history['value_losses'][-1] if self.training_history['value_losses'] else None
            }
        }
        
        info_path = os.path.join(run_dir, 'run_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def plot_training_curves(self, run_dir):
        """Generate training curve plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            if not self.training_history['iterations']:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss curves
            axes[0, 0].plot(self.training_history['iterations'], self.training_history['policy_losses'], label='Policy Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Policy Loss Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(self.training_history['iterations'], self.training_history['value_losses'], label='Value Loss', color='orange')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Value Loss Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Games played
            axes[1, 0].plot(self.training_history['iterations'], self.training_history['games_played'], label='Total Games', color='green')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Games')
            axes[1, 0].set_title('Games Played Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Time per iteration
            total_times = [s + t for s, t in zip(self.training_history['self_play_times'], self.training_history['training_times'])]
            axes[1, 1].plot(self.training_history['iterations'], total_times, label='Time/Iter', color='red')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Time per Iteration')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(run_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved training plots to {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    """Example training run."""
    # Allow config selection via environment variable
    config_name = os.environ.get('ALPHASETTLER_CONFIG', 'TRAINING_CONFIG')
    
    # Map config names to actual config objects
    config_map = {
        'QuickTestConfig': QuickTestConfig(),
        'SmallTrainingConfig': SmallTrainingConfig(),
        'MediumTrainingConfig': MediumTrainingConfig(),
        'LargeTrainingConfig': LargeTrainingConfig(),
        'PACEGPUConfig': PACEGPUConfig(),
        'H100Config': H100Config(),
        'H200Config': H200Config(),
        'TRAINING_CONFIG': TRAINING_CONFIG  # Default from top of file
    }
    
    # Select config
    if config_name in config_map:
        config = config_map[config_name]
        if config_name != 'TRAINING_CONFIG':
            print(f"Using config from environment: {config_name}\n")
    else:
        print(f"Warning: Unknown config '{config_name}', using default TRAINING_CONFIG\n")
        config = TRAINING_CONFIG
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_run_dir = os.environ.get('ALPHASETTLER_RUN_DIR', 'training_runs')
    run_dir = os.path.join(base_run_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create checkpoint subdirectory
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.checkpoint_dir = checkpoint_dir
    
    # Print configuration
    config.summary()
    print(f"\nRun directory: {run_dir}")
    print(f"All outputs will be saved to this directory.\n")
    
    # Initialize trainer
    trainer = AlphaZeroTrainer(config)
    
    # Track overall timing
    run_start_time = time.time()
    
    # Print training header
    print(f"\n{'='*90}")
    print(f"{'  AlphaSettler Training':^90}")
    print(f"{'='*90}")
    print(f"  {config.num_iterations} iterations × {config.games_per_iteration} games = {config.num_iterations * config.games_per_iteration} total games")
    print(f"  Device: {config.device}")
    print(f"  Workers: {config.selfplay_workers if config.selfplay_workers else 'auto'}")
    print(f"  Batch size: {config.max_batch_size}")
    print(f"{'='*90}\n")
    
    # Training loop
    for iteration in range(config.num_iterations):
        iter_start = time.time()
        print(f"\n{'─'*90}")
        print(f"  Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'─'*90}")
        
        # Self-play
        print(f"\n[1/3] Self-play: Generating {config.games_per_iteration} games...")
        sp_start = time.time()
        trainer.generate_self_play_data(config.games_per_iteration)
        sp_time = time.time() - sp_start
        print(f"✓ Self-play completed in {sp_time:.1f}s")
        
        # Force cleanup to prevent segfaults between iterations
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all CUDA operations to finish
            torch.cuda.empty_cache()
        
        # Train
        print(f"\n[2/3] Training: Processing {len(trainer.replay_buffer)} examples for {config.training_epochs_per_iteration} epochs...")
        train_start = time.time()
        losses = trainer.train_on_buffer(config.training_epochs_per_iteration, config.batch_size)
        train_time = time.time() - train_start
        print(f"✓ Training completed in {train_time:.1f}s")
        
        # Log iteration (fast, in-memory)
        trainer.log_iteration(iteration + 1, losses, sp_time, train_time)
        
        # Print summary
        iter_time = time.time() - iter_start
        print(f"\n{'─'*90}")
        print(f"  ✓ Iteration {iteration+1} Complete")
        print(f"  │ Policy Loss: {losses['policy_loss']:.4f}  │  Value Loss: {losses['value_loss']:.4f}")
        print(f"  │ Time: {iter_time:.1f}s (Self-play: {sp_time:.1f}s | Training: {train_time:.1f}s)")
        print(f"  │ Replay Buffer: {len(trainer.replay_buffer):,} examples  │  Total Games: {trainer.self_play_games:,}")
        print(f"{'─'*90}")
        
        # Save checkpoint
        if (iteration + 1) % config.checkpoint_interval == 0:
            print(f"\n[3/3] Saving checkpoint...")
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration+1}.pt')
            trainer.save_checkpoint(checkpoint_path)
            
            # Save training log (lightweight JSON write)
            trainer.save_training_log(run_dir)
            print(f"✓ Checkpoint saved")
    
    run_end_time = time.time()
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Total time: {(run_end_time - run_start_time) / 3600:.2f} hours")
    print(f"Total games: {trainer.self_play_games}")
    print("="*70)
    
    # Save final outputs
    print("\nSaving final outputs...")
    final_checkpoint = os.path.join(checkpoint_dir, 'final_model.pt')
    trainer.save_checkpoint(final_checkpoint)
    trainer.save_training_log(run_dir)
    trainer.save_run_info(run_dir, run_start_time, run_end_time)
    trainer.plot_training_curves(run_dir)
    
    print(f"\nAll results saved to: {run_dir}")
    print("  - training_log.json (all metrics)")
    print("  - run_info.json (config and metadata)")
    print("  - training_curves.png (loss plots)")
    print(f"  - checkpoints/ ({config.num_iterations // config.checkpoint_interval} checkpoints)")


if __name__ == '__main__':
    main()
