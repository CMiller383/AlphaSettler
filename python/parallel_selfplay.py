"""
Parallel self-play engine with batched NN evaluation.
Runs multiple games concurrently with shared batched neural network inference.
"""

import threading
import time
from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from catan_engine import (
    GameState,
    AlphaZeroMCTS,
    AlphaZeroConfig,
    BatchedEvaluator,
    BatchedEvaluatorConfig,
    make_batched_nn_evaluator,
    generate_legal_actions,
    apply_action,
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork


class ParallelSelfPlayEngine:
    """
    Run multiple self-play games in parallel with batched NN evaluation.
    
    Architecture:
    - Multiple worker threads run MCTS games concurrently
    - All workers share a single BatchedEvaluator
    - NN evaluations are batched across all workers
    - Main thread handles batched inference on GPU
    """
    
    def __init__(
        self,
        network: CatanNetwork,
        config: AlphaZeroConfig,
        batch_config: BatchedEvaluatorConfig,
        num_workers: int = 8,
        device: str = 'cuda'
    ):
        """
        Initialize parallel self-play engine.
        
        Args:
            network: CatanNetwork for policy/value prediction
            config: AlphaZeroConfig for MCTS
            batch_config: BatchedEvaluatorConfig for batching
            num_workers: Number of parallel game workers
            device: Device for NN inference ('cuda' or 'cpu')
        """
        self.network = network.to(device)
        self.network.eval()  # Evaluation mode
        self.mcts_config = config
        self.batch_config = batch_config
        self.num_workers = num_workers
        self.device = device
        
        # State encoder for converting states to features
        self.encoder = StateEncoder()
        
        # Statistics
        self.stats = defaultdict(float)
        self.stats_lock = threading.Lock()
        
        # Batched evaluator (shared across all workers)
        self.batched_evaluator: Optional[BatchedEvaluator] = None
    
    def _batch_evaluate_callback(
        self, 
        stacked_states_flat: np.ndarray,
        num_actions_list: List[int],
        action_types_flat: np.ndarray,
        batch_size: int,
        feature_size: int
    ) -> List[Tuple]:
        """
        Callback for batched NN evaluation (ULTRA-OPTIMIZED with pre-stacked arrays).
        Called by C++ BatchedEvaluator with pre-stacked 2D array.
        
        Args:
            stacked_states_flat: Flat numpy array of shape (batch_size * feature_size,)
            num_actions_list: List of num_legal_actions for each state
            action_types_flat: Flat array of action types (uint8) for all actions in batch
            batch_size: Number of states in batch
            feature_size: Size of each encoded state
        
        Returns:
            List of (policy_list, value_float) tuples
        """
        if batch_size == 0:
            return []
        
        # OPTIMIZATION: Reshape flat array to 2D (zero-copy view)
        states_np = stacked_states_flat.reshape(batch_size, feature_size)
        
        # Convert to PyTorch tensor
        states_tensor = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        
        # Run batch inference
        with torch.no_grad():
            policy_logits, values = self.network(states_tensor)
            
            # OPTIMIZATION: Batch softmax on GPU
            policy_probs = torch.softmax(policy_logits, dim=1)
            
            # OPTIMIZATION: Single GPU→CPU transfer
            policy_np = policy_probs.cpu().numpy()
            values_np = torch.tanh(values).squeeze(1).cpu().numpy()
        
        # Apply heuristic adjustments to policy using action types
        policy_np = self._apply_heuristic_to_batch(
            policy_np, 
            num_actions_list, 
            action_types_flat
        )
        
        # OPTIMIZATION: Minimize per-state operations
        results = []
        for i in range(batch_size):
            num_actions = num_actions_list[i]
            if num_actions == 0:
                results.append(([], 0.0))
            else:
                policy = policy_np[i, :num_actions].tolist()
                value = float(values_np[i])
                results.append((policy, value))
        
        return results
    
    def _apply_heuristic_to_batch(
        self,
        policy_np: np.ndarray,
        num_actions_list: List[int],
        action_types_flat: np.ndarray
    ) -> np.ndarray:
        """
        Apply heuristic adjustments to batched policies.
        
        Strongly encourages building actions, discourages EndTurn.
        This is critical for early training when network is essentially random.
        
        Args:
            policy_np: Array of shape (batch_size, max_actions)
            num_actions_list: Number of legal actions for each state
            action_types_flat: Flat array of action types (ActionType enum values as uint8)
        
        Returns:
            Adjusted policy array (same shape)
        """
        from catan_engine import ActionType
        
        # Action type constants (from C++ enum)
        ENDTURN = int(ActionType.EndTurn)
        PLACE_INITIAL_SETTLEMENT = int(ActionType.PlaceInitialSettlement)
        PLACE_INITIAL_ROAD = int(ActionType.PlaceInitialRoad)
        PLACE_SETTLEMENT = int(ActionType.PlaceSettlement)
        PLACE_ROAD = int(ActionType.PlaceRoad)
        UPGRADE_CITY = int(ActionType.UpgradeToCity)
        BANK_TRADE = int(ActionType.BankTrade)
        PORT_TRADE = int(ActionType.PortTrade)
        
        adjusted = policy_np.copy()
        
        action_idx = 0  # Index into action_types_flat
        for batch_idx, num_actions in enumerate(num_actions_list):
            if num_actions == 0:
                continue
            
            # Extract action types for this state
            action_types = action_types_flat[action_idx:action_idx + num_actions]
            
            # Check if any building actions available
            has_buildings = np.any(
                (action_types == PLACE_SETTLEMENT) |
                (action_types == PLACE_ROAD) |
                (action_types == UPGRADE_CITY)
            )
            
            # Apply heuristic adjustments
            for i in range(num_actions):
                action_type = action_types[i]
                
                if action_type == ENDTURN:
                    # VERY strongly discourage EndTurn when buildings available
                    adjusted[batch_idx, i] *= 0.001 if has_buildings else 0.05
                elif action_type == PLACE_INITIAL_SETTLEMENT:
                    # CRITICAL: Flatten initial settlement probabilities
                    # This prevents agents from always picking first vertex (vertex 0, 1, 2, ...)
                    # Give all initial settlements equal strong weight to encourage exploration
                    adjusted[batch_idx, i] *= 1.0  # Keep uniform
                elif action_type == PLACE_INITIAL_ROAD:
                    # Initial roads: keep uniform (all adjacent roads equally valid)
                    adjusted[batch_idx, i] *= 1.0
                elif action_type in [PLACE_SETTLEMENT, PLACE_ROAD, UPGRADE_CITY]:
                    # VERY strongly encourage building
                    adjusted[batch_idx, i] *= 100.0
                elif action_type in [BANK_TRADE, PORT_TRADE]:
                    # Encourage trading
                    adjusted[batch_idx, i] *= 5.0
            
            # Renormalize this state's policy
            policy_sum = adjusted[batch_idx, :num_actions].sum()
            if policy_sum > 0:
                adjusted[batch_idx, :num_actions] /= policy_sum
            else:
                # Fallback to uniform if something went wrong
                adjusted[batch_idx, :num_actions] = 1.0 / num_actions
            
            action_idx += num_actions
        
        return adjusted
    
    def play_games(
        self,
        num_games: int,
        num_players: int = 4,
        seed_offset: int = 0
    ) -> List[dict]:
        """
        Play multiple games in parallel with batched evaluation.
        
        Args:
            num_games: Total number of games to play
            num_players: Players per game
            seed_offset: Offset for random seeds
        
        Returns:
            List of game results (one dict per game)
        """
        # Create batched evaluator
        self.batched_evaluator = BatchedEvaluator(
            self.batch_config,
            self._batch_evaluate_callback
        )
        self.batched_evaluator.start()
        
        # Create tasks (game seeds)
        game_seeds = list(range(seed_offset, seed_offset + num_games))
        
        # Results storage
        results = [None] * num_games
        results_lock = threading.Lock()
        
        # Progress bar
        pbar = tqdm(total=num_games, desc="  Self-play", unit=" game", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        # Worker function
        def worker(worker_id: int):
            while True:
                # Get next game seed
                with results_lock:
                    if not game_seeds:
                        break
                    game_idx = len(game_seeds) - 1
                    seed = game_seeds.pop()
                
                try:
                    # Play game
                    game_result = self._play_single_game(
                        seed=seed,
                        num_players=num_players,
                        worker_id=worker_id
                    )
                    
                    # Store result
                    with results_lock:
                        results[seed - seed_offset] = game_result
                        pbar.update(1)
                except Exception as e:
                    # Log error but don't crash other workers
                    with results_lock:
                        print(f"\nWorker {worker_id} error on game {seed}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Start workers
        workers = []
        for worker_id in range(self.num_workers):
            t = threading.Thread(target=worker, args=(worker_id,), daemon=False)
            t.start()
            workers.append(t)
        
        # Wait for completion
        for t in workers:
            t.join()
        
        pbar.close()
        
        # Stop batched evaluator
        self.batched_evaluator.stop()
        
        # Print statistics
        print(f"\nBatched Evaluation Statistics:")
        print(f"  Total requests: {self.batched_evaluator.get_total_requests()}")
        print(f"  Total batches: {self.batched_evaluator.get_total_batches()}")
        print(f"  Average batch size: {self.batched_evaluator.get_average_batch_size():.2f}")
        
        return results
    
    def _play_single_game(
        self,
        seed: int,
        num_players: int,
        worker_id: int
    ) -> dict:
        """
        Play a single self-play game using batched evaluation.
        
        Args:
            seed: Random seed for game
            num_players: Number of players
            worker_id: Worker thread ID (for debugging)
        
        Returns:
            Game result dict with states, actions, policies, winner
        """
        # Create game state
        state = GameState.create_new_game(num_players, seed)
        
        # Create MCTS with batched evaluator
        # Pass both the wrapper function (for backward compat) and the batched_evaluator for intra-game batching
        nn_evaluator = make_batched_nn_evaluator(self.batched_evaluator)
        mcts = AlphaZeroMCTS(self.mcts_config, nn_evaluator, self.batched_evaluator)
        
        # Game data for training
        states = []
        actions = []
        policies = []
        
        turn_count = 0
        max_turns = 500
        
        while not state.is_game_over() and turn_count < max_turns:
            # Get legal actions
            legal_actions = generate_legal_actions(state)
            if not legal_actions:
                break
            
            # Run MCTS
            selected_action = mcts.search(state)
            action_probs = mcts.get_action_probabilities()
            
            # Store training data
            states.append(state)
            actions.append(selected_action)
            policies.append(action_probs)
            
            # Apply action
            apply_action(state, selected_action, np.random.randint(0, 1_000_000))
            turn_count += 1
        
        winner = state.get_winner() if state.is_game_over() else None
        
        return {
            'states': states,
            'actions': actions,
            'policies': policies,
            'winner': winner,
            'turns': turn_count,
            'seed': seed,
            'worker_id': worker_id,
        }
