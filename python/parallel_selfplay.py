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
    
    def _batch_evaluate_callback(self, batch: List[Tuple]) -> List[Tuple]:
        """
        Callback for batched NN evaluation.
        Called by C++ BatchedEvaluator with a batch of encoded states.
        
        Args:
            batch: List of (encoded_state_numpy, num_legal_actions) tuples
        
        Returns:
            List of (policy_list, value_float) tuples
        """
        if not batch:
            return []
        
        # Convert to torch tensor
        batch_size = len(batch)
        encoded_states = [item[0] for item in batch]
        num_actions_list = [item[1] for item in batch]
        
        # Stack into batch tensor
        states_tensor = torch.tensor(
            np.array(encoded_states),
            dtype=torch.float32,
            device=self.device
        )
        
        # Run batch inference
        with torch.no_grad():
            policy_logits, values = self.network(states_tensor)
        
        # Convert to list of results
        results = []
        for i in range(batch_size):
            num_actions = num_actions_list[i]
            
            if num_actions == 0:
                results.append(([], 0.0))
                continue
            
            # Get policy for this state's legal actions
            logits = policy_logits[i, :num_actions]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            # Get value
            value = torch.tanh(values[i]).item()
            
            results.append((probs.tolist(), value))
        
        return results
    
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
        nn_evaluator = make_batched_nn_evaluator(self.batched_evaluator)
        mcts = AlphaZeroMCTS(self.mcts_config, nn_evaluator)
        
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
