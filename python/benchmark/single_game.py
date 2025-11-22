"""Single-game performance benchmark for Catan AlphaZero.

Runs one self-play game with AlphaZero MCTS and reports timing
for key components: state encoding, legal action generation, NN
inference, MCTS search, and C++ state transitions.
"""

import time
from collections import defaultdict

import sys
from pathlib import Path

# Add parent directory to path so we can import from python package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from catan_engine import (
    GameState,
    AlphaZeroMCTS,
    AlphaZeroConfig,
    NNEvaluation,
    generate_legal_actions,
    apply_action,
)
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import SmallTrainingConfig


class TimedAlphaZeroRunner:
    """Run a single AlphaZero self-play game with detailed timing."""

    def __init__(self, config: SmallTrainingConfig):
        self.config = config

        # Core components
        self.encoder = StateEncoder()
        self.network = CatanNetwork(
            input_size=self.encoder.get_feature_size(),
            hidden_size=config.hidden_size,
            num_residual_blocks=config.num_residual_blocks,
        ).to(config.device)

        # Timing accumulators
        self.timers = defaultdict(float)
        self.nn_calls = 0
        self.mcts_calls = 0
        self.actions_applied = 0

    def nn_evaluator(self, state: GameState) -> NNEvaluation:
        """Neural network evaluator with timing instrumentation."""
        t0 = time.perf_counter()
        player = state.current_player

        # State encoding
        enc_start = time.perf_counter()
        features = self.encoder.encode_state(state, player)
        features = np.array(features, dtype=np.float32)  # Ensure proper type
        self.timers["state_encoding"] += time.perf_counter() - enc_start

        # Legal actions
        legal_start = time.perf_counter()
        legal_actions = generate_legal_actions(state)
        self.timers["legal_generation"] += time.perf_counter() - legal_start

        result = NNEvaluation()

        if not legal_actions:
            result.policy = []
            result.value = 0.0
            return result

        legal_indices = list(range(len(legal_actions)))

        # NN forward
        nn_start = time.perf_counter()
        policy_probs, value = self.network.predict(features, legal_indices)
        self.timers["nn_forward"] += time.perf_counter() - nn_start
        self.nn_calls += 1

        result.policy = policy_probs.tolist()
        result.value = float(value)

        self.timers["nn_evaluator_total"] += time.perf_counter() - t0
        return result

    def play_single_game(self, num_players: int = 4, seed: int = 0) -> dict:
        """Play exactly one self-play game and collect timing stats."""
        # Reset timers
        self.timers.clear()
        self.nn_calls = 0
        self.mcts_calls = 0
        self.actions_applied = 0

        # Game setup
        game_start = time.perf_counter()
        state = GameState.create_new_game(num_players, seed)
        self.timers["game_init"] += time.perf_counter() - game_start

        # MCTS config (mirror training config)
        mcts_config = AlphaZeroConfig()
        mcts_config.num_simulations = self.config.mcts_simulations
        mcts_config.cpuct = self.config.cpuct
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_weight = self.config.dirichlet_weight
        mcts_config.add_exploration_noise = True  # Enable exploration noise
        mcts_config.random_seed = seed  # Use game seed for reproducibility
        
        # Virtual loss for parallel tree descent
        mcts_config.num_parallel_sims = self.config.mcts_simulations
        mcts_config.virtual_loss_penalty = 1.0

        turn_count = 0
        max_turns = 500

        loop_start = time.perf_counter()
        while not state.is_game_over() and turn_count < max_turns:
            # Generate legal actions once per turn
            legal_actions = generate_legal_actions(state)
            if not legal_actions:
                break

            # MCTS search
            mcts_search_start = time.perf_counter()
            mcts = AlphaZeroMCTS(mcts_config, self.nn_evaluator)
            selected_action = mcts.search(state)
            self.timers["mcts_search"] += time.perf_counter() - mcts_search_start
            self.mcts_calls += 1

            # Apply chosen action (C++ state transition)
            apply_start = time.perf_counter()
            apply_action(state, selected_action, np.random.randint(0, 1_000_000))
            self.timers["apply_action"] += time.perf_counter() - apply_start
            self.actions_applied += 1

            turn_count += 1

        self.timers["game_loop_total"] += time.perf_counter() - loop_start
        self.timers["game_total_wall"] += time.perf_counter() - game_start

        winner = state.get_winner() if state.is_game_over() else None

        return {
            "turns": turn_count,
            "winner": winner,
            "timers": dict(self.timers),
            "nn_calls": self.nn_calls,
            "mcts_calls": self.mcts_calls,
            "actions_applied": self.actions_applied,
        }


def format_seconds(s: float) -> str:
    return f"{s * 1000.0:.2f} ms"


def main() -> None:
    # Use SmallTrainingConfig so the benchmark matches your training setup.
    config = SmallTrainingConfig()
    config.summary()

    runner = TimedAlphaZeroRunner(config)
    stats = runner.play_single_game(num_players=config.num_players, seed=0)

    print("\n=== Single-Game Performance Benchmark ===")
    print(f"Turns played:        {stats['turns']}")
    print(f"Winner:              {stats['winner']}")
    print(f"NN eval calls:       {stats['nn_calls']}")
    print(f"MCTS search calls:   {stats['mcts_calls']}")
    print(f"Actions applied:     {stats['actions_applied']}")

    timers = stats["timers"]
    print("\nTiming breakdown (wall-clock):")
    for key in sorted(timers.keys()):
        print(f"  {key:20s}: {format_seconds(timers[key])}")

    # Per-call averages
    if stats["nn_calls"] > 0 and "nn_forward" in timers:
        avg_nn = timers["nn_forward"] / stats["nn_calls"]
        print(f"\nAverage NN forward per call:    {format_seconds(avg_nn)}")
    if stats["mcts_calls"] > 0 and "mcts_search" in timers:
        avg_mcts = timers["mcts_search"] / stats["mcts_calls"]
        print(f"Average MCTS search per move:  {format_seconds(avg_mcts)}")


if __name__ == "__main__":
    main()
