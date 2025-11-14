"""Benchmark parallel self-play performance."""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from catan_engine import AlphaZeroConfig, BatchedEvaluatorConfig
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import SmallTrainingConfig
from parallel_selfplay import ParallelSelfPlayEngine


def benchmark_parallel_selfplay(num_games: int = 8, num_workers: int = 8):
    """Benchmark parallel self-play with different worker counts."""
    
    print("="*70)
    print(f"Parallel Self-Play Benchmark")
    print("="*70)
    
    # Setup
    config = SmallTrainingConfig()
    encoder = StateEncoder()
    
    network = CatanNetwork(
        input_size=encoder.get_feature_size(),
        hidden_size=config.hidden_size,
        num_residual_blocks=config.num_residual_blocks,
    ).to(config.device)
    
    # MCTS config
    mcts_config = AlphaZeroConfig()
    mcts_config.num_simulations = config.mcts_simulations
    mcts_config.cpuct = config.cpuct
    mcts_config.dirichlet_alpha = config.dirichlet_alpha
    mcts_config.dirichlet_weight = config.dirichlet_weight
    mcts_config.add_exploration_noise = True
    mcts_config.random_seed = 0
    
    # Batch config
    batch_config = BatchedEvaluatorConfig()
    batch_config.max_batch_size = 32
    batch_config.min_batch_size = 1
    batch_config.timeout_ms = 5
    batch_config.enable_batching = True
    
    print(f"\nConfiguration:")
    print(f"  Games: {num_games}")
    print(f"  Workers: {num_workers}")
    print(f"  MCTS sims: {mcts_config.num_simulations}")
    print(f"  Batch size: {batch_config.min_batch_size}-{batch_config.max_batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Network: {config.hidden_size} hidden, {config.num_residual_blocks} blocks")
    
    # Create engine
    engine = ParallelSelfPlayEngine(
        network=network,
        config=mcts_config,
        batch_config=batch_config,
        num_workers=num_workers,
        device=config.device
    )
    
    # Warmup
    print("\nWarmup (1 game)...")
    engine.play_games(num_games=1, num_players=config.num_players, seed_offset=0)
    
    # Benchmark
    print(f"\nBenchmarking {num_games} games with {num_workers} workers...")
    start_time = time.perf_counter()
    
    results = engine.play_games(
        num_games=num_games,
        num_players=config.num_players,
        seed_offset=1000
    )
    
    elapsed = time.perf_counter() - start_time
    
    # Analyze results
    total_turns = sum(r['turns'] for r in results)
    avg_turns = total_turns / len(results)
    
    print(f"\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Games completed: {len(results)}")
    print(f"  Time per game: {elapsed / len(results):.2f} seconds")
    print(f"  Games per hour: {3600 * len(results) / elapsed:.1f}")
    print(f"  Average turns: {avg_turns:.1f}")
    print(f"  Total turns: {total_turns}")
    print("="*70)
    
    # Compare to sequential baseline (from previous benchmark)
    sequential_time_per_game = 6.4  # seconds (from single_game.py)
    sequential_total = sequential_time_per_game * num_games
    speedup = sequential_total / elapsed
    
    print(f"\nSpeedup vs Sequential:")
    print(f"  Sequential (estimated): {sequential_total:.2f} seconds")
    print(f"  Parallel: {elapsed:.2f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {speedup / num_workers * 100:.1f}%")
    
    return results, elapsed


def main():
    # Test different worker counts
    for num_workers in [1, 2, 4, 8]:
        print(f"\n{'='*70}")
        print(f"Testing with {num_workers} workers")
        print(f"{'='*70}")
        
        benchmark_parallel_selfplay(num_games=8, num_workers=num_workers)
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Quick single test
    benchmark_parallel_selfplay(num_games=8, num_workers=8)
