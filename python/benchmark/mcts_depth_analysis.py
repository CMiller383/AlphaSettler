"""Comprehensive MCTS depth analysis for performance/quality tradeoff.

Tests different MCTS simulation counts to find optimal balance between:
- Speed (time per game)
- Quality (game convergence, decision quality)
- Batching efficiency
- Scalability to 1M games
"""

import time
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from catan_engine import AlphaZeroConfig, BatchedEvaluatorConfig
from state_encoder import StateEncoder
from catan_network import CatanNetwork
from config import SmallTrainingConfig
from parallel_selfplay import ParallelSelfPlayEngine


def benchmark_mcts_depth(
    mcts_simulations: int,
    num_games: int = 10,
    num_workers: int = 12,
    device: str = "cuda"
) -> Dict:
    """Benchmark a specific MCTS depth configuration."""
    
    # Setup
    config = SmallTrainingConfig()
    encoder = StateEncoder()
    
    network = CatanNetwork(
        input_size=encoder.get_feature_size(),
        hidden_size=config.hidden_size,
        num_residual_blocks=config.num_residual_blocks,
    ).to(device)
    network.eval()
    
    # MCTS config
    mcts_config = AlphaZeroConfig()
    mcts_config.num_simulations = mcts_simulations
    mcts_config.cpuct = config.cpuct
    mcts_config.dirichlet_alpha = config.dirichlet_alpha
    mcts_config.dirichlet_weight = config.dirichlet_weight
    mcts_config.add_exploration_noise = True
    mcts_config.random_seed = 42
    
    # Batch config
    batch_config = BatchedEvaluatorConfig()
    batch_config.max_batch_size = 64
    batch_config.min_batch_size = 1
    batch_config.timeout_ms = 5
    batch_config.enable_batching = True
    
    # Create engine
    engine = ParallelSelfPlayEngine(
        network=network,
        config=mcts_config,
        batch_config=batch_config,
        num_workers=num_workers,
        device=device
    )
    
    # Run games
    start_time = time.perf_counter()
    results = engine.play_games(
        num_games=num_games,
        num_players=config.num_players,
        seed_offset=1000
    )
    elapsed = time.perf_counter() - start_time
    
    # Get batch stats from the batched evaluator
    total_requests = engine.batched_evaluator.get_total_requests()
    total_batches = engine.batched_evaluator.get_total_batches()
    avg_batch_size = engine.batched_evaluator.get_average_batch_size()
    
    # Analyze results
    total_turns = sum(r['turns'] for r in results)
    avg_turns = total_turns / len(results)
    turns_per_sec = total_turns / elapsed
    games_per_hour = 3600 * num_games / elapsed
    
    # Estimate for 1M games
    time_for_1M = (elapsed / num_games) * 1_000_000
    days_for_1M = time_for_1M / (3600 * 24)
    
    return {
        'mcts_sims': mcts_simulations,
        'num_games': num_games,
        'num_workers': num_workers,
        'elapsed_sec': elapsed,
        'time_per_game': elapsed / num_games,
        'games_per_hour': games_per_hour,
        'avg_turns': avg_turns,
        'total_turns': total_turns,
        'turns_per_sec': turns_per_sec,
        'nn_requests': total_requests,
        'nn_requests_per_game': total_requests / num_games,
        'avg_batch_size': avg_batch_size,
        'days_for_1M_games': days_for_1M,
        'hours_for_1M_games': time_for_1M / 3600,
    }


def main():
    print("=" * 80)
    print("MCTS Depth Analysis - Finding Optimal Speed/Quality Tradeoff")
    print("=" * 80)
    print()
    
    # Test configurations
    # Key insight: AlphaZero paper used 800 sims for training, but practical implementations
    # often use 100-400 for speed. We need to find the sweet spot.
    mcts_depths = [25, 50, 100, 200]
    
    # Configuration
    num_games = 10  # Small sample for quick iteration
    num_workers = 12  # Use most cores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Games per test: {num_games}")
    print(f"  Workers: {num_workers}")
    print(f"  MCTS depths to test: {mcts_depths}")
    print()
    print(f"Goal: Find configuration where 1M games takes <240 hours (10 days)")
    print(f"      at reasonable quality for self-play improvement.")
    print()
    print("=" * 80)
    print()
    
    results = []
    
    for depth in mcts_depths:
        print(f"\n{'='*80}")
        print(f"Testing MCTS Depth: {depth} simulations")
        print(f"{'='*80}")
        
        result = benchmark_mcts_depth(
            mcts_simulations=depth,
            num_games=num_games,
            num_workers=num_workers,
            device=device
        )
        results.append(result)
        
        print(f"\nResults for {depth} MCTS simulations:")
        print(f"  Time per game:        {result['time_per_game']:.2f} sec")
        print(f"  Games per hour:       {result['games_per_hour']:.1f}")
        print(f"  Average turns:        {result['avg_turns']:.1f}")
        print(f"  Turns per second:     {result['turns_per_sec']:.1f}")
        print(f"  NN requests/game:     {result['nn_requests_per_game']:.0f}")
        print(f"  Avg batch size:       {result['avg_batch_size']:.2f}")
        print(f"  Time for 1M games:    {result['days_for_1M_games']:.1f} days")
        print()
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'MCTS Sims':<12} {'Sec/Game':<12} {'Games/Hr':<12} {'Batch Size':<12} {'Days/1M':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['mcts_sims']:<12} {r['time_per_game']:<12.2f} {r['games_per_hour']:<12.1f} "
              f"{r['avg_batch_size']:<12.2f} {r['days_for_1M_games']:<12.1f}")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find fastest that's still under 10 days for 1M games
    viable = [r for r in results if r['days_for_1M_games'] < 10]
    
    if viable:
        best = max(viable, key=lambda x: x['mcts_sims'])  # Highest quality in viable range
        print(f"\nRecommended configuration: {best['mcts_sims']} MCTS simulations")
        print(f"  Achieves 1M games in {best['days_for_1M_games']:.1f} days")
        print(f"  {best['time_per_game']:.2f} seconds per game")
        print(f"  {best['games_per_hour']:.1f} games per hour")
        print(f"  Batch efficiency: {best['avg_batch_size']:.2f} average batch size")
    else:
        print("\nWARNING: None of the tested configurations achieve 1M games in <10 days")
        fastest = min(results, key=lambda x: x['days_for_1M_games'])
        print(f"Fastest option: {fastest['mcts_sims']} sims = {fastest['days_for_1M_games']:.1f} days")
    
    print()
    
    # Performance scaling analysis
    if len(results) >= 2:
        print("=" * 80)
        print("SCALING ANALYSIS")
        print("=" * 80)
        print()
        
        baseline = results[0]
        for r in results[1:]:
            sim_ratio = r['mcts_sims'] / baseline['mcts_sims']
            time_ratio = r['time_per_game'] / baseline['time_per_game']
            efficiency = (sim_ratio / time_ratio) * 100
            
            print(f"{baseline['mcts_sims']} → {r['mcts_sims']} sims ({sim_ratio:.1f}x increase):")
            print(f"  Time increased by {time_ratio:.2f}x")
            print(f"  Scaling efficiency: {efficiency:.1f}%")
            print(f"  (100% = perfectly linear scaling)")
            print()


if __name__ == "__main__":
    main()
