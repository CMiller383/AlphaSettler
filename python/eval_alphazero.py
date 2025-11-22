#!/usr/bin/env python3
"""
Evaluate trained AlphaZero model against baselines.
"""

import sys
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

from catan_engine import GameState, apply_action
from agents import AlphaZeroAgent, RandomAgent


def play_game(agents, num_players=4, seed=None, max_actions=10000):
    """Play a single game."""
    game = GameState.create_new_game(num_players, seed or int(time.time() * 1000))
    
    action_count = 0
    while not game.is_game_over() and action_count < max_actions:
        current_player = game.current_player
        agent = agents[current_player]
        
        action = agent.select_action(game)
        if action is None:
            break
        
        apply_action(game, action, action_count * 7919)
        action_count += 1
    
    winner = game.get_winner() if game.is_game_over() else 0xFF
    final_vps = [game.get_player_vp(p) for p in range(num_players)]
    
    return {
        'winner': winner,
        'game_over': game.is_game_over(),
        'action_count': action_count,
        'turn_number': game.turn_number,
        'final_vps': final_vps,
    }


def evaluate_matchup(agent1_name, agent1_factory, 
                     agent2_name, agent2_factory,
                     num_games=100, num_players=4):
    """Evaluate two agents against each other."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {agent1_name} vs {agent2_name}")
    print(f"{'='*70}")
    print(f"Games: {num_games} | Players: {num_players}")
    print(f"{'='*70}\n")
    
    agent1_wins = 0
    agent2_wins = 0
    total_actions = 0
    total_turns = 0
    completed_games = 0
    
    agent1_positions = list(range(0, num_players, 2))
    agent2_positions = list(range(1, num_players, 2))
    
    start_time = time.time()
    
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        agents = []
        for p in range(num_players):
            if p in agent1_positions:
                agents.append(agent1_factory())
            else:
                agents.append(agent2_factory())
        
        result = play_game(agents, num_players, seed=game_idx + 12345, max_actions=10000)
        
        if result['game_over']:
            completed_games += 1
            winner = result['winner']
            if winner in agent1_positions:
                agent1_wins += 1
            elif winner in agent2_positions:
                agent2_wins += 1
        
        total_actions += result['action_count']
        total_turns += result['turn_number']
    
    elapsed = time.time() - start_time
    
    completion_rate = completed_games / num_games * 100 if num_games > 0 else 0
    agent1_winrate = agent1_wins / completed_games * 100 if completed_games > 0 else 0
    agent2_winrate = agent2_wins / completed_games * 100 if completed_games > 0 else 0
    avg_actions = total_actions / num_games if num_games > 0 else 0
    avg_turns = total_turns / num_games if num_games > 0 else 0
    
    print(f"\nResults:")
    print(f"  Completed: {completed_games}/{num_games} ({completion_rate:.1f}%)")
    print(f"\n  Win Rates:")
    print(f"    {agent1_name:>20}: {agent1_wins:3d} wins ({agent1_winrate:5.1f}%)")
    print(f"    {agent2_name:>20}: {agent2_wins:3d} wins ({agent2_winrate:5.1f}%)")
    print(f"\n  Game Stats:")
    print(f"    Avg turns:   {avg_turns:.1f}")
    print(f"    Avg actions: {avg_actions:.1f}")
    print(f"\n  Performance:")
    print(f"    Total time:  {elapsed:.2f}s")
    print(f"    Time/game:   {elapsed/num_games:.2f}s")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'agent1_winrate': agent1_winrate,
        'agent2_winrate': agent2_winrate,
        'completed_games': completed_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--players", type=int, default=4, help="Players per game")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create AlphaZero agent from checkpoint
    alphazero_agent = AlphaZeroAgent.from_checkpoint(
        args.checkpoint,
        mcts_simulations=args.simulations,
        device=args.device,
        add_noise=False  # No exploration noise during evaluation
    )
    
    print(f"AlphaZero agent loaded successfully!")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Device: {args.device}")
    
    # Agent factories
    def make_alphazero():
        return alphazero_agent
    
    def make_random():
        return RandomAgent()
    
    # Evaluate
    try:
        evaluate_matchup(
            f"AlphaZero({args.simulations})", make_alphazero,
            "Random", make_random,
            num_games=args.games,
            num_players=args.players
        )
    finally:
        alphazero_agent.cleanup()
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
