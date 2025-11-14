#!/usr/bin/env python3
"""
Evaluation script for Catan agents.
Compares MCTS agent against random baseline.
"""

import time
import argparse
from typing import Dict
from tqdm import tqdm

try:
    from catan_engine import (
        GameState, MCTSAgent, MCTSConfig, apply_action
    )
    from agents import RandomAgent
except ImportError:
    print("Error: catan_engine module not found!")
    print("Build the Python bindings first: pip install -e .")
    exit(1)


def play_game(agents, num_players=4, seed=None, max_actions=10000, verbose=False):
    """
    Play a single game with given agents.
    
    Args:
        agents: List of agents (length = num_players)
        num_players: Number of players
        seed: Random seed for game
        max_actions: Maximum actions before timeout
        verbose: Print game progress
    
    Returns:
        Dict with game results
    """
    game = GameState.create_new_game(num_players, seed or int(time.time() * 1000))
    
    action_count = 0
    while not game.is_game_over() and action_count < max_actions:
        current_player = game.current_player
        agent = agents[current_player]
        
        # Get action from agent
        action = agent.select_action(game)
        if action is None:
            if verbose:
                print(f"ERROR: No legal action at move {action_count}")
            break
        
        # Apply action
        apply_action(game, action, action_count * 7919)  # Prime seed
        action_count += 1
    
    # Collect results
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
    """
    Evaluate two agents against each other.
    
    Args:
        agent1_name: Name of first agent type
        agent1_factory: Function that creates agent1 instances
        agent2_name: Name of second agent type
        agent2_factory: Function that creates agent2 instances
        num_games: Number of games to play
        num_players: Players per game (must be even for fair matchup)
    
    Returns:
        Dict with evaluation statistics
    """
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
    
    # Track wins by position
    agent1_positions = list(range(0, num_players, 2))  # Even positions
    agent2_positions = list(range(1, num_players, 2))  # Odd positions
    
    start_time = time.time()
    
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        # Create agents for this game
        agents = []
        for p in range(num_players):
            if p in agent1_positions:
                agents.append(agent1_factory())
            else:
                agents.append(agent2_factory())
        
        # Play game
        result = play_game(agents, num_players, seed=game_idx + 12345, max_actions=10000)
        
        # Collect stats
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
    
    # Calculate statistics
    completion_rate = completed_games / num_games * 100 if num_games > 0 else 0
    agent1_winrate = agent1_wins / completed_games * 100 if completed_games > 0 else 0
    agent2_winrate = agent2_wins / completed_games * 100 if completed_games > 0 else 0
    avg_actions = total_actions / num_games if num_games > 0 else 0
    avg_turns = total_turns / num_games if num_games > 0 else 0
    
    # Print results
    print(f"\nResults:")
    print(f"  Completed: {completed_games}/{num_games} ({completion_rate:.1f}%)")
    print(f"\n  Win Rates:")
    print(f"    {agent1_name:>15}: {agent1_wins:3d} wins ({agent1_winrate:5.1f}%)")
    print(f"    {agent2_name:>15}: {agent2_wins:3d} wins ({agent2_winrate:5.1f}%)")
    print(f"\n  Game Stats:")
    print(f"    Avg turns:   {avg_turns:.1f}")
    print(f"    Avg actions: {avg_actions:.1f}")
    print(f"\n  Performance:")
    print(f"    Total time:  {elapsed:.2f}s")
    print(f"    Time/game:   {elapsed/num_games:.2f}s")
    print(f"    Games/min:   {(num_games/elapsed)*60:.1f}")
    
    return {
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'agent1_winrate': agent1_winrate,
        'agent2_winrate': agent2_winrate,
        'completed_games': completed_games,
        'completion_rate': completion_rate,
        'avg_turns': avg_turns,
        'avg_actions': avg_actions,
        'total_time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Catan agents")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--iterations", type=int, default=100, help="MCTS iterations")
    parser.add_argument("--players", type=int, default=4, help="Players per game")
    parser.add_argument("--mode", choices=["mcts_vs_random", "mcts_selfplay"], 
                       default="mcts_vs_random", help="Evaluation mode")
    
    args = parser.parse_args()
    
    # Agent factories
    def make_mcts():
        config = MCTSConfig()
        config.num_iterations = args.iterations
        config.exploration_constant = 1.41
        return MCTSAgent(config)
    
    def make_random():
        return RandomAgent()
    
    if args.mode == "mcts_vs_random":
        evaluate_matchup(
            f"MCTS({args.iterations})", make_mcts,
            "Random", make_random,
            num_games=args.games,
            num_players=args.players
        )
    
    elif args.mode == "mcts_selfplay":
        evaluate_matchup(
            f"MCTS({args.iterations})", make_mcts,
            f"MCTS({args.iterations})", make_mcts,
            num_games=args.games,
            num_players=args.players
        )
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
