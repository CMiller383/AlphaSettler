#!/usr/bin/env python3
"""
Quick test to verify Python bindings work correctly.
"""

try:
    from catan_engine import *
    print("✓ Successfully imported catan_engine")
except ImportError as e:
    print(f"✗ Failed to import catan_engine: {e}")
    print("\nPlease build the Python bindings first:")
    print("  pip install pybind11 numpy")
    print("  pip install -e .")
    exit(1)

print("\nTesting basic functionality...")

# Test game creation
print("  Creating game...")
game = GameState.create_new_game(4, 42)
print(f"  ✓ {game}")

# Test legal actions
print("  Generating legal actions...")
actions = generate_legal_actions(game)
print(f"  ✓ Found {len(actions)} legal actions")

# Test MCTS agent
print("  Creating MCTS agent...")
config = MCTSConfig()
config.num_iterations = 50
agent = MCTSAgent(config)
print(f"  ✓ {config}")

# Test action selection
print("  Selecting action...")
action = agent.select_action(game)
print(f"  ✓ Selected {action}")

# Test action policy
print("  Getting action policy...")
policy = agent.get_action_policy(game)
print(f"  ✓ Policy has {len(policy)} actions")
if policy:
    print(f"     Top action probability: {policy[0].probability:.3f}")

# Test state copy
print("  Testing state copy...")
game_copy = game.copy()
print(f"  ✓ Copied state: {game_copy}")

# Test resource access
print("  Testing player data access...")
vp = game.get_player_vp(0)
resources = game.get_player_resources(0)
print(f"  ✓ Player 0: {vp} VP, {len(resources)} resource types")

print("\n" + "="*50)
print("All basic tests passed!")
print("="*50)
print("\nRun full evaluation:")
print("  python python/evaluate.py --mode single")
print("  python python/evaluate.py --mode selfplay --games 10")
