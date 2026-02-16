"""Evaluate trained checkpoints vs random agents.
Uses the same AlphaZeroAgent that would be used in real play — no custom evaluator."""
import os, sys, numpy as np, torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(_ROOT)
    _BUILD_RELEASE = os.path.join(_ROOT, 'build', 'Release')
    if os.path.isdir(_BUILD_RELEASE):
        os.add_dll_directory(_BUILD_RELEASE)

# Import catan_engine BEFORE adding python/ (python/ has a stale .pyd copy)
import catan_engine
from catan_engine import (
    GameState, generate_legal_actions, apply_action, ActionType
)

# Python modules
sys.path.insert(0, os.path.join(_ROOT, 'python'))
from agents.alphazero_agent import AlphaZeroAgent
from state_encoder import StateEncoder
from catan_network import CatanNetwork

FEATURE_SIZE = StateEncoder().get_feature_size()
HIDDEN_SIZE = 128
NUM_BLOCKS = 2
MCTS_SIMS = 50
NUM_GAMES = 30

def make_agent(checkpoint_path=None):
    """Create an AlphaZeroAgent — untrained if no checkpoint, else from saved weights."""
    if checkpoint_path is None:
        # Untrained network — random weights
        net = CatanNetwork(input_size=FEATURE_SIZE, hidden_size=HIDDEN_SIZE,
                           num_residual_blocks=NUM_BLOCKS)
        net.eval()
        return AlphaZeroAgent(net, mcts_simulations=MCTS_SIMS, device='cpu',
                              enable_batching=True, add_noise=False)
    else:
        return AlphaZeroAgent.from_checkpoint(
            checkpoint_path, mcts_simulations=MCTS_SIMS, device='cpu',
            hidden_size=HIDDEN_SIZE, num_residual_blocks=NUM_BLOCKS)

def play_game(agent, seed):
    """Play one 4-player game: agent as P0 vs 3 random opponents."""
    state = GameState.create_new_game(4, seed)
    turn = 0
    while not state.is_game_over() and turn < 2000:
        actions = generate_legal_actions(state)
        if not actions:
            break
        if state.current_player == 0:
            action = agent.select_action(state)
        else:
            action = actions[np.random.randint(len(actions))]
        apply_action(state, action, np.random.randint(0, 1_000_000))
        turn += 1
    winner = state.get_winner() if state.is_game_over() else None
    vps = [state.players[i].total_victory_points() for i in range(4)]
    return winner, vps, turn

# --- Run evaluation ---
RUN_DIR = os.path.join(_ROOT, 'training_runs', '20260216_105636', 'checkpoints')

agents_spec = [
    ("Untrained", None),
    ("Iter-5",    os.path.join(RUN_DIR, 'checkpoint_iter_5.pt')),
    ("Iter-10",   os.path.join(RUN_DIR, 'checkpoint_iter_10.pt')),
    ("Final",     os.path.join(RUN_DIR, 'final_model.pt')),
]

print(f"{'Agent':<15} {'Wins':>5} {'Rate':>6} {'Avg P0 VP':>10} {'Avg Turns':>10}")
print("-" * 55)

for name, ckpt in agents_spec:
    agent = make_agent(ckpt)
    wins, total_vp, total_turns = 0, 0, 0
    for g in range(NUM_GAMES):
        winner, vps, turns = play_game(agent, seed=g * 137 + 7)
        if winner == 0:
            wins += 1
        total_vp += vps[0]
        total_turns += turns
    agent.cleanup()
    avg_vp = total_vp / NUM_GAMES
    avg_turns = total_turns / NUM_GAMES
    print(f"{name:<15} {wins:>5} {100 * wins / NUM_GAMES:>5.0f}% {avg_vp:>10.1f} {avg_turns:>10.0f}")

print(f"\n{NUM_GAMES} games per agent, {MCTS_SIMS} MCTS sims, pure NN (no heuristics).")
print("Expected: untrained ~25%, trained increasing → learning is real.")
