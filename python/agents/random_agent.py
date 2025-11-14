"""
Random agent baseline for Catan.
Selects uniformly random legal actions.
"""

import random
from catan_engine import generate_legal_actions


class RandomAgent:
    """Baseline agent that selects random legal actions."""
    
    def __init__(self, seed=None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.rng = random.Random(seed)
    
    def select_action(self, game):
        """
        Select a random legal action.
        
        Args:
            game: GameState object
            
        Returns:
            Random Action from legal actions, or None if no legal actions
        """
        actions = generate_legal_actions(game)
        if not actions:
            return None
        return self.rng.choice(actions)
