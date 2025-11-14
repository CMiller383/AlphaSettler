"""
Full state encoder for Catan RL.
Converts GameState into a comprehensive feature vector for neural network input.
Uses fast C++ backend for performance.
"""

import numpy as np
from catan_engine import GameState, StateEncoder as CppStateEncoder


class StateEncoder:
    """
    Encodes Catan game state into a feature vector for neural network input.
    
    This is a thin Python wrapper around the C++ StateEncoder for performance.
    The C++ implementation is 10-50x faster than pure Python.
    
    Feature categories:
    - Board state: tiles (resource type, number), robber position
    - Player state: resources, dev cards, pieces remaining, VP
    - Pieces on board: settlements, cities, roads (one-hot per player)
    - Turn information: current player, game phase, turn phase
    - Special achievements: longest road, largest army
    
    Total features: ~900 dimensions
    """
    
    def __init__(self):
        # Use C++ encoder backend for performance
        self._cpp_encoder = CppStateEncoder()
        self.total_features = CppStateEncoder.get_feature_size()
    
    def encode_state(self, state: GameState, perspective_player: int = None) -> np.ndarray:
        """
        Encode game state from a specific player's perspective.
        
        Args:
            state: GameState object
            perspective_player: Player index (0-3). If None, uses current_player.
        
        Returns:
            Feature vector of shape (total_features,) as numpy array
        """
        if perspective_player is None:
            perspective_player = state.current_player
        
        # Call C++ encoder (returns numpy array directly)
        return self._cpp_encoder.encode_state(state, perspective_player)
    
    def get_feature_size(self) -> int:
        """Return total number of features in encoded state."""
        return self.total_features
