"""
Full state encoder for Catan RL.
Converts GameState into a comprehensive feature vector for neural network input.
"""

import numpy as np
from catan_engine import GameState, PieceType, HarborType, Resource


class StateEncoder:
    """
    Encodes Catan game state into a feature vector for neural network input.
    
    Feature categories:
    - Board state: tiles (resource type, number), robber position
    - Player state: resources, dev cards, pieces remaining, VP
    - Pieces on board: settlements, cities, roads (one-hot per player)
    - Turn information: current player, game phase, turn phase
    - Special achievements: longest road, largest army
    
    Total features: ~900 dimensions
    """
    
    def __init__(self):
        # Constants from C++
        self.NUM_TILES = 19
        self.NUM_VERTICES = 54
        self.NUM_EDGES = 72
        self.NUM_HARBORS = 9
        self.MAX_PLAYERS = 4
        self.NUM_RESOURCE_TYPES = 5
        self.NUM_DEV_CARD_TYPES = 5
        
        # Feature dimensions
        self.tile_features = self.NUM_TILES * 8  # resource (6) + number (1) + robber (1)
        self.vertex_features = self.NUM_VERTICES * 9  # empty/settlement/city (3) + owners (5) + harbor (1)
        self.edge_features = self.NUM_EDGES * 5  # occupied (1) + owner (4)
        self.player_features = self.MAX_PLAYERS * 17  # resources + dev cards + pieces + VP + achievements
        self.global_features = 20  # current player, phase, longest road/army owners
        
        self.total_features = (
            self.tile_features +
            self.vertex_features +
            self.edge_features +
            self.player_features +
            self.global_features
        )
    
    def encode_state(self, state: GameState, perspective_player: int = None) -> np.ndarray:
        """
        Encode game state from a specific player's perspective.
        
        Args:
            state: GameState object
            perspective_player: Player index (0-3). If None, uses current_player.
        
        Returns:
            Feature vector of shape (total_features,)
        """
        if perspective_player is None:
            perspective_player = state.current_player
        
        features = []
        
        # 1. Board state (tiles)
        features.extend(self._encode_tiles(state))
        
        # 2. Pieces on board (vertices and edges)
        features.extend(self._encode_vertices(state, perspective_player))
        features.extend(self._encode_edges(state, perspective_player))
        
        # 3. Player states (from perspective player's view)
        features.extend(self._encode_players(state, perspective_player))
        
        # 4. Global game state
        features.extend(self._encode_global(state, perspective_player))
        
        return np.array(features, dtype=np.float32)
    
    def _encode_tiles(self, state: GameState) -> list:
        """Encode board tiles: resource type, number, robber position."""
        features = []
        
        for tile_idx in range(self.NUM_TILES):
            tile = state.board.tiles[tile_idx]
            
            # One-hot encode resource type (6 types: Desert, Brick, Lumber, Wool, Grain, Ore)
            resource_val = int(tile.resource)
            resource_one_hot = [0.0] * 6
            if 0 <= resource_val < 6:
                resource_one_hot[resource_val] = 1.0
            features.extend(resource_one_hot)
            
            # Normalized dice number (0-1 range)
            features.append(tile.number / 12.0 if tile.number > 0 else 0.0)
            
            # Robber on this tile
            features.append(1.0 if state.robber_tile == tile_idx else 0.0)
        
        return features
    
    def _encode_vertices(self, state: GameState, perspective: int) -> list:
        """Encode vertex occupancy (settlements/cities) from perspective player's view."""
        features = []
        
        for vertex_idx in range(self.NUM_VERTICES):
            piece = state.vertex_pieces[vertex_idx]
            
            # Piece type: empty, settlement, city (one-hot)
            piece_type = [0.0, 0.0, 0.0]
            if piece.type == PieceType.Settlement:
                piece_type[1] = 1.0
            elif piece.type == PieceType.City:
                piece_type[2] = 1.0
            else:
                piece_type[0] = 1.0
            features.extend(piece_type)
            
            # Owner relative to perspective (one-hot: none, self, opponent1, opponent2, opponent3)
            owner_encoding = [0.0] * 5
            if piece.type == PieceType.None_:
                owner_encoding[0] = 1.0
            else:
                owner = piece.owner
                if owner == perspective:
                    owner_encoding[1] = 1.0  # Self
                elif owner < state.num_players:
                    # Opponents (relative indexing)
                    opponent_idx = (owner - perspective) % state.num_players
                    if 1 <= opponent_idx <= 3:
                        owner_encoding[1 + opponent_idx] = 1.0
            features.extend(owner_encoding)
            
            # Harbor access (if any harbor is adjacent to this vertex)
            harbor_info = self._get_harbor_at_vertex(state, vertex_idx)
            features.append(harbor_info)
        
        return features
    
    def _encode_edges(self, state: GameState, perspective: int) -> list:
        """Encode edge occupancy (roads) from perspective player's view."""
        features = []
        
        for edge_idx in range(self.NUM_EDGES):
            piece = state.edge_pieces[edge_idx]
            
            # Road presence (binary)
            features.append(1.0 if piece.type == PieceType.Road else 0.0)
            
            # Owner relative to perspective (one-hot: self, opponent1, opponent2, opponent3)
            owner_encoding = [0.0] * 4
            if piece.type == PieceType.Road:
                owner = piece.owner
                if owner == perspective:
                    owner_encoding[0] = 1.0  # Self
                elif owner < state.num_players:
                    opponent_idx = (owner - perspective) % state.num_players
                    if 1 <= opponent_idx <= 3:
                        owner_encoding[opponent_idx] = 1.0
            features.extend(owner_encoding)
        
        return features
    
    def _encode_players(self, state: GameState, perspective: int) -> list:
        """Encode all player states (self first, then opponents in turn order)."""
        features = []
        
        for offset in range(self.MAX_PLAYERS):
            player_idx = (perspective + offset) % state.num_players
            
            if player_idx < state.num_players:
                player = state.players[player_idx]
                
                # Resources (normalized by max reasonable amount ~20)
                for r in range(self.NUM_RESOURCE_TYPES):
                    features.append(player.resources[r] / 20.0)
                
                # Dev cards (normalized by max in deck)
                for d in range(self.NUM_DEV_CARD_TYPES):
                    features.append(player.dev_cards[d] / 5.0)
                
                # Pieces remaining (normalized by starting amount)
                features.append(player.settlements_remaining / 5.0)
                features.append(player.cities_remaining / 4.0)
                features.append(player.roads_remaining / 15.0)
                
                # Victory points (normalized by win condition)
                features.append(player.public_victory_points / 10.0)
                features.append(player.hidden_victory_points / 10.0)
                
                # Knights played (for largest army)
                features.append(player.knights_played / 10.0)
                
                # Special achievements
                features.append(1.0 if player.has_longest_road else 0.0)
                features.append(1.0 if player.has_largest_army else 0.0)
            else:
                # Padding for games with <4 players
                features.extend([0.0] * 17)
        
        return features
    
    def _encode_global(self, state: GameState, perspective: int) -> list:
        """Encode global game state information."""
        features = []
        
        # Current player (relative to perspective)
        current_relative = (state.current_player - perspective) % state.num_players
        current_one_hot = [0.0] * 4
        if current_relative < 4:
            current_one_hot[current_relative] = 1.0
        features.extend(current_one_hot)
        
        # Game phase (one-hot: Setup, MainGame)
        features.append(1.0 if state.game_phase == 0 else 0.0)  # Setup
        features.append(1.0 if state.game_phase == 1 else 0.0)  # MainGame
        
        # Turn phase (one-hot: RollDice, Discard, Robber, Trading)
        turn_phase_one_hot = [0.0] * 4
        if state.turn_phase < 4:
            turn_phase_one_hot[state.turn_phase] = 1.0
        features.extend(turn_phase_one_hot)
        
        # Turn number (normalized)
        features.append(state.turn_number / 100.0)
        
        # Longest road owner (relative to perspective)
        if state.longest_road_owner == 0xFF:
            features.extend([1.0, 0.0, 0.0, 0.0])  # No owner
        else:
            lr_relative = (state.longest_road_owner - perspective) % state.num_players
            lr_one_hot = [0.0] * 4
            if lr_relative < 4:
                lr_one_hot[lr_relative] = 1.0
            features.extend(lr_one_hot)
        
        # Largest army owner (relative to perspective)
        if state.largest_army_owner == 0xFF:
            features.extend([1.0, 0.0, 0.0, 0.0])  # No owner
        else:
            la_relative = (state.largest_army_owner - perspective) % state.num_players
            la_one_hot = [0.0] * 4
            if la_relative < 4:
                la_one_hot[la_relative] = 1.0
            features.extend(la_one_hot)
        
        # Number of players in game
        features.append(state.num_players / 4.0)
        
        return features
    
    def _get_harbor_at_vertex(self, state: GameState, vertex_idx: int) -> float:
        """
        Check if vertex has harbor access. Returns normalized harbor value.
        0.0 = no harbor, 0.5 = generic 3:1, 1.0 = specific 2:1
        """
        vertex = state.board.vertices[vertex_idx]
        
        for edge_idx in vertex.edges:
            if edge_idx == 0xFF:  # INVALID_EDGE
                continue
            
            # Check all harbors
            for harbor in state.board.harbors:
                if harbor.edge_idx == edge_idx:
                    # Harbor found
                    if harbor.type == HarborType.Generic:
                        return 0.5  # 3:1 harbor
                    else:
                        return 1.0  # 2:1 specific harbor
        
        return 0.0  # No harbor
    
    def get_feature_size(self) -> int:
        """Return total number of features in encoded state."""
        return self.total_features
