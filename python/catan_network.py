"""Neural network architecture for Catan AlphaZero.

ResNet-style trunk with policy and value heads. The policy head
produces a fixed-size logit vector; for any particular state we
mask this down to the first N entries corresponding to that state's
legal actions (0..N-1). No global action encoding is used; the
indexing is purely local to the current legal action list, which
matches the C++ MCTS implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and skip connections.
    Standard building block for deep RL networks.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class CatanNetwork(nn.Module):
    """
    Neural network for Catan game state evaluation.
    
    Architecture:
    - Input: State features (~800-1000 dims)
    - Shared trunk: Fully connected layers with residual blocks
    - Policy head: Outputs action logits (variable size, masked during training)
    - Value head: Outputs win probability for current player
    
    Args:
        input_size: Number of input features from state encoder
        hidden_size: Hidden layer dimension (default 512)
        num_residual_blocks: Number of residual blocks (default 4)
        max_action_space: Maximum number of possible actions (default 300)
    """
    
    def __init__(self, input_size, hidden_size=512, num_residual_blocks=4, max_action_space=300):
        super(CatanNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_action_space = max_action_space
        
        # Input projection
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # Shared trunk with residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_bn = nn.BatchNorm1d(hidden_size // 2)
        self.policy_fc2 = nn.Linear(hidden_size // 2, max_action_space)
        
        # Value head
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.value_bn = nn.BatchNorm1d(hidden_size // 4)
        self.value_fc2 = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, state_features, action_mask=None):
        """
        Forward pass through the network.
        
        Args:
            state_features: Tensor of shape (batch_size, input_size)
            action_mask: Optional boolean tensor of shape (batch_size, max_action_space)
                        True for legal actions, False for illegal. If None, no masking.
        
        Returns:
            policy_logits: Tensor of shape (batch_size, max_action_space)
            value: Tensor of shape (batch_size, 1)
        """
        # Input projection
        x = F.relu(self.input_bn(self.input_fc(state_features)))
        
        # Shared trunk
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_fc1(x)))
        policy_logits = self.policy_fc2(policy)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set logits of illegal actions to large negative value
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
        
        # Value head
        value = F.relu(self.value_bn(self.value_fc1(x)))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy_logits, value
    
    def predict(self, state_features, legal_action_indices):
        """
        Predict policy and value for a single state (inference mode).
        
        Args:
            state_features: numpy array of shape (input_size,)
            legal_action_indices: list of legal action indices
        
        Returns:
            policy: numpy array of shape (len(legal_action_indices),) - probabilities
            value: float - expected value in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and move to correct device
            state_tensor = torch.from_numpy(state_features).unsqueeze(0).float()
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            
            # Create action mask for local indices 0..N-1, clamped to
            # the network's max_action_space.
            valid_indices = [int(i) for i in legal_action_indices if 0 <= int(i) < self.max_action_space]
            if not valid_indices:
                valid_indices = [0]

            action_mask = torch.zeros(1, self.max_action_space, dtype=torch.bool, device=device)
            for idx in valid_indices:
                action_mask[0, idx] = True
            
            # Forward pass
            policy_logits, value = self.forward(state_tensor, action_mask)
            
            # Convert policy logits to probabilities (only for valid legal actions)
            policy_probs = F.softmax(policy_logits, dim=1)

            # Extract probabilities for valid legal actions (indices 0..N-1)
            legal_probs = policy_probs[0, valid_indices].cpu().numpy()
            
            # Renormalize to sum to 1 (in case of numerical issues)
            prob_sum = legal_probs.sum()
            if prob_sum > 0:
                legal_probs = legal_probs / prob_sum
            else:
                # All probabilities are zero - use uniform distribution
                legal_probs = np.ones_like(legal_probs) / len(legal_probs)

            return legal_probs, value.item()


def create_network(input_size, hidden_size=512, num_blocks=4):
    """
    Factory function to create network with standard configuration.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_blocks: Number of residual blocks
    
    Returns:
        CatanNetwork instance
    """
    return CatanNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        num_residual_blocks=num_blocks,
        max_action_space=300
    )
