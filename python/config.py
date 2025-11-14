"""
Training configuration for Catan AlphaZero.
Adjust these settings to control training behavior.
"""

import torch


class TrainingConfig:
    """Configuration for AlphaZero training."""
    
    def __init__(self):
        # ====================================================================
        # Network Architecture
        # ====================================================================
        self.hidden_size = 256  # Hidden layer size (256/512/1024)
        self.num_residual_blocks = 3  # Number of ResNet blocks (2-6)
        
        # ====================================================================
        # Training Hyperparameters
        # ====================================================================
        self.learning_rate = 0.001  # Adam learning rate
        self.weight_decay = 1e-4  # L2 regularization
        self.batch_size = 128  # Training batch size
        self.replay_buffer_size = 50000  # Max training examples to keep
        
        # ====================================================================
        # MCTS Parameters
        # ====================================================================
        self.mcts_simulations = 100  # MCTS simulations per move (100-400)
        self.cpuct = 1.5  # PUCT exploration constant
        self.dirichlet_alpha = 0.3  # Dirichlet noise alpha
        self.dirichlet_weight = 0.25  # Dirichlet noise weight
        
        # ====================================================================
        # Game Settings
        # ====================================================================
        self.num_players = 4  # Players per game (2-4)
        
        # ====================================================================
        # Training Schedule
        # ====================================================================
        self.games_per_iteration = 50  # Self-play games per iteration
        self.training_epochs_per_iteration = 3  # Training epochs per iteration
        self.num_iterations = 20  # Total training iterations
        
        # ====================================================================
        # Device & Performance
        # ====================================================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0  # DataLoader workers (0 for Windows compatibility)
        
        # ====================================================================
        # Checkpointing & Logging
        # ====================================================================
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_interval = 5  # Save every N iterations
        self.log_interval = 1  # Log stats every N iterations
        
        # ====================================================================
        # Evaluation
        # ====================================================================
        self.eval_games = 20  # Games to play for evaluation
        self.eval_interval = 5  # Evaluate every N iterations
    
    def summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Network: {self.hidden_size} hidden, {self.num_residual_blocks} blocks")
        print(f"MCTS: {self.mcts_simulations} simulations per move")
        print(f"Training: {self.num_iterations} iterations × {self.games_per_iteration} games")
        print(f"         = {self.num_iterations * self.games_per_iteration} total games")
        print(f"Batch size: {self.batch_size}, LR: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Checkpoints: {self.checkpoint_dir}/ (every {self.checkpoint_interval} iters)")
        print("="*60 + "\n")


class QuickTestConfig(TrainingConfig):
    """Quick test configuration for debugging (very small)."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.num_residual_blocks = 2
        self.mcts_simulations = 25
        self.games_per_iteration = 5
        self.training_epochs_per_iteration = 2
        self.num_iterations = 3
        self.batch_size = 32
        self.replay_buffer_size = 1000
        self.checkpoint_interval = 1
        self.eval_interval = 1
        self.eval_games = 5


class SmallTrainingConfig(TrainingConfig):
    """Small training run for initial testing."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 256
        self.num_residual_blocks = 3
        self.mcts_simulations = 50
        self.games_per_iteration = 50
        self.training_epochs_per_iteration = 3
        self.num_iterations = 10  # 1000 total games
        self.batch_size = 128
        self.replay_buffer_size = 50000
        self.checkpoint_interval = 10
        self.eval_interval = 5
        self.eval_games = 20


class MediumTrainingConfig(TrainingConfig):
    """Medium training run for longer experiments."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        self.num_residual_blocks = 4
        self.mcts_simulations = 200
        self.games_per_iteration = 100
        self.training_epochs_per_iteration = 5
        self.num_iterations = 100  # 10k total games
        self.batch_size = 256
        self.replay_buffer_size = 100000
        self.checkpoint_interval = 10
        self.eval_interval = 10
        self.eval_games = 50


class LargeTrainingConfig(TrainingConfig):
    """Large training run for full-scale experiments."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        self.num_residual_blocks = 6
        self.mcts_simulations = 400
        self.games_per_iteration = 200
        self.training_epochs_per_iteration = 10
        self.num_iterations = 500  # 100k total games
        self.batch_size = 512
        self.replay_buffer_size = 200000
        self.checkpoint_interval = 20
        self.eval_interval = 20
        self.eval_games = 100
