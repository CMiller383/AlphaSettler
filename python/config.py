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
        self.selfplay_workers = None  # Auto-detect: min(games, cpu_count*2, 16)
        self.max_batch_size = 64  # Max batch size for batched NN evaluation
        
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


class PACEGPUConfig(TrainingConfig):
    """
    Optimized configuration for PACE GPU cluster (A100/H100).
    Large batch sizes and network to leverage GPU memory and compute.
    """
    
    def __init__(self):
        super().__init__()
        # Large network for GPU
        self.hidden_size = 768
        self.num_residual_blocks = 8
        
        # High MCTS simulations for quality
        self.mcts_simulations = 400
        
        # Large batches to saturate GPU
        self.batch_size = 512  # Increase to 1024 if using A100 80GB or H100
        
        # Aggressive self-play
        self.games_per_iteration = 200
        self.training_epochs_per_iteration = 10
        self.num_iterations = 500  # 100k games
        
        # Large replay buffer
        self.replay_buffer_size = 500000
        
        # More frequent checkpoints for long runs
        self.checkpoint_interval = 10
        self.eval_interval = 20
        self.eval_games = 100
        
        # GPU performance settings
        self.pin_memory = torch.cuda.is_available()
        self.selfplay_workers = 24  # Generous (1.5x cores), won't crash
        self.max_batch_size = 64  # Moderate batches on GPU
        
    def summary(self):
        """Print configuration summary with GPU info."""
        super().summary()
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Pin Memory: {self.pin_memory}")
            print("="*60 + "\n")


class H100Config(PACEGPUConfig):
    """
    Optimized for H100 80GB HGX.
    Higher batch sizes and larger network than A100.
    """
    
    def __init__(self):
        super().__init__()
     #  Medium-sized network for balance
        self.hidden_size = 512
        self.num_residual_blocks = 4
         
        # Moderate MCTS for reasonable quality
        self.mcts_simulations = 200
        
        # 100 games per iteration × 100 iterations = 10,000 games
        self.games_per_iteration = 100
        self.training_epochs_per_iteration = 5
        self.num_iterations = 100
        
        # Standard batch sizes
        self.batch_size = 256
        self.replay_buffer_size = 100000
        
        # SPARSE CHECKPOINTS - only save every 25 iterations (4 checkpoints total)
        # This avoids accumulating too many large checkpoint files
        self.checkpoint_interval = 25
        
        # Reasonable evaluation
        self.eval_interval = 20
        self.eval_games = 50
        
        # GPU/performance settings
        self.selfplay_workers = 16  # Conservative parallel games
        self.max_batch_size = 64  # Reasonable batch size
        
        

class H200Config(PACEGPUConfig):
    """
    Optimized for H200 80GB HGX (or 141GB HBM3e variant).
    Maximum batch sizes and deepest network.
    H200 has 4.8 TB/s memory bandwidth vs H100's 3.35 TB/s.
    """
    
    def __init__(self):
        super().__init__()
        # Maximum batch size for H200's massive bandwidth
        self.batch_size = 1024  # Can go to 2048 with 141GB model
        
        # Very deep network
        self.hidden_size = 1024
        self.num_residual_blocks = 12
        
        # Maximum parallel self-play
        self.games_per_iteration = 400
        
        # Training can be more aggressive
        self.learning_rate = 0.002
        
        # Maximum parallelism
        self.selfplay_workers = 48  # Very generous (3x cores) for max batching
        self.max_batch_size = 256  # Very large NN batches

