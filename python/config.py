"""
Training configuration for Catan AlphaZero.
"""

import torch


class TrainingConfig:
    """
    Single training config — tuned for a ~45-minute CPU run.
    
    60 iterations × 20 games = 1,200 total games.
    ~43s/iter → ~43 min on a modern CPU.
    """
    
    def __init__(self):
        # Network
        self.hidden_size = 128
        self.num_residual_blocks = 2
        
        # Training
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 64
        self.replay_buffer_size = 100000
        
        # MCTS
        self.mcts_simulations = 50
        self.cpuct = 1.5
        self.dirichlet_alpha = 0.3
        self.dirichlet_weight = 0.25
        
        # Game
        self.num_players = 4
        
        # Schedule
        self.games_per_iteration = 20
        self.training_epochs_per_iteration = 3
        self.num_iterations = 60
        
        # Device & performance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.selfplay_workers = 4
        self.max_batch_size = 16
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_interval = 10
        self.log_interval = 1
        
        # Value heuristic bootstrap
        self.heuristic_value_weight = 0.8
        self.heuristic_decay_iterations = None  # defaults to num_iterations
        
        # Evaluation
        self.eval_games = 10
        self.eval_interval = 10
    
    def summary(self):
        total_games = self.num_iterations * self.games_per_iteration
        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Network:  {self.hidden_size} hidden, {self.num_residual_blocks} res blocks")
        print(f"MCTS:     {self.mcts_simulations} sims/move, cpuct={self.cpuct}")
        print(f"Schedule: {self.num_iterations} iters × {self.games_per_iteration} games = {total_games} games")
        print(f"Training: batch={self.batch_size}, lr={self.learning_rate}, epochs={self.training_epochs_per_iteration}")
        print(f"Device:   {self.device}")
        print(f"Workers:  {self.selfplay_workers}")
        print(f"Heuristic value weight: {self.heuristic_value_weight} (decay over {self.heuristic_decay_iterations or self.num_iterations} iters)")
        print("=" * 60 + "\n")

