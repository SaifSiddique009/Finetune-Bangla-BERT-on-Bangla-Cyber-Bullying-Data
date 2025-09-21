EXPERIMENT_CONFIGS = [
    # Original Batch 16
    {'batch_size': 16, 'learning_rate': 3e-5, 'num_epochs': 10, 'freeze_base': False},
    {'batch_size': 16, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': False},
    {'batch_size': 16, 'learning_rate': 2e-5, 'num_epochs': 30, 'freeze_base': False},

    # Original Batch 32
    {'batch_size': 32, 'learning_rate': 3e-5, 'num_epochs': 10, 'freeze_base': False},
    {'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': False},
    {'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 30, 'freeze_base': False},

    # Original Batch 64
    {'batch_size': 64, 'learning_rate': 3e-5, 'num_epochs': 10, 'freeze_base': False},
    {'batch_size': 64, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': False},
    {'batch_size': 64, 'learning_rate': 2e-5, 'num_epochs': 30, 'freeze_base': False},

    # Additions for Higher Performance
    # Lower LR for stability
    {'batch_size': 32, 'learning_rate': 1e-5, 'num_epochs': 20, 'freeze_base': False},
    {'batch_size': 32, 'learning_rate': 1e-5, 'num_epochs': 30, 'freeze_base': False},

    # Slightly higher LR for faster convergence
    {'batch_size': 32, 'learning_rate': 5e-5, 'num_epochs': 10, 'freeze_base': False},

    # With Freezing for Better Generalization
    {'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': True},
    {'batch_size': 32, 'learning_rate': 1e-5, 'num_epochs': 30, 'freeze_base': True},
]