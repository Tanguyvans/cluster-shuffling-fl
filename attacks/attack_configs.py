"""
Attack configuration presets for different scenarios
"""

ATTACK_CONFIGS = {
    'quick_test': {
        'signed': True,
        'boxed': True,
        'cost_fn': 'sim',
        'indices': 'def',
        'weights': 'equal',
        'lr': 0.1,
        'optim': 'adam',
        'restarts': 1,
        'max_iterations': 8000,
        'total_variation': 1e-2,
        'init': 'randn',
        'filter': 'none',
        'lr_decay': True,
        'scoring_choice': 'loss'
    },
    
    'default': {
        'signed': True,
        'boxed': True,
        'cost_fn': 'sim',
        'indices': 'def',
        'weights': 'equal',
        'lr': 0.1,
        'optim': 'adam',
        'restarts': 2,
        'max_iterations': 24000,
        'total_variation': 1e-2,
        'init': 'randn',
        'filter': 'none',
        'lr_decay': True,
        'scoring_choice': 'loss'
    },
    
    'aggressive': {
        'signed': True,
        'boxed': True,
        'cost_fn': 'sim',
        'indices': 'def',
        'weights': 'equal',
        'lr': 0.01,  # Lower learning rate for better reconstruction
        'optim': 'adam',
        'restarts': 5,  # More restarts
        'max_iterations': 48000,  # More iterations
        'total_variation': 1e-3,  # Less regularization
        'init': 'randn',
        'filter': 'none',
        'lr_decay': True,
        'scoring_choice': 'loss'
    },
    
    'conservative': {
        'signed': True,
        'boxed': True,
        'cost_fn': 'sim',
        'indices': 'def',
        'weights': 'equal',
        'lr': 0.2,  # Higher learning rate
        'optim': 'adam',
        'restarts': 1,
        'max_iterations': 12000,  # Fewer iterations
        'total_variation': 1e-1,  # More regularization
        'init': 'randn',
        'filter': 'none',
        'lr_decay': True,
        'scoring_choice': 'loss'
    },
    
    'high_quality': {
        'signed': True,
        'boxed': True,
        'cost_fn': 'l2',  # L2 loss for better pixel accuracy
        'indices': 'def',
        'weights': 'equal',
        'lr': 0.05,
        'optim': 'adam',
        'restarts': 8,  # Many restarts
        'max_iterations': 60000,  # Long optimization
        'total_variation': 5e-4,  # Very little regularization
        'init': 'randn',
        'filter': 'none',
        'lr_decay': True,
        'scoring_choice': 'loss'
    }
}