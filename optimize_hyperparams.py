#!/usr/bin/env python3
"""
Hyperparameter optimization using Bayesian optimization
"""

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def objective(params):
    """Objective function to maximize utilization"""
    learning_rate, gamma, eps_decay = params
    
    # Train model with these parameters
    # Return negative utilization (since we minimize)
    utilization = train_model_with_params(learning_rate, gamma, eps_decay)
    return -utilization

# Define search space
space = [
    Real(1e-5, 1e-2, name='learning_rate'),
    Real(0.9, 0.999, name='gamma'), 
    Real(0.99, 0.9999, name='eps_decay')
]

# Run optimization
result = gp_minimize(objective, space, n_calls=50)
print(f"Best parameters: {result.x}")