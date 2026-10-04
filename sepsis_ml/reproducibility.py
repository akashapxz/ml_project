"""
Reproducibility and Random Seed Enforcement Module.
Ensures deterministic runs across all experiments in 24DS601 Sepsis ML project.
"""

import os
import random
import numpy as np


DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """
    Enforces deterministic random state across Python built-in random,
    NumPy, and OS environment variables.
    
    Args:
        seed: Integer random seed (default: 42).
        
    Returns:
        The seed integer set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Try setting for XGBoost if present
    try:
        import xgboost as xgb
        # XGBoost handles random_state parameter on model level
    except ImportError:
        pass
        
    return seed
