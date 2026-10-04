"""
Centralized Research Logger for 24DS601 Sepsis ML Project.
Produces formatted console output and writes structured logs to disk.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "sepsis_ml", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Constructs or retrieves a configured logger with consistent formatting.
    
    Args:
        name: Name of the logger.
        log_file: Optional file path for persistent logs on disk.
        
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Optional File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
