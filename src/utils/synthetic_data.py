"""
Synthetic data generator for Pre-Flight compute cluster checks.
Bypasses raw file parsing to return mock data dictionaries that can be fed
directly into the preprocessing and memmap serialization pipeline.
"""
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("PerconAI.synthetic")

def generate_synthetic_subjects(dataset_name: str, num_subjects: int = 3) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Generates synthetic subject data matching the raw parsing output format.
    
    Expected shapes for raw parsing output (before LOSO grouping and windowing,
    but here we can just mock the windowed output directly if we bypass earlier steps.
    Wait, the raw parsers return windowed data:
    _parse_uci_har -> returns X: (n_windows, n_channels, window_len), y: (n_windows,)
    )
    
    UCI-HAR: 9 channels, 6 classes, 128 window
    PAMAP2: 13 channels, 12 classes, 128 window
    HHAR: 6 channels, 6 classes, 128 window
    """
    logger.info("Generating SYNTHETIC data for %s", dataset_name)
    
    if dataset_name == "uci_har":
        n_ch, n_classes = 9, 6
    elif dataset_name == "pamap2":
        n_ch, n_classes = 13, 12
    elif dataset_name == "hhar":
        n_ch, n_classes = 6, 6
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    window_len = 128
    windows_per_subject = 50  # Small number for fast micro-runs
    
    subjects = {}
    # Use deterministic synthetic data
    rng = np.random.RandomState(42)
    
    for sid in range(1, num_subjects + 1):
        X = rng.randn(windows_per_subject, n_ch, window_len).astype(np.float32)
        y = rng.randint(0, n_classes, size=(windows_per_subject,)).astype(np.int64)
        subjects[sid] = {"X": X, "y": y}
        
    return subjects
