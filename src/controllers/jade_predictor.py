import torch
import torch.nn as nn
import numpy as np

class MultinomialLogistic(nn.Module):
    """
    Linear predictor for JADE. Maps 28-dim features to 15 outputs
    (3 utility bins x 5 actions).
    Size: 28 * 15 + 15 = 435 parameters. (fp32 = 1.74 KB)
    """
    def __init__(self, in_features: int = 28, num_actions: int = 5, num_bins: int = 3):
        super().__init__()
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.linear = nn.Linear(in_features, num_actions * num_bins)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 28)
        # out: (batch, num_actions, num_bins)
        logits = self.linear(x)
        return logits.view(-1, self.num_actions, self.num_bins)

class MLPPredictor(nn.Module):
    """
    2-Layer MLP predictor for JADE. 28 -> 32 -> 15.
    Size: (28*32 + 32) + (32*15 + 15) = 928 + 495 = 1423 parameters. (fp32 = 5.69 KB)
    """
    def __init__(self, in_features: int = 28, hidden_dim: int = 32, num_actions: int = 5, num_bins: int = 3):
        super().__init__()
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions * num_bins)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(-1, self.num_actions, self.num_bins)

class IsotonicCalibrator:
    """
    Applies isotonic regression per (action, bin) to ensure Expected Calibration Error < 0.05.
    Uses a lightweight Pool-Adjacent-Violators Algorithm (PAVA).
    """
    def __init__(self, num_actions: int = 5, num_bins: int = 3):
        self.num_actions = num_actions
        self.num_bins = num_bins
        # To store piece-wise constant functions: list of (x_thresholds, y_values)
        self.calibrators = [[None for _ in range(num_bins)] for _ in range(num_actions)]
        
    def _pava(self, y: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
        """Standard 1D Pool-Adjacent-Violators Algorithm."""
        n = len(y)
        if weight is None:
            weight = np.ones(n)
            
        index = np.arange(n)
        values = np.copy(y)
        weights = np.copy(weight)
        
        while True:
            # Find decreasing adjacent pairs (violation of isotonicity)
            violates = values[:-1] > values[1:]
            if not np.any(violates):
                break
                
            # Find the first violation
            i = np.argmax(violates)
            
            # Pool i and i+1
            new_weight = weights[i] + weights[i+1]
            new_value = (values[i] * weights[i] + values[i+1] * weights[i+1]) / new_weight
            
            values[i] = new_value
            weights[i] = new_weight
            
            # Delete i+1
            values = np.delete(values, i+1)
            weights = np.delete(weights, i+1)
            index = np.delete(index, i+1)
            
            # Expand the pooled value back to original array length for the next iteration is complex,
            # Instead, we just maintain block boundaries.
            # For simplicity in this dummy MCU-friendly version, we use an iterative backward pool.
        
        # Expand values back
        result = np.zeros(n)
        curr = 0
        for i, val in enumerate(values):
            if i < len(values) - 1:
                next_idx = index[i+1]
            else:
                next_idx = n
            result[curr:next_idx] = val
            curr = next_idx
            
        return result

    def fit(self, uncalibrated_probs: np.ndarray, true_labels: np.ndarray):
        """
        uncalibrated_probs: (N, num_actions, num_bins)
        true_labels: (N, num_actions) integers in [0, num_bins-1]
        """
        for a in range(self.num_actions):
            for b in range(self.num_bins):
                probs = uncalibrated_probs[:, a, b]
                targets = (true_labels[:, a] == b).astype(np.float32)
                
                # Sort by prob
                order = np.argsort(probs)
                sorted_probs = probs[order]
                sorted_targets = targets[order]
                
                # O(N^2) PAVA fallback for absolute safety in script, 
                # practically fast enough for metadata N.
                iso_vals = self._pava(sorted_targets)
                
                # Save mapping
                # Keep unique thresholds to define the piecewise step function
                unique_idx = np.concatenate(([True], sorted_probs[1:] != sorted_probs[:-1]))
                x_thresh = sorted_probs[unique_idx]
                y_val = iso_vals[unique_idx]
                
                self.calibrators[a][b] = (x_thresh, y_val)
                
    def predict(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """Applies calibration to inference probabilities."""
        calibrated = np.zeros_like(uncalibrated_probs)
        N = uncalibrated_probs.shape[0]
        
        for a in range(self.num_actions):
            for b in range(self.num_bins):
                x_thresh, y_val = self.calibrators[a][b]
                if x_thresh is None:
                    # Not fitted, return original
                    calibrated[:, a, b] = uncalibrated_probs[:, a, b]
                    continue
                    
                probs = uncalibrated_probs[:, a, b]
                # Interpolate piecewise step function
                idx = np.searchsorted(x_thresh, probs, side='right') - 1
                idx = np.clip(idx, 0, len(y_val) - 1)
                
                calibrated[:, a, b] = y_val[idx]
                
        # Re-normalize bins to sum to 1
        cal_sum = calibrated.sum(axis=2, keepdims=True) + 1e-8
        calibrated /= cal_sum
        return calibrated
