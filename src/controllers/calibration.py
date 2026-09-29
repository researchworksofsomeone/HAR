import numpy as np

class IsotonicCalibrator:
    """Pure-Python PAVA (Pool Adjacent Violators Algorithm) for Isotonic Regression.
    Minimizes dependencies and stays lightweight."""
    def __init__(self):
        self.X_ = None
        self.y_ = None
        
    def fit(self, X, y):
        # Sort X and y by X
        order = np.argsort(X)
        x_sorted = X[order]
        y_sorted = y[order]
        
        n = len(y_sorted)
        val = np.array(y_sorted, dtype=np.float64)
        weight = np.ones(n, dtype=np.float64)
        
        # PAVA
        i = 1
        while i < n:
            while i > 0 and val[i-1] > val[i]:
                # Merge
                new_weight = weight[i-1] + weight[i]
                new_val = (val[i-1] * weight[i-1] + val[i] * weight[i]) / new_weight
                val[i-1] = new_val
                weight[i-1] = new_weight
                
                # Shift remainder left
                val = np.delete(val, i)
                weight = np.delete(weight, i)
                x_sorted = np.delete(x_sorted, i)
                i -= 1
                n -= 1
                if i == 0:
                    break
            i += 1
        
        self.X_ = x_sorted
        self.y_ = val
        return self
        
    def predict(self, X):
        if self.X_ is None:
            raise ValueError("Calibrator not fitted.")
        return np.interp(X, self.X_, self.y_)

class JADECalibrator:
    def __init__(self, num_actions, num_bins=3):
        self.calibrators = {}
        for a in range(num_actions):
            for b in range(num_bins):
                self.calibrators[(a, b)] = IsotonicCalibrator()
                
    def fit(self, action_probs, true_labels, actions):
        """Fit independent PAVA for each action and bin."""
        num_actions = len(self.calibrators) // 3
        for a in range(num_actions):
            mask = (actions == a)
            if not mask.any(): continue
            for b in range(3):
                y_true_bin = (true_labels[mask] == b).astype(float)
                prob_bin = action_probs[mask, a, b]
                self.calibrators[(a, b)].fit(prob_bin, y_true_bin)
                
    def calibrate(self, action_probs):
        """action_probs: (N, num_actions, num_bins)"""
        calibrated = np.zeros_like(action_probs)
        N, num_actions, num_bins = action_probs.shape
        for a in range(num_actions):
            for b in range(num_bins):
                calibrated[:, a, b] = self.calibrators[(a, b)].predict(action_probs[:, a, b])
        return calibrated
