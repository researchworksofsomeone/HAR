import numpy as np
import collections

class PageHinkley:
    """Lightweight Page-Hinkley change detector."""
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 1 - 0.0001):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.n = 0
        self.sum = 0.0
        self.m_t = 0.0
        self.M_t = 0.0
        
    def update(self, x: float) -> bool:
        self.n += 1
        self.x_mean = self.x_mean + (x - self.x_mean) / self.n
        
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)
        
        if self.sum < self.m_t:
            self.m_t = self.sum
            
        # Detect drift (increase in signal)
        drift = (self.sum - self.m_t) > self.threshold
        if drift:
            self.reset()
        return drift
        
    def reset(self):
        self.n = 0
        self.x_mean = 0.0
        self.sum = 0.0
        self.m_t = 0.0
        self.M_t = 0.0

class SimpleADWIN:
    """
    Minimalist change detector simulating ADWIN mechanics 
    by comparing means of recent sub-windows.
    """
    def __init__(self, window_size: int = 100, threshold: float = 0.2):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = collections.deque(maxlen=window_size)
        
    def update(self, x: float) -> bool:
        self.buffer.append(x)
        if len(self.buffer) == self.window_size:
            half = self.window_size // 2
            arr = list(self.buffer)
            mean1 = np.mean(arr[:half])
            mean2 = np.mean(arr[half:])
            diff = abs(mean1 - mean2)
            if diff > self.threshold:
                self.buffer.clear()
                return True
        return False

class TriggerController:
    """Evaluates proxy signals to trigger adaptation actions."""
    def __init__(self, thresholds: dict):
        self.tau_E = thresholds.get("tau_E", 1.5)
        self.tau_S = thresholds.get("tau_S", 0.5)
        
        # Stateful detectors
        self.ph = PageHinkley(threshold=thresholds.get("ph_threshold", 5.0))
        self.adwin = SimpleADWIN(threshold=thresholds.get("adwin_threshold", 0.2))
        
        self.surprise_consecutive_ticks = 0
        
    def entropy_trigger(self, entropies: np.ndarray) -> tuple[bool, str]:
        """Fires if mean predictive entropy > tau_E."""
        mean_ent = np.mean(entropies)
        fire = mean_ent > self.tau_E
        return fire, "a3_TENT_k" if fire else "a0_NOOP"
        
    def surprise_trigger(self, prototype_distances: np.ndarray) -> tuple[bool, str]:
        """Fires if cosine distance to source class prototypes > tau_S."""
        mean_dist = np.mean(prototype_distances)
        if mean_dist > self.tau_S:
            self.surprise_consecutive_ticks += 1
            if self.surprise_consecutive_ticks >= 2:
                self.surprise_consecutive_ticks = 0 # reset after escalation
                return True, "a3_TENT_k"
            return True, "a1_BN_RECAL"
        else:
            self.surprise_consecutive_ticks = 0
            return False, "a0_NOOP"
            
    def shift_trigger(self, bn_displacement: float) -> tuple[bool, str]:
        """ADWIN on BN-displacement series."""
        fire = self.adwin.update(bn_displacement)
        return fire, "a3_TENT_k" if fire else "a0_NOOP"
        
    def ph_drift(self, entropy: float) -> tuple[bool, str]:
        """Page-Hinkley on predictive entropy series."""
        fire = self.ph.update(entropy)
        return fire, "a3_TENT_k" if fire else "a0_NOOP"
