import numpy as np

class FeatureExtractor:
    """
    Extracts the strictly causal 28-dimensional state vector phi_t for JADE.
    Uses NO extra forward passes; operates entirely on ring buffers and 
    existing inference activations.
    """
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.prev_preds = None
        self.ticks_since_adapt = 0
        self.cum_entropy_change = 0.0
        self.last_mean_entropy = 0.0
        
        # Stateful trackers for slopes and EMA
        self.ema_entropy = 0.0
        self.ema_displacement = 0.0
        self.ema_margin = 0.0
        self.ema_agreement = 0.0
        
        # Normalization EMAs
        self.feat_mean = np.zeros(28, dtype=np.float32)
        self.feat_var = np.ones(28, dtype=np.float32)
        self.norm_t = 0
        
    def _ema_slope(self, current: float, ema_val: float, alpha: float = 0.1) -> tuple[float, float]:
        """Returns (slope, new_ema)"""
        if ema_val == 0.0:
            return 0.0, current
        slope = current - ema_val
        new_ema = (1 - alpha) * ema_val + alpha * current
        return slope, new_ema

    def extract(
        self,
        logits: np.ndarray,          # (N, C)
        penultimate_feats: np.ndarray, # (N, hidden_dim)
        prototypes: np.ndarray,      # (C, hidden_dim)
        bn_stats: list[dict],        # list of 4 dicts with 'ring_mean', 'ring_var', 'run_mean', 'run_var'
        act_scales: list[float],     # list of 4 floats (ring activation scales)
        src_act_scales: list[float], # list of 4 floats (source activation scales)
        tick: int,
        D: int = 32
    ) -> np.ndarray:
        
        features = np.zeros(28, dtype=np.float32)
        
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        entropies = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # 1-4: Entropy stats (last N windows)
        mean_ent = float(np.mean(entropies))
        features[0] = mean_ent
        features[1] = float(np.std(entropies))
        features[2] = float(np.max(entropies))
        
        slope_ent, self.ema_entropy = self._ema_slope(mean_ent, self.ema_entropy)
        features[3] = slope_ent
        
        # 5-6: Confidence
        mean_margin = float(np.mean(margins))
        features[4] = mean_margin
        features[5] = float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -3])) if self.num_classes >= 3 else mean_margin
        
        # 7: Prediction change rate
        curr_preds = np.argmax(probs, axis=1)
        if self.prev_preds is not None and len(self.prev_preds) == len(curr_preds):
            features[6] = float(np.mean(curr_preds != self.prev_preds))
        else:
            features[6] = 0.0
        self.prev_preds = curr_preds
        
        # 8: Output distribution entropy
        class_counts = np.bincount(curr_preds, minlength=self.num_classes)
        p_class = class_counts / (np.sum(class_counts) + 1e-8)
        features[7] = float(-np.sum(p_class * np.log(p_class + 1e-8)))
        
        # 9-12: BN Displacement (L2 norm)
        mean_disp = 0.0
        for i in range(4):
            if i < len(bn_stats):
                st = bn_stats[i]
                diff = st['ring_mean'] - st['run_mean']
                disp = float(np.linalg.norm(diff))
                features[8 + i] = disp
                mean_disp += disp / 4.0
                
        # 13-16: Activation scale ratio
        for i in range(4):
            if i < len(act_scales) and i < len(src_act_scales):
                features[12 + i] = float(act_scales[i] / (src_act_scales[i] + 1e-8))
                
        # 17: Ticks since last non-NOOP (log-scaled)
        self.ticks_since_adapt += 1
        features[16] = float(np.log1p(self.ticks_since_adapt))
        
        # 18: Cumulative entropy change
        self.cum_entropy_change += (mean_ent - self.last_mean_entropy)
        self.last_mean_entropy = mean_ent
        features[17] = self.cum_entropy_change
        
        # 19: Low-confidence window fraction
        features[18] = float(np.mean(sorted_probs[:, -1] < 0.6))
        
        # 20: Prototype deviation
        # Normalize features and prototypes for cosine distance
        norm_feats = penultimate_feats / (np.linalg.norm(penultimate_feats, axis=1, keepdims=True) + 1e-8)
        norm_protos = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        
        # Cosine distance = 1 - cosine_similarity
        cos_sim = np.dot(norm_feats, norm_protos.T) # (N, C)
        # Minimum distance to any prototype per sample, then mean across batch
        min_dists = 1.0 - np.max(cos_sim, axis=1)
        mean_proto_dev = float(np.mean(min_dists))
        features[19] = mean_proto_dev
        
        # 21: Softmax <-> Prototype-NN agreement
        proto_preds = np.argmax(cos_sim, axis=1)
        agreement = float(np.mean(curr_preds == proto_preds))
        features[20] = agreement
        
        # 22: Dummy Page-Hinkley (tracked externally or simulated here)
        features[21] = features[3] * 10.0 # Mock PH score correlating with entropy slope
        
        # 23: Dummy ADWIN width (mocked as correlating with margin variance)
        features[22] = float(np.std(margins)) * 5.0
        
        # 24: Tick phase
        features[23] = float((tick % D) / float(D))
        
        # 25-28: Lagged trends (EMA slopes)
        slope_disp, self.ema_displacement = self._ema_slope(mean_disp, self.ema_displacement)
        features[24] = slope_disp
        
        slope_marg, self.ema_margin = self._ema_slope(mean_margin, self.ema_margin)
        features[25] = slope_marg
        
        slope_agree, self.ema_agreement = self._ema_slope(agreement, self.ema_agreement)
        features[26] = slope_agree
        
        features[27] = slope_ent # Redundant with 3 for completion or lag variant
        
        # Handle exact zero or nan locally
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # EMA Z-Scoring Normalization
        self.norm_t += 1
        alpha_norm = 2.0 / (min(self.norm_t, 1000) + 1.0)
        
        # Update mean and variance
        diff = features - self.feat_mean
        self.feat_mean += alpha_norm * diff
        self.feat_var = (1 - alpha_norm) * (self.feat_var + alpha_norm * diff**2)
        
        # Apply z-score
        std = np.sqrt(self.feat_var) + 1e-8
        norm_features = (features - self.feat_mean) / std
        
        # Hard clip extreme outliers to prevent network blowup
        norm_features = np.clip(norm_features, -5.0, 5.0)
        
        return norm_features
        
    def reset_adaptation_state(self):
        """Called when a non-NOOP action is taken."""
        self.ticks_since_adapt = 0
        self.cum_entropy_change = 0.0
