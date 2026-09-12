import numpy as np
from collections import deque

class TubeletGatingConcept:
    """
    Reference module illustrating the stationary cashier variance check
    and the dual-criteria soft anomaly recovery formula.
    """
    def __init__(self, history_len=64, stationary_threshold=15.0):
        self.history_len = history_len
        self.stationary_threshold = stationary_threshold
        self.centroids = deque(maxlen=history_len)
        self.action_probabilities = deque(maxlen=history_len)

    def update_centroid(self, cx: float, cy: float) -> bool:
        """
        Stationary cashier variance check.
        If the maximum displacement from the current position over the history
        is less than the threshold (e.g. 15px), the person is considered stationary.
        """
        self.centroids.append((cx, cy))
        if len(self.centroids) < self.history_len:
            return False # Not enough history to confirm stationary
            
        pts = np.array(self.centroids)
        # Max displacement from current position over the last 64 frames
        max_disp = float(np.max(np.linalg.norm(pts - pts[-1], axis=1)))
        return max_disp < self.stationary_threshold

    def evaluate_anomaly_recovery(self, new_probability: float) -> bool:
        """
        Dual-criteria soft anomaly recovery formula:
        (max(probs) >= 0.18) or (sum(probs) >= 0.45)
        This is a conceptual illustration of buffering predictions to smooth anomalies.
        """
        self.action_probabilities.append(new_probability)
        if len(self.action_probabilities) == 0:
            return False
            
        probs = np.array(self.action_probabilities)
        max_prob = np.max(probs)
        sum_prob = np.sum(probs)
        
        return (max_prob >= 0.18) or (sum_prob >= 0.45)
