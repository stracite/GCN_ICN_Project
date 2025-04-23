# detect.py
import numpy as np
from scipy.stats import gamma


class IndustrialAnomalyDetector:
    """工业异常检测器
    Args:
        window_size (int): 滑动窗口大小 (default: 60)
        alpha (float): 置信度阈值 (default: 0.99)
        error_weights (tuple): 误差权重 (pred, recon) (default: (0.6, 0.4))
    """

    def __init__(self, window_size=60, alpha=0.99, error_weights=(0.6, 0.4)):
        self.window_size = window_size
        self.alpha = alpha
        self.weights = error_weights
        self.error_buffer = []
        self.gamma_params = None

    def update(self, pred_errors, recon_errors):
        """更新误差缓冲区
        Args:
            pred_errors (np.ndarray): 预测误差 (samples, nodes, features)
            recon_errors (np.ndarray): 重构误差
        """
        combined = self._combine_errors(pred_errors, recon_errors)
        self.error_buffer.extend(combined.flatten().tolist())

        # 保持窗口大小
        if len(self.error_buffer) > self.window_size:
            self.error_buffer = self.error_buffer[-self.window_size:]

        self._fit_distribution()

    def detect(self, pred_errors, recon_errors):
        """执行异常检测
        Returns:
            tuple: (异常标记, 综合得分)
        """
        scores = self._combine_errors(pred_errors, recon_errors)
        if self.gamma_params is None:
            return np.zeros_like(scores), scores

        threshold = gamma.ppf(self.alpha, *self.gamma_params)
        anomalies = scores > threshold
        return anomalies, scores