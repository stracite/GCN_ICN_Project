# utils/data_processor.py
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


class TemporalDataProcessor:
    """时序数据处理管道
    Args:
        window_size (int): 时间窗口大小 (default: 24)
        stride (int): 滑动步长 (default: 1)
        smooth_window (int): 平滑窗口大小 (default: 5)
        poly_order (int): 多项式阶数 (default: 3)
    """

    def __init__(self, window_size=24, stride=1, smooth_window=5, poly_order=3):
        self.window_size = window_size
        self.stride = stride
        self.smooth_window = smooth_window
        self.poly_order = poly_order
        self.scaler = StandardScaler()

    def process(self, raw_data):
        """完整数据处理流程
        Args:
            raw_data (np.ndarray): 原始数据 (samples, nodes, features)
        Returns:
            np.ndarray: 处理后的数据 (samples, nodes, time_steps, features)
        """
        # 数据平滑
        smoothed = self._apply_smoothing(raw_data)

        # 标准化
        normalized = self._normalize(smoothed)

        # 时间窗口划分
        windowed = self._create_windows(normalized)
        return windowed

    def _apply_smoothing(self, data):
        """应用Savitzky-Golay滤波"""
        return np.apply_along_axis(
            lambda x: savgol_filter(x, self.smooth_window, self.poly_order),
            axis=0,
            arr=data
        )

    def _normalize(self, data):
        """标准化处理"""
        orig_shape = data.shape
        data_2d = data.reshape(-1, orig_shape[-1])
        normalized = self.scaler.fit_transform(data_2d)
        return normalized.reshape(orig_shape)

    def _create_windows(self, data):
        """创建时间窗口"""
        num_samples = data.shape[0]
        windows = []

        for start in range(0, num_samples - self.window_size, self.stride):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)

        return np.array(windows).transpose(0, 2, 1, 3)  # (samples, nodes, time, features)
