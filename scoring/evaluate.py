import numpy as np


def dynamic_threshold(errors, window=64):
    """滑动窗口动态阈值（使用中位数和IQR）

    Args:
        errors : 输入误差序列，形状为[num_samples]
        window : 滑动窗口大小，默认为64

    Returns:
        array: 动态阈值序列，形状与errors相同

    实现逻辑：
        1. 对每个数据点使用中心对称滑动窗口
        2. 移除窗口内前5%的极端值（当窗口数据>10时生效）
        3. 基于中位数和四分位距(IQR)计算动态阈值
    """
    thresholds = []
    for i in range(len(errors)):
        start = max(0, i - window // 2)
        end = min(len(errors), i + window // 2)
        window_errors = errors[start:end]

        # 异常值处理：移除窗口内前5%的极端值
        if len(window_errors) > 10:
            q95 = np.percentile(window_errors, 95)
            window_errors = window_errors[window_errors <= q95]

        # 使用中位数和IQR替代均值和标准差
        median = np.median(window_errors)
        q1 = np.percentile(window_errors, 25)
        q3 = np.percentile(window_errors, 75)
        iqr = max(q3 - q1, 1e-6)  # 防止除零

        # 调整阈值系数（原3σ对应约2.22iqr）
        thresholds.append(median + 2.22 * iqr)  # 系数可调

    return np.array(thresholds)


def fuse_anomaly_scores(pred_errors, recon_errors, weights):
    """多指标异常分数融合器


    Args:
        pred_errors : 预测误差序列，形状为[num_samples]
        recon_errors : 重构误差序列，形状为[num_samples]
        weights : 融合权重，格式为(pred_weight, recon_weight)
    Returns:
        array: 融合后的综合异常分数，形状为[num_samples]

    处理流程：
        1. 对多维预测误差进行平均化处理
        2. 对两个误差序列分别进行归一化
        3. 按权重线性组合得到最终分数
    """
    # 确保 pred_errors 是单维
    if pred_errors.ndim > 1:
        pred_errors = np.mean(pred_errors, axis=1)  # 多特征取平均
    """多指标融合（预测误差 + 重构误差）"""
    pred_norm = (pred_errors - np.min(pred_errors)) / (np.ptp(pred_errors) + 1e-8)
    recon_norm = (recon_errors - np.min(recon_errors)) / (np.ptp(recon_errors) + 1e-8)
    return weights[0] * pred_norm + weights[1] * recon_norm

def calc_anomaly_level(combined_scores, thresholds):
    """异常等级分类器

    Args:
        combined_scores : 综合异常分数，形状为[num_samples]
        thresholds : 动态阈值序列，形状为[num_samples]

    Returns:
        array: 异常等级序列，0-正常，1-警告，2-严重

    分级规则：
        - 超过阈值1.5倍：2级（严重）
        - 超过阈值但不足1.5倍：1级（警告）
        - 低于阈值：0级（正常）
    """
    levels = []
    for score, thresh in zip(combined_scores, thresholds):
        if score > thresh * 1.5:
            levels.append(2)
        elif score > thresh:
            levels.append(1)
        else:
            levels.append(0)
    return np.array(levels)
