import numpy as np
import torch
import torch.nn.functional as F


def dynamic_threshold(errors, window=64):
    """滑动窗口动态阈值（使用中位数和IQR）"""
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
    # 确保 pred_errors 是单维
    if pred_errors.ndim > 1:
        pred_errors = np.mean(pred_errors, axis=1)  # 多特征取平均
    """多指标融合（预测误差 + 重构误差）"""
    pred_norm = (pred_errors - np.min(pred_errors)) / (np.ptp(pred_errors) + 1e-8)
    recon_norm = (recon_errors - np.min(recon_errors)) / (np.ptp(recon_errors) + 1e-8)
    return weights[0] * pred_norm + weights[1] * recon_norm

def calc_anomaly_level(combined_scores, thresholds):
    """生成告警等级（0-正常，1-警告，2-严重）"""
    levels = []
    for score, thresh in zip(combined_scores, thresholds):
        if score > thresh * 1.5:
            levels.append(2)
        elif score > thresh:
            levels.append(1)
        else:
            levels.append(0)
    return np.array(levels)

def loss_func(y_pred, y_true, x_recon, x_orig, mu, logvar):
    # 预测任务RMSE
    rmse_loss = torch.sqrt(F.mse_loss(y_pred, y_true))

    # 重构任务损失（展平为 [B*N, slide_win]）
    x_recon_flat = x_recon.view(-1, x_recon.size(-1))  # [batch*node_num, input_dim]
    x_orig_flat = x_orig.view(-1, x_orig.size(-1))     # [batch*node_num, input_dim]
    recon_loss = F.mse_loss(x_recon_flat, x_orig_flat)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x_orig.size(0) * x_orig.size(1))   # 归一化KL散度

    # 总损失（权重可调整）
    total_loss = 0.5 * rmse_loss + 0.4 * recon_loss + 0.1 * kl_loss
    return total_loss, {
        'total': total_loss.item(),
        'rmse': rmse_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item()
    }