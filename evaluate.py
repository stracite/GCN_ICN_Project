import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from util.data import *


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)
    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]
    labels = np_test_result[2, :, 0].tolist()
    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i]
        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)
        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))
    return all_scores, all_normals

def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result)
    all_scores = np.max(full_scores, axis=0)
    return all_scores

def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res[0], test_res[2]  # test_res 格式 [pred, recon, labels]
    val_predict, val_gt = val_res[0], val_res[2]
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2
    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
    return smoothed_err_scores


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]
    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)
    total_topk_err_scores = []
    for i, indexs in enumerate(topk_indices):
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )
        total_topk_err_scores.append(sum_score)
    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)
    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    thresold = np.max(normal_scores)
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    return f1, pre, rec, auc_score, thresold

def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]
    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    return max(final_topk_fmeas), pre, rec, auc_score, thresold


def dynamic_threshold(errors, window=64):
    """滑动窗口动态阈值（改进版使用中位数和IQR）"""
    thresholds = []
    for i in range(len(errors)):
        start = max(0, i - window // 2)
        end = min(len(errors), i + window // 2)
        window_errors = errors[start:end]

        # 异常值处理：移除窗口内前5%的极端值
        if len(window_errors) > 10:
            q95 = np.percentile(window_errors, 95)
            window_errors = window_errors[window_errors <= q95]

        # 使用中位数和IQR替代均值和标准差（更鲁棒）
        median = np.median(window_errors)
        q1 = np.percentile(window_errors, 25)
        q3 = np.percentile(window_errors, 75)
        iqr = max(q3 - q1, 1e-6)  # 防止除零

        # 调整阈值系数（原3σ对应约2.22iqr）
        thresholds.append(median + 1.5 * iqr)  # 系数可调

    return np.array(thresholds)

def fuse_anomaly_scores(pred_errors, recon_errors, weights=(0.6, 0.4)):
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