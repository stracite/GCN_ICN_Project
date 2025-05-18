# util functions about data
import numpy as np
from scipy.stats import rankdata, iqr
from sklearn.metrics import f1_score

# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_scores) - len(scores))
    # print(padding_list)
    if len(padding_list) > 0:
        scores = padding_list + scores
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)
        fmeas[i] = f1_score(true_scores, cur_pred)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]
    if return_thresold:
        return fmeas, thresholds
    return fmeas

def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)
    return err_median, err_iqr