# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from datasets.TimeDataset import TimeDataset
from evaluate import fuse_anomaly_scores, \
    dynamic_threshold, calc_anomaly_level
from models.stgcn import STGCN
from test import test
from train import train
from util.env import get_device, set_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


class Main():

    def __init__(self, train_config, env_config):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(rf'data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(rf'data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride']
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)


        self.model = STGCN(edge_index_sets, len(feature_map),
                           dim=train_config['dim'],
                           input_dim=train_config['slide_win'],
                           out_layer_num=train_config['out_layer_num'],
                           topk=train_config['topk']
                           ).to(self.device)


    def run(self, progress_callback=None, should_continue=None):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                progress_callback=progress_callback,  # 新增回调参数
                should_continue=should_continue  # 新增中断检查
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path, weights_only=True))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)


    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    # main.py 的 Main 类中修改 get_score 方法

    def get_score(self, test_result, val_result):
        # 解析多任务输出
        # test_result 格式：[test_pred, test_recon, test_labels]
        test_pred, test_recon, test_labels = test_result
        val_pred, val_recon, val_labels = val_result  # 验证集结果（可选使用）

        # ================== 异常评分计算 ==================
        # 1. 计算预测误差（按样本维度）
        # test_pred: [num_samples, num_features]
        # test_labels: [num_samples]（假设为二进制标签）
        # 注意：需确保预测结果与标签的维度对齐
        if test_pred.shape[0] != test_labels.shape[0]:
            raise ValueError("预测结果与标签的样本数不一致")

        # 计算每个样本的预测误差（平均所有特征）
        # 获取真实值（取最后一个时间步）
        ground_truth = self.test_dataset.x[:, :, -1].numpy()
        # 计算预测误差（逐节点计算）
        pred_errors = np.abs(test_pred - ground_truth)  # [num_samples, node_num]
        # 按样本平均所有节点的误差
        pred_errors = np.max(pred_errors, axis=1)  # [num_samples]

        # 2. 计算重构误差（按样本维度）
        orig_data = self.test_dataset.x.numpy()
        test_recon = test_recon.reshape(orig_data.shape)  # 直接对齐原始数据维度
        # 计算逐时间步的 MSE
        mse_errors = np.mean((test_recon - orig_data) ** 2, axis=(1, 2))
        # 计算逐节点的动态加权误差（关注波动较大的节点）
        node_std = np.std(orig_data, axis=2)  # 形状 (983, 51)
        node_weights = node_std / np.sum(node_std, axis=1, keepdims=True)
        node_weights_expanded = node_weights[:, :, np.newaxis]  # 扩展维度 -> (983, 51, 1)
        weighted_errors = np.sum(np.abs(test_recon - orig_data) * node_weights_expanded, axis=(1, 2))
        # 综合误差
        recon_errors = 0.7 * mse_errors + 0.3 * weighted_errors

        # 3. 动态阈值（基于预测误差的滑动窗口）
        dyn_thresh = dynamic_threshold(pred_errors, window=64)  # 窗口大小可调

        # 4. 综合评分（融合预测和重构误差）
        combined_scores = fuse_anomaly_scores(
            pred_errors,
            recon_errors,
            weights=(0.8, 0.2)  # 权重可调整
        )

        # 5. 生成告警等级
        alert_levels = calc_anomaly_level(combined_scores, dyn_thresh)

        # ================== 评估指标计算 ==================
        # 将连续评分转换为二进制预测（基于动态阈值）
        binary_pred = (combined_scores > dyn_thresh).astype(int)

        # 确保标签为二进制（0/1）
        binary_labels = test_labels.astype(int)
        if np.unique(binary_labels).size > 2:
            raise ValueError("标签应为二进制（0或1）")

        # 计算指标
        f1 = f1_score(binary_labels, binary_pred)
        precision = precision_score(binary_labels, binary_pred)
        recall = recall_score(binary_labels, binary_pred)

        # 保存结果
        self.final_scores = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'alerts': alert_levels.tolist()  # 转换为列表便于存储
        }


    def get_ground_truth(self):
        """获取传感器数据的真实值（滑动窗口的下一时间步）"""
        # 原始数据形状 [num_samples, slide_win]
        orig_data = self.test_dataset.x.numpy()
        # 真实值为每个窗口的最后一个时间步 [num_samples, 1]
        return orig_data[:, -1].reshape(-1, 1)  # 与预测输出维度对齐


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m_%d_%H-%M-%S')
        datestr = self.datestr          

        paths = [
            rf'pretrained/{dir_path}/best_{datestr}.pt',
            rf'results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths





