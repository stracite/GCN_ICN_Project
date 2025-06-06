# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from datasets.SpaceDataset import get_feature_map, get_fc_graph_struc, build_loc_net, construct_data
from datasets.TimeDataset import TimeDataset
from models.stgcn import STGCN
from models.train import train
from scoring.evaluate import fuse_anomaly_scores, \
    dynamic_threshold, calc_anomaly_level
from scoring.test import test
from util.env import get_device, set_device


class Main():
    """主运行类，负责模型训练和测试的整体流程控制
    Args:
        train_config (dict): 训练配置参数，包含slide_win、batch等参数
        env_config (dict): 环境配置参数，包含设备设置、数据集路径等
    """

    def __init__(self, train_config, env_config):
        """类初始化方法，完成数据加载、模型构建等准备工作"""
        # 基础配置存储
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        # 设备设置
        set_device(env_config['device'])
        self.device = get_device()
        # 数据加载与预处理
        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(rf'data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(rf'data/{dataset}/test.csv', sep=',', index_col=0)
        # 数据清洗（移除攻击列）
        train, test = train_orig, test_orig
        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        # 特征图与图结构获取
        feature_list = get_feature_map(dataset)
        struc_map = get_fc_graph_struc(feature_list)
        # 构建图结构数据
        fc_edge_index = build_loc_net(struc_map, feature_list, feature_map=feature_list)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.feature_list = feature_list
        # 数据集构建
        train_dataset_indata = construct_data(train, feature_list, labels=0)
        test_dataset_indata = construct_data(test, feature_list, labels=test.attack.tolist())

        # 时间序列数据集配置
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride']
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_config['batch'], shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)

        # 图结构集合初始化
        edge_index_sets = [fc_edge_index]

        # 模型初始化
        self.model = STGCN(edge_index_sets, len(feature_list),
                           dim=train_config['dim'],
                           slide_win=train_config['slide_win'],
                           mlp_layer_num=train_config['mlp_layer_num'],
                           topk=train_config['topk']
                           ).to(self.device)


    def run(self, progress_callback=None, should_continue=None):
        """执行完整训练测试流程

        Args:
            progress_callback (function, optional): 训练进度回调函数
            should_continue (function, optional): 训练中断检查函数
        """
        # 模型路径处理
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            # 执行训练流程
            self.train_log = train(self.model, model_save_path, 
                config = self.train_config,
                train_dataloader=self.train_dataloader,
                progress_callback=progress_callback,  # 新增回调参数
                should_continue=should_continue  # 新增中断检查
            )
        
        # 测试阶段
        self.model.load_state_dict(torch.load(model_save_path, weights_only=True))
        best_model = self.model.to(self.device)
        _, self.test_result = test(best_model, self.test_dataloader)
        self.get_score(self.test_result)

    def get_score(self, test_result):
        """计算评估指标和异常分数

        Args:
            test_result (tuple): 测试结果元组 (预测值, 重构值, 真实标签)
        """
        # 结果解析
        # test_result 格式：[test_pred, test_recon, test_labels]
        test_pred, test_recon, test_labels = test_result

        # ================== 异常评分计算 ==================
        # 1. 计算预测误差（按样本维度）
        # test_pred: [num_samples, num_features]
        # test_labels: [num_samples]（假设为二进制标签）
        if test_pred.shape[0] != test_labels.shape[0]:
            raise ValueError("预测结果与标签的样本数不一致")

        # 计算每个样本的预测误差
        ground_truth = self.test_dataset.feature[:, :, -1].numpy()
        # 计算预测误差（逐节点计算）
        pred_errors = np.abs(test_pred - ground_truth)  # [num_samples, node_num]
        # 按样本取节点最大误差
        pred_errors = np.max(pred_errors, axis=1)  # [num_samples]

        # 2. 计算重构误差（按样本维度）
        orig_data = self.test_dataset.feature.numpy()
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
            weights=(0.1, 0.9)  # 权重可调整
        )
        # print("动态阈值",dyn_thresh)
        # print("异常评分",combined_scores)
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

    def get_save_path(self, feature_name=''):
        """生成模型保存路径

        Args:
            feature_name (str, optional): 特征名称

        Returns:
            list: 包含模型路径和结果路径的列表
        """

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m_%d_%H-%M-%S')
        datestr = self.datestr          

        paths = [
            rf'pretrained/{dir_path}/best_{datestr}.pt',
            rf'results/{dir_path}/{datestr}.csv',
        ]
        # 目录创建保障
        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths





