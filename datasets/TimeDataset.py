import torch
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    """时间序列数据集处理类，用于生成图神经网络训练所需的滑动窗口样本

    Args:
        raw_data (array-like): 原始时间序列数据，形状应为[特征数+1, 时间步长]，最后一行为标签
        edge_index (Tensor): 图结构的边索引，形状为[2, 边数]
        mode (str): 模式选择，可选'train'或其他值用于区分训练/验证模式
        config (dict, optional): 配置参数字典，包含窗口大小等超参数
    """
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data
        self.config = config if config is not None else {}
        self.edge_index = edge_index
        self.mode = mode
        data = raw_data[:-1]
        labels = raw_data[-1]

        # 数据转换与预处理流程
        # 转换为张量 [节点数, 时间步长]
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        # 预处理：噪声过滤 + Min-Max标准化
        window_size = self.config.get("window_size", 5)
        data = self.apply_moving_average(data, window_size)
        data = self.apply_minmax(data)
        self.feature, self.y, self.labels = self.process(data, labels)

    def apply_moving_average(self, data, window_size):
        """滑动窗口均值滤波（保持不变）"""
        """对时间序列数据进行滑动窗口均值滤波
        Args:
            data (Tensor): 输入数据，形状为[节点数, 时间步长]
            window_size (int): 滑动窗口大小
        Returns:
            Tensor: 平滑后的数据，保持原始形状
        """
        data = data.unsqueeze(1)  # [nodes, 1, time]
        kernel = torch.ones(1, 1, window_size).double() / window_size
        pad = window_size // 2
        padded = torch.nn.functional.pad(data, (pad, pad), mode='reflect')
        smoothed = torch.nn.functional.conv1d(padded, kernel, padding=0)
        return smoothed.squeeze(1)  # [nodes, time]

    def apply_minmax(self, data):
        """Min-Max标准化（按节点独立处理）"""
        """对每个节点的时间序列进行独立的最小-最大归一化
        Args:
            data (Tensor): 输入数据，形状为[节点数, 时间步长]
        Returns:
            Tensor: 归一化后的数据，范围[0,1]
        """
        min_vals = torch.min(data, dim=1, keepdim=True).values
        max_vals = torch.max(data, dim=1, keepdim=True).values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 避免除零
        return (data - min_vals) / range_vals

    def process(self, data, labels):
        """滑动窗口"""
        """滑动窗口样本生成方法
        Args:
            data (Tensor): 预处理后的特征数据，形状[节点数, 时间步长]
            labels (Tensor): 原始标签数据，形状[时间步长]
        Returns:
            tuple: (特征序列, 目标值, 标签) 三元组
        """
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win = self.config.get("slide_win", 10)
        slide_stride = self.config.get("slide_stride", 1)
        is_train = self.mode == 'train'
        node_num, total_time_len = data.shape
        # 根据模式选择不同的滑动策略
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append([labels[i]])
        # 将列表转换为张量
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()
        return x, y, labels

    def __getitem__(self, x):
        """获取单个样本
        Returns:
            tuple: (特征窗口, 目标值, 标签) 三元组
        """
        feature = self.feature[x].double()
        y = self.y[x].double()
        label = self.labels[x].double()
        # print("feature:",feature[0])
        # print("y:",y)
        # print("label:",label)
        return feature, y, label

    def __len__(self):
        """获取数据集长度"""
        return len(self.feature)