import torch
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data
        self.config = config if config is not None else {}
        self.edge_index = edge_index
        self.mode = mode
        x_data = raw_data[:-1]
        labels = raw_data[-1]
        data = x_data

        # 转换为张量 [节点数, 时间步长]
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        # 预处理：噪声过滤 + Min-Max标准化
        window_size = self.config.get("window_size", 5)
        data = self.apply_moving_average(data, window_size)
        data = self.apply_minmax(data)

        self.x, self.y, self.labels = self.process(data, labels)

    def apply_moving_average(self, data, window_size):
        """滑动窗口均值滤波（保持不变）"""
        data = data.unsqueeze(1)  # [nodes, 1, time]
        kernel = torch.ones(1, 1, window_size).double() / window_size
        pad = window_size // 2
        padded = torch.nn.functional.pad(data, (pad, pad), mode='reflect')
        smoothed = torch.nn.functional.conv1d(padded, kernel, padding=0)
        return smoothed.squeeze(1)  # [nodes, time]

    def apply_minmax(self, data):
        """Min-Max标准化（按节点独立处理，无参数传递）"""
        min_vals = torch.min(data, dim=1, keepdim=True).values
        max_vals = torch.max(data, dim=1, keepdim=True).values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 避免除零
        return (data - min_vals) / range_vals

    def process(self, data, labels):
        """滑动窗口分割（保持不变）"""
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win = self.config.get("slide_win", 10)
        slide_stride = self.config.get("slide_stride", 1)
        is_train = self.mode == 'train'
        node_num, total_time_len = data.shape
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append([labels[i]] * node_num)
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()
        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()
        return feature, y, label, edge_index

    def __len__(self):
        return len(self.x)