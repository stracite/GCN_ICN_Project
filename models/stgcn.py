# models/stgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import GCN, DilatedTCN


class STGCN(nn.Module):
    """时空图卷积网络主模型
    Args:
        node_features (int): 每个节点的特征维度
        hidden_dim (int): 隐藏层维度
        adj_matrix (torch.Tensor): 邻接矩阵
        time_window (int): 时间窗口大小
        num_nodes (int): 节点数量
        num_gcn_layers (int): GCN层数 (default: 2)
        num_tcn_layers (int): TCN层数 (default: 2)
        dropout (float): Dropout概率 (default: 0.2)
    """

    def __init__(self, node_features, hidden_dim, adj_matrix,
                 time_window, num_nodes, num_gcn_layers=2,
                 num_tcn_layers=2, dropout=0.2):
        super().__init__()
        self.register_buffer('adj', adj_matrix)
        self.num_nodes = num_nodes
        self.time_window = time_window

        # 空间特征提取模块
        self.gcn_layers = nn.ModuleList([
            GCN(node_features if i == 0 else hidden_dim,
                hidden_dim,
                adj_matrix)
            for i in range(num_gcn_layers)
        ])

        # 时间特征提取模块
        self.tcn_layers = nn.ModuleList([
            DilatedTCN(hidden_dim, hidden_dim, dilation=2 ** i)
            for i in range(num_tcn_layers)
        ])

        # 多任务输出头
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_features)
        )

        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, node_features)
        )

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量 (batch_size, time_window, num_nodes, node_features)
        Returns:
            tuple: (预测结果, 重构结果)
        """
        batch_size = x.size(0)

        # 空间特征提取
        spatial_feats = []
        for t in range(self.time_window):
            x_t = x[:, t, :, :]
            for gcn in self.gcn_layers:
                x_t = gcn(x_t)
            spatial_feats.append(x_t)
        spatial_feats = torch.stack(spatial_feats, dim=1)

        # 时间特征提取
        temporal_feats = spatial_feats
        for tcn in self.tcn_layers:
            temporal_feats = tcn(temporal_feats)

        # 时空特征融合
        fused = torch.cat([spatial_feats, temporal_feats], dim=-1)
        fused = self.fusion(fused.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # 多任务输出
        prediction = self.pred_head(fused[:, -1, :, :])
        reconstruction = self.recon_head(fused)

        return prediction, reconstruction