# modules/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """图卷积网络层
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        adj_matrix (torch.Tensor): 邻接矩阵
        residual (bool): 是否使用残差连接 (default: True)
    """

    def __init__(self, in_features, out_features, adj_matrix, residual=True):
        super().__init__()
        self.register_buffer('adj', adj_matrix)  # 固定邻接矩阵
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.residual = residual

        if residual:
            self.res_connect = nn.Linear(in_features, out_features) \
                if in_features != out_features else nn.Identity()

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量 (batch_size, num_nodes, in_features)
        Returns:
            torch.Tensor: 输出张量
        """
        # 图卷积操作
        support = torch.matmul(self.adj, x)
        out = self.linear(support)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)

        # 残差连接
        if self.residual:
            res = self.res_connect(x)
            out = out + res

        return F.relu(out)
