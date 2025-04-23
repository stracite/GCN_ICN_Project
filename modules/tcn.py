# modules/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedTCN(nn.Module):
    """膨胀时间卷积网络层
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小 (default: 3)
        dilation (int): 膨胀系数 (default: 1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量 (batch, time_steps, num_nodes, features)
        Returns:
            torch.Tensor: 输出张量
        """
        batch, T, N, C = x.shape

        # 维度转换 [batch, nodes, features, time]
        x = x.permute(0, 2, 3, 1).reshape(-1, C, T)

        # 因果卷积
        out = self.conv(x)
        out = out[..., :-self.padding] if self.padding != 0 else out
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # 恢复原始维度
        out = out.reshape(batch, N, C, -1).permute(0, 3, 1, 2)
        return out