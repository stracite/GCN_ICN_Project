import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros

# 空间图
class ChebGCN(ChebConv):
    def __init__(self, in_channels, out_channels, K=1, normalization='sym'):
        super().__init__(in_channels, out_channels, K, normalization)

        # 显式注册子模块（必须在父类初始化之后）
        self.res_linear = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        # 手动初始化参数（绕过父类 reset_parameters 的覆盖）
        self._init_res_linear()

    def _init_res_linear(self):
        glorot(self.res_linear.weight)
        zeros(self.res_linear.bias)
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # 父类 ChebConv 前向传播
        out = super().forward(x, edge_index, edge_weight)
        # 残差连接
        residual = self.res_linear(x)
        # 批量归一化 + ReLU
        return torch.relu(self.bn(out) + residual)


# 时间图
class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):  # 新增dilation参数
        super().__init__()
        assert in_channels == out_channels, "输入输出通道必须相同"

        # 主分支：膨胀因果卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=(dilation * (3 - 1)) // 2,  # 因果填充（左侧填充）
                    dilation=dilation
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    padding=(dilation * (5 - 1)) // 2,  # 因果填充（左侧填充）
                    dilation=dilation
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        ])

        # 残差分支
        self.res_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        # 融合层（保持不变）
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=1),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        residual = self.res_conv(x)
        scale_features = []
        for conv in self.conv_layers:
            scale_features.append(conv(x))
        fused = self.fusion(torch.cat(scale_features, dim=1))
        return F.relu(fused + residual)
