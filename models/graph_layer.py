import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros

# 空间图
class ChebGCN(ChebConv):
    """基于Chebyshev谱图卷积的改进GCN模块，包含残差连接和批量归一化

    Args:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        K (int, optional): Chebyshev多项式阶数，默认为3
        normalization (str, optional): 邻接矩阵归一化方式，默认为'sym'
    """
    def __init__(self, in_channels, out_channels, K=3, normalization='sym'):
        super().__init__(in_channels, out_channels, K, normalization)

        # 显式注册子模块（必须在父类初始化之后）
        self.res_linear = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        # 手动初始化参数（绕过父类 reset_parameters 的覆盖）
        self.init_res_linear()

    def init_res_linear(self):
        """自定义残差连接层的参数初始化方法，使用Glorot和零初始化"""
        glorot(self.res_linear.weight)
        zeros(self.res_linear.bias)
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """前向传播过程，包含残差连接和激活函数

        Args:
            x (Tensor): 节点特征矩阵
            edge_index (Tensor): 边索引张量
            edge_weight (Tensor, optional): 边权重张量
            batch (Tensor, optional): 批处理索引

        Returns:
            Tensor: 经过图卷积和残差连接后的特征矩阵
        """
        # 父类 ChebConv 前向传播
        out = super().forward(x, edge_index, edge_weight)
        # 残差连接
        residual = self.res_linear(x)
        # 批量归一化 + ReLU
        return torch.relu(self.bn(out) + residual)


# 时间图
class MultiScaleTemporalBlock(nn.Module):
    """多尺度时序特征提取模块，包含膨胀卷积和残差连接

    Args:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        dilation (int, optional): 膨胀系数，默认为1
    """
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
                    kernel_size=7,
                    padding=(dilation * (7 - 1)) // 2,  # 因果填充（左侧填充）
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

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=1),
            nn.Dropout(0.2)
        )

    def forward(self, feature):
        """前向传播过程，包含多尺度特征融合

        Args:
            feature (Tensor): 输入时序特征张量

        Returns:
            Tensor: 融合多尺度特征后的时序特征张量
        """
        residual = self.res_conv(feature)
        scale_features = []
        for conv in self.conv_layers:
            scale_features.append(conv(feature))
        fused = self.fusion(torch.cat(scale_features, dim=1))
        return F.relu(fused + residual)
