import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from graph_layer import GraphLayer

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()

# max min(0-1)
def norm(train, test):
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels, "输入输出通道必须相同"
        # 主分支：多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        ])
        # 残差分支：深度可分离卷积
        self.res_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=1),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        residual = self.res_conv(x)
        # 并行多尺度处理
        scale_features = []
        for conv in self.conv_layers:
            scale_features.append(conv(x))
        # 特征融合
        fused = self.fusion(torch.cat(scale_features, dim=1))
        return F.relu(fused + residual)


class STGCN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim, input_dim,
                 out_layer_num, topk):
        super().__init__()
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.topk = topk
        self.input_dim = input_dim
        self.embedding = nn.Embedding(node_num, dim)
        self.bn_outlayer_in = nn.BatchNorm1d(dim)
        edge_set_num = len(edge_index_sets)
        # 空间卷积层
        self.gnn_layers = nn.ModuleList([
            GraphLayer(input_dim, dim, inter_dim=dim + dim, heads=1)
            for _ in range(edge_set_num)
        ])
        # 时间卷积层
        self.tcn = nn.Sequential(
            MultiScaleTemporalBlock(dim * edge_set_num, dim * edge_set_num),
            MultiScaleTemporalBlock(dim * edge_set_num, dim * edge_set_num),
            nn.AdaptiveAvgPool1d(1)  # 保持原有池化层
        )
        # 输出层
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num)
        self.dp = nn.Dropout(0.2)
        self.cache_edge_index_sets = [None] * edge_set_num

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index=None):
        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        # 动态计算时间步长
        seq_len = all_feature // self.input_dim
        assert all_feature == self.input_dim * seq_len, \
            f"特征维度{all_feature}必须等于input_dim({self.input_dim})×时间步长({seq_len})"
        # 维度重构 [B, N, D*T] -> [B*T, N, D]
        x = x.view(batch_num, node_num, self.input_dim, seq_len)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(-1, node_num, self.input_dim)
        batch_size = x.shape[0]
        x = x.view(-1, self.input_dim)
        # 空间图卷积
        gcn_outs = []
        for i, edge_index in enumerate(self.edge_index_sets):
            # 动态拓扑生成
            all_embeddings = self.embedding(torch.arange(node_num).to(x.device))
            weights = all_embeddings.detach()
            cos_ji_mat = torch.matmul(weights, weights.T) / torch.matmul(
                weights.norm(dim=-1).unsqueeze(1),
                weights.norm(dim=-1).unsqueeze(0))
            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
            # 构建gated_edge_index
            gated_i = torch.arange(node_num, device=x.device).unsqueeze(1).repeat(1, self.topk).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_size // seq_len, node_num).to(
                x.device)
            # 图卷积处理
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index,embedding=all_embeddings.repeat(batch_size, 1))
            gcn_outs.append(gcn_out)
        # 特征聚合
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_size, node_num, -1)
        x = x.view(batch_num, seq_len, node_num, -1)
        x = x.permute(0, 2, 3, 1)
        # 时间卷积处理
        x = x.contiguous().view(batch_num * node_num, -1, seq_len)
        x = self.tcn(x)
        x = x.squeeze(-1)
        x = x.view(batch_num, node_num, -1)
        # 输出处理
        indexes = torch.arange(node_num).to(x.device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        out = self.out_layer(out)
        return out.view(-1, node_num)  # [B*N, 1]

    def __repr__(self):
        return f"STGCN(input_dim={self.input_dim}, dim={self.dim}, " + \
            f"edge_sets={len(self.edge_index_sets)}, tcn={self.use_tcn})"