import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from .graph_layer import ChebGCN, MultiScaleTemporalBlock


class ST_VAE(nn.Module):
    def __init__(self, fused_dim, hidden_dim, output_dim):
        super().__init__()
        # Encoder: 融合特征 -> 潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 * 2)  # 输出mu和logvar
        )

        # Decoder: 潜在空间 -> 目标输出维度
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出维度对齐input_dim=16
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

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


class STGCN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim, input_dim, out_layer_num, topk):
        super().__init__()
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.topk = topk
        self.input_dim = input_dim
        self.embedding = nn.Embedding(node_num, dim)
        self.bn_outlayer_in = nn.BatchNorm1d(dim)
        edge_set_num = len(edge_index_sets)

        self.attention = nn.Sequential(
            nn.Linear(dim * len(edge_index_sets), dim),
            nn.Sigmoid()
        )
        # 空间卷积层
        self.gnn_layers = nn.ModuleList([
            ChebGCN(input_dim, dim, K=1)  # K=1 表示一阶切比雪夫近似
            for _ in range(edge_set_num)
        ])
        # 时间卷积层
        self.tcn = nn.Sequential(
            MultiScaleTemporalBlock(in_channels=dim * edge_set_num, out_channels=dim * edge_set_num, dilation=1),
            MultiScaleTemporalBlock(in_channels=dim * edge_set_num, out_channels=dim * edge_set_num, dilation=2),
            MultiScaleTemporalBlock(in_channels=dim * edge_set_num, out_channels=dim * edge_set_num, dilation=4),
            nn.AdaptiveAvgPool1d(1)
        )
        # 时空特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear((dim * edge_set_num) * 2, dim * edge_set_num),  # 融合GCN和TCN特征
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 输出层
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num)
        self.dp = nn.Dropout(0.2)
        self.cache_edge_index_sets = [None] * edge_set_num

        # ============ 新增多任务输出层 ============
        # 预测任务MLP
        self.predictor = nn.Sequential(
            nn.Linear(dim * len(edge_index_sets), dim),  # ✅ 明确输入输出维度
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # 重构任务VAE
        self.vae = ST_VAE(
            fused_dim=dim * len(edge_index_sets),  # 输入维度=融合特征维度
            hidden_dim=dim,
            output_dim=input_dim  # 输出维度=原始输入特征维度
        )



    def forward(self, data, org_edge_index=None):
        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        seq_len = all_feature // self.input_dim
        assert all_feature == self.input_dim * seq_len

        # 维度重构 [B, N, D*T] -> [B*T, N, D]
        x = x.view(batch_num, node_num, self.input_dim, seq_len)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(-1, node_num, self.input_dim)
        batch_size = x.shape[0]
        x = x.view(-1, self.input_dim)

        # 空间图卷积
        gcn_outs = []
        for i, edge_index in enumerate(self.edge_index_sets):
            batch_edge_index = get_batch_edge_index(
                edge_index,
                batch_size // seq_len,
                node_num
            ).to(x.device)
            gcn_out = self.gnn_layers[i](x, batch_edge_index)
            gcn_outs.append(gcn_out)

        # 空间特征聚合
        x_spatial = torch.cat(gcn_outs, dim=1)  # [batch_size * seq_len, node_num, dim * edge_set_num]
        x_spatial = x_spatial.view(batch_num, seq_len, node_num, -1)  # [B, T, N, dim*edge_set_num]
        x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()  # [B, N, T, dim*edge_set_num]

        # 时间卷积处理（使用空间卷积后的特征）
        x_temporal = x_spatial.permute(0, 3, 1, 2)  # [B, dim*edge_set_num, N, T]
        x_temporal = x_temporal.contiguous().view(batch_num * node_num, -1, seq_len)  # [B*N, dim*edge_set_num, T]
        x_temporal = self.tcn(x_temporal)  # [B*N, dim*edge_set_num, 1]
        x_temporal = x_temporal.squeeze(-1)  # [B*N, dim*edge_set_num]
        x_temporal = x_temporal.view(batch_num, node_num, -1)  # [B, N, dim*edge_set_num]

        # 空间特征聚合
        x_spatial = x_spatial.mean(dim=2)  # 沿时间维度取平均 [B, N, dim*edge_set_num]

        # 时空特征融合
        fused_features = torch.cat([x_spatial, x_temporal], dim=-1)  # [B, N, (dim*edge_set_num + temporal_features)]
        fused_features = self.fusion_layer(fused_features)  # [B, N, fused_dim]
        # 新增注意力权重
        attn_weights = self.attention(fused_features)  # [B, N, dim]
        fused_features = fused_features * attn_weights  # 特征加权

        # 预测任务输出 [B, N, 1]
        pred_output = self.predictor(fused_features)
        # 调整维度为 [B, N]（与真实标签对齐）
        pred_output = pred_output.squeeze(-1)

        # 重构任务处理
        vae_input = fused_features.view(-1, fused_features.size(-1))  # [B*N, fused_dim=64]
        recon_output, mu, logvar = self.vae(vae_input)  # 输出 [B*N, output_dim=16]

        # 调整重构输出维度 [B, N, input_dim=16]
        recon_output = recon_output.view(batch_num, self.node_num, self.input_dim)


        # 输出处理
        indexes = torch.arange(node_num).to(x.device)
        embedding = self.embedding(indexes)  # [N, dim]
        # 调整 embedding 维度以匹配 fused_features
        embedding = embedding.unsqueeze(0).expand(batch_num, -1, -1)  # [B, N, dim]
        # 如果 fused_dim != dim，需添加投影层
        if fused_features.size(-1) != self.dim:
            self.projection = nn.Linear(fused_features.size(-1), self.dim).to(x.device)
            fused_features = self.projection(fused_features)
        out = torch.mul(fused_features, embedding)  # [B, N, dim]
        out = out.permute(0, 2, 1)  # [B, dim, N]
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)  # [B, N, dim]
        out = self.dp(out)
        out = self.out_layer(out).view(-1, node_num)  # 输入维度需匹配 [B*N, 1]


        return out, pred_output, recon_output, mu, logvar

    # def __repr__(self):
    #     return f"STGCN(input_dim={self.input_dim}, dim={self.dim}, " + \
    #         f"edge_sets={len(self.edge_index_sets)}, tcn={self.use_tcn})"


