import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_layer import ChebGCN, MultiScaleTemporalBlock

class STGCN(nn.Module):
    """时空图卷积网络，集成多任务输出

    Args:
        edge_index_sets (list): 边索引集合列表
        node_num (int): 节点总数
        dim (int): 特征维度
        slide_win (int): 输入特征维度
        mlp_layer_num (int): 输出层层数
        topk (int): 拓扑k值
    """
    def __init__(self, edge_index_sets, node_num, dim, slide_win, mlp_layer_num, topk):
        super().__init__()
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.topk = topk
        self.input_dim = slide_win
        self.embedding = nn.Embedding(node_num, dim)
        self.bn = nn.BatchNorm1d(dim)
        edge_set_num = len(edge_index_sets)

        self.attention = nn.Sequential(
            nn.Linear(dim * len(edge_index_sets), dim),
            nn.Sigmoid()
        )
        # 空间卷积层
        self.gcn_layers = nn.ModuleList([
            ChebGCN(slide_win, dim, K=3)  # K=3 表示三阶切比雪夫近似
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

        # ============ 新增多任务输出层 ============
        # 预测任务MLP
        self.predictor = MLP(
            in_num=dim * len(edge_index_sets),  # 必须等于融合特征维度
            layer_num=max(mlp_layer_num, 2),  # 强制层数≥2
            inter_num=dim * 4  # 推荐设置为dim的倍数
        )
        # 重构任务VAE
        self.vae = ST_VAE(
            fused_dim=dim * len(edge_index_sets),  # 输入维度=融合特征维度
            hidden_dim=dim,
            output_dim=slide_win  # 输出维度=原始输入特征维度
        )
        # 输出层
        self.dp = nn.Dropout(0.2)
        self.cache_edge_index_sets = [None] * edge_set_num



    def forward(self, data):
        """前向传播过程，包含时空特征提取和多任务输出

        Args:
            data (Tensor): 输入数据张量，形状为[B, N, D*T]

        Returns:
            tuple: 包含主输出、预测输出、重构输出、均值和对数方差的五元组
        """
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
            gcn_out = self.gcn_layers[i](x, batch_edge_index)
            gcn_outs.append(gcn_out)

        # 空间特征聚合
        x_spatial = torch.cat(gcn_outs, dim=1)  # [B*T, N, dim * edge_set_num]
        x_spatial = x_spatial.view(batch_num, seq_len, node_num, -1)  # [B, T, N, dim*edge_set_num]
        x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()  # [B, N, T, dim*edge_set_num]


        x_temporal = x_spatial.permute(0, 3, 1, 2)  # [B, dim*edge_set_num, N, T]
        x_temporal = x_temporal.contiguous().view(batch_num * node_num, -1, seq_len)  # [B*N, dim*edge_set_num, T]
        # 时间卷积处理
        x_temporal = self.tcn(x_temporal).squeeze(-1)  # [B*N, dim*edge_set_num, 1]
        #x_temporal = x_temporal.squeeze(-1)  # [B*N, dim*edge_set_num]


        # 空间时间特征聚合
        x_spatial = x_spatial.mean(dim=2)  # 沿时间维度取平均 [B, N, dim*edge_set_num]
        x_temporal = x_temporal.view(batch_num, node_num, -1)  # [B, N, dim*edge_set_num]
        # print("空间图数据形状：",x_spatial.shape)
        # print("时间图数据形状：",x_temporal.shape)
        # 时空特征融合
        fused_features = torch.cat([x_spatial, x_temporal], dim=-1)  # [B, N, (dim*edge_set_num + temporal_features)]
        fused_features = self.fusion_layer(fused_features)  # [B, N, fused_dim]
        # 注意力权重
        attn_weights = self.attention(fused_features)  # [B, N, dim]
        fused_features = fused_features * attn_weights  # 特征加权


        # 预测任务输出 [B, N]
        # 1. 调整输入维度 [B, N, fused_dim] -> [B*N, fused_dim]
        pred_input = fused_features.view(-1, fused_features.size(-1))  # [B*N, fused_dim]
        pred_output = self.predictor(pred_input)  # [B*N, 1]
        pred_output = pred_output.view(batch_num, node_num)  # [B, N]
        # 重构任务处理
        vae_input = fused_features.view(-1, fused_features.size(-1))  # [B*N, fused_dim=64]
        recon_output, mu, logvar = self.vae(vae_input)  # 输出 [B*N, output_dim=16]
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
        out = F.relu(self.bn(out))
        out = out.permute(0, 2, 1)  # [B, N, dim]
        out = self.dp(out).reshape(-1, node_num) # 输入维度需匹配 [B*N, 1]

        # print("out:",out[0])
        # print("pred_output:",pred_output[0])
        # print("recon_output:",recon_output[0])
        # print("mu:",mu[0])
        # print("logvar:",logvar[0])

        return out, pred_output, recon_output, mu, logvar

class MLP(nn.Module):
    """实现多层感知机（Multi-Layer Perceptron）模型

    参数：
        in_num (int): 输入特征维度
        layer_num (int): 网络总层数（包含输入层和输出层）
        inter_num (int, optional): 中间层特征维度，默认512

    属性：
        mlp (nn.Sequential): 按配置参数构建的序列网络模型
    """
    def __init__(self, in_num, layer_num, inter_num):
        super(MLP, self).__init__()
        modules = []
        # 构建输入层：线性变换+归一化+激活+随机失活
        # 将输入维度调整到中间层统一维度，完成特征预处理
        modules.append(nn.Linear(in_num, inter_num))
        modules.append(nn.LayerNorm(inter_num))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.1))

        # 构建隐藏层：重复堆叠相同结构的中间层
        # 总层数-2表示扣除首层和末层后的中间层数量
        for _ in range(layer_num - 2):  # 总层数-2（首层+末层）
            modules.append(nn.Linear(inter_num, inter_num))
            modules.append(nn.LayerNorm(inter_num))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.1))

        # 构建输出层：将特征维度压缩到1实现最终预测
        modules.append(nn.Linear(inter_num, 1))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        """前向传播函数

        参数：
            x (torch.Tensor): 输入张量，形状为(batch_size, in_num)

        返回：
            torch.Tensor: 输出张量，形状为(batch_size, 1)
        """
        return self.mlp(x)


class ST_VAE(nn.Module):
    """时空变分自编码器，用于特征重构

    Args:
        fused_dim (int): 融合特征维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出特征维度（需与原始输入维度对齐）
    """
    def __init__(self, fused_dim, hidden_dim, output_dim):
        super().__init__()
        # Encoder: 融合特征 -> 潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 * 2)
        )

        # Decoder: 潜在空间 -> 目标输出维度
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧，用于生成潜在变量"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """前向传播过程

        Returns:
            tuple: (重构输出, 均值向量, 对数方差向量)
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """生成批量处理的边索引

    Args:
        org_edge_index (Tensor): 原始边索引矩阵，形状为[2, edge_num]
        batch_num (int): 批量大小
        node_num (int): 单个样本的节点数量

    Returns:
        Tensor: 扩展后的批量边索引矩阵，形状为[2, batch_num*edge_num]
    """
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()