# utils/graph_builder.py
import numpy as np
from scipy.sparse.csgraph import shortest_path


class IndustrialGraphBuilder:
    """工业设备图结构构建器
    Args:
        physical_adj (np.ndarray): 物理连接邻接矩阵
        feature_dim (int): 节点特征维度 (default: 3)
    """

    def __init__(self, physical_adj, feature_dim=3):
        self.physical_adj = physical_adj
        self.feature_dim = feature_dim

    def build_static_graph(self, weighted=True):
        """构建静态图结构
        Args:
            weighted (bool): 是否带权 (default: True)
        Returns:
            np.ndarray: 邻接矩阵
        """
        # 计算最短路径权重
        dist_matrix = shortest_path(self.physical_adj, directed=False)
        np.fill_diagonal(dist_matrix, 1.0)
        weights = 1 / (dist_matrix + 1e-6)

        if not weighted:
            weights = (weights > 0).astype(np.float32)

        return self._symmetrize(weights)

    def build_dynamic_graph(self, node_features, k=5):
        """构建动态图结构
        Args:
            node_features (np.ndarray): 节点特征 (nodes, features)
            k (int): 最近邻数量 (default: 5)
        """
        # 计算余弦相似度
        norm_features = node_features / np.linalg.norm(node_features, axis=1, keepdims=True)
        sim_matrix = np.dot(norm_features, norm_features.T)

        # 保留top-k连接
        adj = np.zeros_like(sim_matrix)
        for i in range(len(sim_matrix)):
            topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
            adj[i, topk_idx] = 1

        return self._symmetrize(adj * self.physical_adj)

    def _symmetrize(self, matrix):
        """对称化处理"""
        return np.maximum(matrix, matrix.T)