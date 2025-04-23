# example_usage.py
import numpy as np
import torch
from utils import TemporalDataProcessor, IndustrialGraphBuilder
from models import STGCN
from train import STGCNDispatcher
from detect import IndustrialAnomalyDetector


def main():
    # 配置参数
    num_nodes = 10  # 设备数量
    time_steps = 1000  # 总时间步
    features = 3  # 每个设备的特征数（温度、压力、流量）
    window_size = 24  # 时间窗口大小

    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA GPU支持，当前环境无法找到可用GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 生成模拟数据
    print("生成模拟数据...")
    np.random.seed(42)
    raw_data = np.random.randn(time_steps, num_nodes, features)

    # 构建物理连接图（随机生成示例）
    physical_adj = np.random.randint(0, 2, (num_nodes, num_nodes))
    np.fill_diagonal(physical_adj, 0)  # 移除自连接

    # ========== 数据预处理 ==========
    print("\n数据预处理...")
    processor = TemporalDataProcessor(
        window_size=window_size,
        smooth_window=5,
        poly_order=3
    )
    processed_data = processor.process(raw_data)  # (samples, nodes, time_steps, features)

    # ========== 图结构构建 ==========
    print("\n构建图结构...")
    graph_builder = IndustrialGraphBuilder(physical_adj)
    static_graph = graph_builder.build_static_graph(weighted=True)

    # ========== 准备训练数据 ==========
    print("\n准备训练数据...")
    # 划分输入和标签（预测最后一个时间步）
    inputs = processed_data[:, :, :-1, :]  # (samples, nodes, window-1, features)
    targets = processed_data[:, :, -1, :]  # (samples, nodes, features)

    # 转换为PyTorch张量并调整维度顺序
    inputs = torch.FloatTensor(inputs).permute(0, 2, 1, 3)  # (batch, time, nodes, features)
    targets = torch.FloatTensor(targets)

    # 划分训练验证集
    split = int(0.8 * len(inputs))
    train_data = (inputs[:split], targets[:split])
    val_data = (inputs[split:], targets[split:])

    # ========== 模型初始化 ==========
    print("\n初始化模型...")
    model = STGCN(
        node_features=features,
        hidden_dim=64,
        adj_matrix=torch.FloatTensor(static_graph),
        time_window=window_size - 1,  # 输入窗口比总窗口小1
        num_nodes=num_nodes,
        num_gcn_layers=2,
        num_tcn_layers=2
    ).to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")

    # ========== 模型训练 ==========
    print("\n开始训练...")
    dispatcher = STGCNDispatcher(
        model=model,
        device=device,
        loss_weights=(0.7, 0.3),
        lr=1e-3
    )
    dispatcher.train(
        train_data=train_data,
        val_data=val_data,
        epochs=50,
        batch_size=32
    )

    # ========== 异常检测 ==========
    print("\n进行异常检测...")
    # 加载最佳模型
    model.load_state_dict(torch.load('best_stgcn.pth'))
    model.eval()

    # 生成测试数据
    test_inputs = val_data[0].to(device)
    with torch.no_grad():
        preds, recons = model(test_inputs)

    # 计算误差
    pred_errors = torch.abs(preds - val_data[1].to(device)).cpu().numpy()
    recon_errors = torch.abs(recons - test_inputs.permute(0, 2, 1, 3)).cpu().numpy()

    # 初始化检测器
    detector = IndustrialAnomalyDetector(
        window_size=60,
        alpha=0.99,
        error_weights=(0.6, 0.4)
    )

    # 在线检测模拟
    batch_size = 10
    for i in range(0, len(pred_errors), batch_size):
        batch_pred = pred_errors[i:i + batch_size]
        batch_recon = recon_errors[i:i + batch_size]

        # 更新检测器
        detector.update(batch_pred, batch_recon)

        # 检测异常
        anomalies, scores = detector.detect(batch_pred, batch_recon)
        print(f"Batch {i // batch_size}: 异常设备数 - {np.sum(anomalies)}")

    print("\n检测完成！")


if __name__ == "__main__":
    main()