import torch
import torch.nn as nn

from util.env import *


def test(model, dataloader):
    """
    测试模型在给定数据集上的性能

    参数:
    model (nn.Module): 待测试的模型
    dataloader (DataLoader): 测试数据加载器，每次迭代返回(x, y, labels, edge_index)

    返回值:
    tuple: (平均损失, [预测结果, 重构结果, 标签])
        平均损失 (float): 所有batch的平均MSE损失
        结果列表包含:
            预测结果 (np.ndarray): 所有样本的预测值，形状为[总样本数, 特征维度]
            重构结果 (np.ndarray): 所有样本的重构值，形状为[总样本数, 特征维度]
            标签 (np.ndarray): 所有样本的标签，形状为[总样本数]
    """
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []  # 存储每个batch的损失值
    # 初始化结果收集容器
    t_test_predicted_list = []  # 收集模型预测输出
    t_test_recon_list = []  # 收集重构输出
    t_test_labels_list = []  # 收集原始标签

    model.eval()  # 设置模型为评估模式

    i = 0  # batch计数器
    acu_loss = 0  # 累积损失
    for x, y, labels in dataloader:
        # 解析输入数据维度
        batch_num = x.size(0)  # 当前batch的样本数
        node_num = x.size(1)  # 传感器节点数量
        input_dim = x.size(2)  # 输入特征维度

        # 将数据转移到计算设备
        x = x.float().to(device)  # 输入特征 [B, N, F]
        y = y.float().to(device)  # 真实值 [B, N, F]
        labels = labels.to(device)  # 异常标签 [B, N]


        with torch.no_grad():  # 禁用梯度计算
            # 前向传播获取多任务输出
            # 模型返回: (异常检测结果, 预测结果, 重构结果, mu, logvar)
            _, pred, recon, _, _ = model(x)

            # 计算预测任务损失
            loss = loss_func(pred, y)
            test_loss_list.append(loss.item())
            acu_loss += loss.item()

            # 处理并收集各任务输出结果
            # 调整预测结果维度为[B*N, F]
            pred_flat = pred.view(-1, pred.size(-1))
            t_test_predicted_list.append(pred_flat.cpu())

            # 重构结果保持原始输入维度[B, N, F]后展平
            recon_output = recon.view(batch_num, node_num, input_dim)
            recon_flat = recon_output.view(-1, input_dim)
            t_test_recon_list.append(recon_flat.cpu())

            # 标签展平为[B*N]维度
            labels_flat = labels.view(-1)
            t_test_labels_list.append(labels_flat.cpu())

        i += 1

    # 合并所有batch的结果并转换为numpy数组
    test_pred = torch.cat(t_test_predicted_list, dim=0).numpy()
    test_recon = torch.cat(t_test_recon_list, dim=0).numpy()
    test_labels = torch.cat(t_test_labels_list, dim=0).numpy()

    # 计算平均损失
    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss, [test_pred, test_recon, test_labels]
