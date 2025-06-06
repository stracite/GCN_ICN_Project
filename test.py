import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *


def test(model, dataloader):
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    # 重构结果收集
    t_test_predicted_list = []
    t_test_recon_list = []
    t_test_labels_list = []

    test_len = len(dataloader)
    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:

        batch_num = x.size(0)  # 当前batch的样本数
        node_num = x.size(1)   # 传感器节点数量
        input_dim = x.size(2)

        # 数据转移到设备
        x = x.float().to(device)  # 输入数据 [batch_size, node_num, features]
        y = y.float().to(device)  # 真实值 [batch_size, node_num, features]
        labels = labels.to(device)  # 标签 [batch_size, node_num]
        edge_index = edge_index.to(device)

        with torch.no_grad():
            # 前向传播获取多任务输出
            # 模型返回：异常检测结果, 预测结果, 重构结果, mu, logvar
            _, pred, recon, _, _ = model(x, edge_index)

            # 计算预测任务的损失
            loss = loss_func(pred, y)
            test_loss_list.append(loss.item())
            acu_loss += loss.item()

            # 收集各任务结果 -------------------------------------
            # 预测结果维度调整：[batch, node, features] -> [batch*node, features]
            pred_flat = pred.view(-1, pred.size(-1))
            t_test_predicted_list.append(pred_flat.cpu())

            # 重构结果维度调整：[batch, node, features] -> [batch*node, features]
            # 改为保持原始输入维度
            recon_output = recon.view(batch_num, node_num, input_dim)  # [B, N, input_dim]
            recon_flat = recon_output.view(-1, input_dim)  # [B*N, input_dim]
            t_test_recon_list.append(recon_flat.cpu())

            # 标签维度调整：[batch, node] -> [batch*node]
            labels_flat = labels.view(-1)
            t_test_labels_list.append(labels_flat.cpu())
            # ---------------------------------------------------

        i += 1
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # 合并结果 ---------------------------------------------------
    # 预测结果：numpy数组 [total_samples, features]
    test_pred = torch.cat(t_test_predicted_list, dim=0).numpy()
    # 重构结果：numpy数组 [total_samples, features]
    test_recon = torch.cat(t_test_recon_list, dim=0).numpy()
    # 标签：numpy数组 [total_samples]
    test_labels = torch.cat(t_test_labels_list, dim=0).numpy()

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss, [test_pred, test_recon, test_labels]




