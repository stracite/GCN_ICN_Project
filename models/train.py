import torch
import torch.nn.functional as F

from util.env import get_device


def train(model=None, save_path='', config={}, train_dataloader=None, progress_callback=None, should_continue=None):
    """
    训练模型的主函数，支持多任务训练和用户中断

    Args:
        model (nn.Module): 待训练的模型
        save_path (str): 模型参数保存路径
        config (dict): 配置字典
        train_dataloader (DataLoader): 训练数据加载器
        progress_callback (function): 训练进度回调函数，接收指标字典
        should_continue (function): 中断检查函数，返回False时停止训练

    Returns:
        list: 训练过程中总损失的历史记录
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=config['decay'])
    train_loss_list = []
    device = get_device()
    epoch = config['epoch']

    # 主训练循环
    for i_epoch in range(epoch):
        # 检查用户中断请求
        if should_continue and not should_continue():
            print("Training stopped by user")
            break

        model.train()

        # 批量训练过程
        for x, labels, attack_labels in train_dataloader:
            x = x.float().to(device)
            y_true = labels.float().to(device)

            optimizer.zero_grad()

            # 前向传播获取多任务输出（异常检测/预测/重构）
            anomaly_out, pred_out, recon_out, mu, logvar = model(x)

            # 多任务损失计算（预测/重构/变分）
            loss, loss_dict = loss_func(
                y_pred=pred_out,
                y_true=y_true,
                x_recon=recon_out,
                x_orig=x.view(-1, x.size(-1)),
                mu=mu,
                logvar=logvar
            )

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 记录损失并回调
            train_loss_list.append(loss_dict['total'])
            if progress_callback:
                metrics = {
                    'epoch': i_epoch + 1,
                    'loss': loss_dict['total'],
                    **loss_dict,
                    'loss_history': train_loss_list.copy()
                }
                progress_callback(metrics)

        # 每epoch保存模型参数
        torch.save(model.state_dict(), save_path)

    return train_loss_list


def loss_func(y_pred, y_true, x_recon, x_orig, mu, logvar):
    """
    计算多任务组合损失函数

    Args:
        y_pred (Tensor): 预测任务输出
        y_true (Tensor): 预测任务真实值
        x_recon (Tensor): 重构任务输出
        x_orig (Tensor): 原始输入数据
        mu (Tensor): 变分编码均值
        logvar (Tensor): 变分编码对数方差

    Returns:
        tuple: (总损失Tensor, 各损失分量字典)
    """
    # 预测任务均方根误差
    mlp_loss = torch.sqrt(F.mse_loss(y_pred, y_true))

    # 重构任务均方误差（展平后计算）
    x_recon_flat = x_recon.view(-1, x_recon.size(-1))
    x_orig_flat = x_orig.view(-1, x_orig.size(-1))
    recon_loss = F.mse_loss(x_recon_flat, x_orig_flat)

    # 变分自编码器KL散度（归一化处理）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x_orig.size(0) * x_orig.size(1))

    # 加权组合总损失（权重参数需实验调整）
    total_loss = 0.5 * mlp_loss + 0.4 * recon_loss + 0.1 * kl_loss
    return total_loss, {
        'total': total_loss.item(),
        'mlp': mlp_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item()
    }
