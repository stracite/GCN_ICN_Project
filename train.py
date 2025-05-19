import torch.nn.functional as F

from test import *


# train.py
def loss_func(y_pred, y_true, x_recon, x_orig, mu, logvar):
    # 预测任务RMSE
    rmse_loss = torch.sqrt(F.mse_loss(y_pred, y_true))

    # 重构任务损失（展平为 [B*N, slide_win]）
    x_recon_flat = x_recon.view(-1, x_recon.size(-1))  # [batch*node_num, input_dim]
    x_orig_flat = x_orig.view(-1, x_orig.size(-1))     # [batch*node_num, input_dim]
    recon_loss = F.mse_loss(x_recon_flat, x_orig_flat)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x_orig.size(0) * x_orig.size(1))   # 归一化KL散度

    # 总损失（权重可调整）
    total_loss = 1.6 * rmse_loss + 0.6 * recon_loss + 0.2 * kl_loss
    return total_loss, {
        'total': total_loss.item(),
        'rmse': rmse_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item()
    }

def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, progress_callback=None, should_continue=None):
    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=config['decay'])
    train_loss_list = []
    device = get_device()
    epoch = config['epoch']

    for i_epoch in range(epoch):
        if should_continue and not should_continue():
            print("Training stopped by user")
            break

        model.train()

        for x, labels, attack_labels, edge_index in train_dataloader:
            x, edge_index = [item.float().to(device) for item in [x, edge_index]]
            y_true = labels.float().to(device)

            optimizer.zero_grad()

            # 单次前向传播获取多任务输出
            anomaly_out, pred_out, recon_out, mu, logvar = model(x, edge_index)

            # 计算多任务损失
            loss, loss_dict = loss_func(
                y_pred=pred_out,
                y_true=y_true,  # 使用传感器数据作为预测目标
                x_recon=recon_out,
                x_orig=x.view(-1, x.size(-1)),  # 输入数据展平
                mu=mu,
                logvar=logvar
            )

            loss.backward()
            optimizer.step()

            # 回调传递多任务损失
            # 记录损失到列表
            train_loss_list.append(loss_dict['total'])
            if progress_callback:
                metrics = {
                    'epoch': i_epoch + 1,
                    'loss': loss_dict['total'],
                    **loss_dict,
                    'loss_history': train_loss_list.copy()
                }
                progress_callback(metrics)


        torch.save(model.state_dict(), save_path)


        # if progress_callback:  # 回调触发
        #     metrics = {
        #         'epoch': i_epoch + 1,
        #         'loss': loss_dict['total'],
        #         **loss_dict,
        #         'loss_history': train_loss_list
        #     }
        #     progress_callback(metrics)

    return train_loss_list

