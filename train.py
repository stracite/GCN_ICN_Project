import torch

from evaluate import loss_func
from util.env import get_device



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

    return train_loss_list

