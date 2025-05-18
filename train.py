import torch
from test import *
import torch.nn.functional as F


# train.py

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, progress_callback=None,
          should_continue=None):
    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=config['decay'])
    train_loss_list = []
    device = get_device()
    epoch = config['epoch']

    for i_epoch in range(epoch):
        if should_continue and not should_continue():
            print("Training stopped by user")
            break

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in train_dataloader:
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()

            if progress_callback:
                metrics = {
                    'epoch': i_epoch + 1,
                    'loss': loss.item(),
                    'loss_history': train_loss_list.copy()
                }
                progress_callback(metrics)


        torch.save(model.state_dict(), save_path)


        if progress_callback:  # 回调触发
            metrics = {
                'epoch': i_epoch + 1,
                'loss': acu_loss,
                'loss_history': train_loss_list
            }
            progress_callback(metrics)

    return train_loss_list

