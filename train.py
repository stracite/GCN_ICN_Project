import torch
from test import *
import torch.nn.functional as F


# train.py

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, progress_callback=None, should_continue=None):

    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    time.time()
    train_loss_list = []
    device = get_device()
    min_loss = 1e+8
    i = 0
    epoch = config['epoch']
    early_stop_win = 15
    model.train()
    stop_improve_count = 0
    dataloader = train_dataloader

    for i_epoch in range(epoch):

        if should_continue and not should_continue():  # 中断检查
            print("Training stopped by user")
            break

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch+1, epoch,
                        acu_loss/len(dataloader), acu_loss), flush=True
            )



        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)


            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

        if progress_callback:  # 回调触发
            metrics = {
                'epoch': i_epoch + 1,
                'loss': acu_loss / len(dataloader),
                'loss_history': train_loss_list
            }
            progress_callback(metrics)

    return train_loss_list

