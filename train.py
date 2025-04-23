# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class STGCNDispatcher:
    """模型训练调度器
    Args:
        model (nn.Module): 待训练模型
        device (torch.device): 训练设备
        loss_weights (tuple): 损失权重 (pred, recon) (default: (0.7, 0.3))
        lr (float): 学习率 (default: 1e-3)
        grad_clip (float): 梯度裁剪阈值 (default: 5.0)
    """

    def __init__(self, model, device, loss_weights=(0.7, 0.3),
                 lr=1e-3, grad_clip=5.0):
        self.model = model.to(device)
        self.device = device
        self.weights = loss_weights
        self.grad_clip = grad_clip

        # 损失函数和优化器
        self.criterion = {
            'pred': torch.nn.MSELoss(),
            'recon': torch.nn.L1Loss()
        }
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, verbose=True)

    def train(self, train_data, val_data, epochs=100, batch_size=32):
        """完整训练流程
        Args:
            train_data (tuple): 训练数据 (inputs, targets)
            val_data (tuple): 验证数据
            epochs (int): 训练轮次
            batch_size (int): 批大小
        """
        train_loader = self._create_loader(train_data, batch_size, shuffle=True)
        val_loader = self._create_loader(val_data, batch_size * 2)

        best_loss = float('inf')
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                preds, recons = self.model(inputs)
                loss = self._compute_loss(preds, recons, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            val_loss = self._validate(val_loader)
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_stgcn.pth')

            print(f"Train Loss: {train_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")

    def _compute_loss(self, preds, recons, targets):
        """计算组合损失"""
        pred_loss = self.criterion['pred'](preds, targets)
        recon_loss = self.criterion['recon'](recons, targets.unsqueeze(1))
        return self.weights[0] * pred_loss + self.weights[1] * recon_loss

    def _validate(self, loader):
        """验证阶段"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                preds, recons = self.model(inputs)
                loss = self._compute_loss(preds, recons, targets)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _create_loader(self, data, batch_size, shuffle=False):
        """创建数据加载器"""
        dataset = TensorDataset(torch.FloatTensor(data[0]),
                                torch.FloatTensor(data[1]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)