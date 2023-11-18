from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import Accuracy


class SNModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,  # 模型
        loss: torch.nn.Module,  # 损失函数
        optimizer: torch.optim.Optimizer,  # 优化器
        scheduler: torch.optim.lr_scheduler,  # 学习率调度器
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss"])

        self.net = net  # 网络模型
        self.criterion = loss  # 损失函数

        # 用于记录点云补全的距离损失
        self.train_cd = MeanMetric()
        self.val_cd = MeanMetric()
        self.test_cd = MeanMetric()

        self.val_cd_best = MinMetric()

    def forward(self, partial_pcd: torch.Tensor) -> torch.Tensor:
        # 前向传播，根据部分点云生成补全的点云
        return self.net(partial_pcd)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        partial_pcd, complete_pcd = batch  # 部分点云和完整点云
        predicted_pcd = self.forward(partial_pcd)  # 预测的补全点云
        loss = self.criterion(predicted_pcd, complete_pcd)  # 计算损失
        self.train_cd(loss)  # 更新损失
        self.log("train/cd", self.train_cd, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        partial_pcd, complete_pcd = batch
        predicted_pcd = self.forward(partial_pcd)
        loss = self.criterion(predicted_pcd, complete_pcd)
        self.val_cd(loss)
        self.log("val/cd", self.val_cd, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        partial_pcd, complete_pcd = batch
        predicted_pcd = self.forward(partial_pcd)
        loss = self.criterion(predicted_pcd, complete_pcd)
        self.test_cd(loss)
        self.log("test/cd", self.test_cd, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/cd",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

# 使用该模块
# if __name__ == '__main__':
#     model = SNModule(net, loss, optimizer, scheduler)
