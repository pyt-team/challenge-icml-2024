import os
from torch import optim, nn, utils, Tensor
import lightning as L

class LitEMPSN(L.LightningModule):
    def __init__(self, model, mae, mad, mean, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr
        self.criterion = nn.L1Loss(reduction='sum')
        self.mae = mae
        self.mad = mad
        self.mean = mean

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)

        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("train_loss", loss)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)

        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("val_loss", loss)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def test_step(self, batch):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)
        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("test_loss", loss)
        self.log("test_mae", mae, prog_bar=True)
        return loss
