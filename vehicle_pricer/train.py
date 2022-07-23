from pytorch_lightning import LightningModule, Trainer, seed_everything
from model import VehiclePricingModel
from datamodule import VehiclePricingDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Normalize


class VehiclePricingModule(LightningModule):
    def __init__(self, data_dir="data", num_workers=0):
        super(VehiclePricingModule, self).__init__()
        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.num_workers = num_workers
        self.model = VehiclePricingModel()
        self.data_module = VehiclePricingDataModule(data_dir, transforms, num_workers=self.num_workers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(batch)
        exit()
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def on_epoch_start(self):
        seed_everything(self.trainer.seed)

    def on_epoch_end(self):
        self.model.eval()
        self.model.cpu()
        self.model.save_weights('model_weights.pt')
        self.model.to(self.model.device)



if __name__ == "__main__":
    trainer = Trainer(gpus=1, max_epochs=10, logger=TensorBoardLogger('logs', name='vehicle_pricer'))
    model = VehiclePricingModule()
    trainer.fit(model)

