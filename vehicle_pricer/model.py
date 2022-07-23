from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Dropout, BatchNorm2d, Flatten, Tanh, Dropout, GRU
from pytorch_lightning import LightningModule
import torch


class VehiclePricingModel(LightningModule):
    def __init__(self, img_shape=(64, 64)):
        super(VehiclePricingModel, self).__init__()
        self.img_shape = img_shape
        self.dropout = 0.2
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.dropout),
            Flatten(),
            Linear(256 * (self.img_shape[0] // 16) * (self.img_shape[1] // 16), 512),
            ReLU(),
            Dropout(self.dropout),
            Linear(512, 256),
            ReLU(),
            Dropout(self.dropout),
            Linear(256, 1),
            Tanh()
        )
    
    def forward(self, x):
        return self.model(x)



