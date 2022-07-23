from datasets import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import os
import pandas as pd
from ast import literal_eval
from torch.utils.data import random_split


class VehiclePricerDataset(Dataset):
    def __init__(self, data_dir: str, transforms=None):
        super(VehiclePricerDataset, self).__init__()
        if not os.path.exists(os.path.join(data_dir, 'master_data.csv')):
            pass
        self.df = pd.read_csv(os.path.join(data_dir, 'master_data.csv'))
        self.df['Filepath'] = self.df['Filepaths'].apply(lambda x: literal_eval(x))
        self.df.drop(columns=['Filepaths'], inplace=True)
        self.df = self.df.explode('Filepaths')
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row['Filepath'])
        msrp = row['MSRP']
        if self.transforms:
            image = self.transforms(image)
        return image, msrp


class VehiclePricerDataModule(LightningDataModule):
    def __init__(self, master_csv, transforms=None, train_split=0.8, val_split=0.1, batch_size=32, shuffle=True, num_workers=4, drop_last=True):
        super(VehiclePricerDataModule, self).__init__()
        self.master_csv = master_csv
        self.transforms = transforms
        self.batch_size = batch_size
        self.dataset = VehiclePricerDataset(master_csv, transforms)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            VehiclePricerDataset(master_csv, transforms), 
            [
                int(len(self.dataset) * train_split), 
                int(len(self.dataset) * val_split), 
                len(self.dataset) - int(len(self.dataset) * (train_split + val_split))
            ]
        )
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
            sampler=SubsetRandomSampler(range(len(self.train_dataset))),
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
            sampler=SubsetRandomSampler(range(len(self.val_dataset))),
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
            sampler=SubsetRandomSampler(range(len(self.test_dataset))),
        )

