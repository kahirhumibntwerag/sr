import numpy as np
import torch.nn.functional as F
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
import os
import s3fs
import zarr
from typing import Union
import dask.array as da

class Dataset(Dataset):
    def __init__(self, dask_data, downsample_factor=1/4, transform=None):
        """
        Args:
            dask_data (numpy.array): dask array of shape (num_images, height, width).
            downsample_factor (float): Factor by which to downsample images.
            transform (callable, optional): Transformation function to apply to images.
        """
        self.dask_data = dask_data
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        return len(self.dask_data)

    def __getitem__(self, idx):        
        
        hr = self.dask_data[idx].compute()
        hight, width = hr.shape

        
        if self.transform:
            hr = self.transform(hr).float().view(-1, 1, 512, 512)
            lr = F.interpolate(hr, size=(int(hight*self.downsample_factor), int(width*self.downsample_factor)), mode='bilinear', align_corners=False)
            
        return lr.squeeze(0), hr.squeeze(0)





class DataModule(L.LightningDataModule):
    def __init__(self, wavelength='171A', batch_size=4, transform=None, random_seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.random_seed = random_seed
        self.wavelength = wavelength

    def prepare_data(self):
        # No need for downloading or tokenizing
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load the full dataset as a Dask array
            data = load_single_aws_zarr(
                path_to_zarr=(
            "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2.zarr/2015"
            ),
                wavelength='171A'
            )

            data = data[:50]
            # Create the Dataset
            dataset = Dataset(data, transform=self.transform)

            # Define split sizes
            train_size = int(0.8 * 50)
            val_size = int(0.1 * 50)
            test_size = 50 - train_size - val_size

            # Split the dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.random_seed)
            )

        elif stage == 'test':
            # If test dataset is already created in 'fit', no need to do anything
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size
        )





AWS_ZARR_ROOT = (
    "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2.zarr/"
)


def s3_connection(path_to_zarr: os.path) -> s3fs.S3Map:
    """
    Instantiate connection to aws for a given path `path_to_zarr`
    """
    return s3fs.S3Map(
        root=path_to_zarr,
        s3=s3fs.S3FileSystem(anon=True),
        # anonymous access requires no credentials
        check=False,
    )


def load_single_aws_zarr(
    path_to_zarr: os.path,
    cache_max_single_size: int = None,
    wavelength='171A',
) -> Union[zarr.Array, zarr.Group]:
    """
    load zarr from s3 using LRU cache
    """
    root = zarr.open(
            zarr.LRUStoreCache(
                store=s3_connection(path_to_zarr),
                max_size=cache_max_single_size,
            ),
            mode="r",
         )
    data = root[wavelength]
    data = da.from_array(data)

    return data
