# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Example of LightningDataModule implementation for managing data loading."""

from lightning import LightningDataModule
from monai.data import DataLoader, ThreadDataLoader

from .torch_dataset import VolumeDataset

import torch

import numpy as np

from multiprocessing import Process, Manager

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandAxisFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    CenterSpatialCrop,
    RandSpatialCrop,
    Flip,
    SpatialPad,
    RandGaussianNoise,
    GaussianSmooth,
    Resize,
    ToDevice,
    RandZoom,
    RandCoarseDropout
    
)


class ExampleDataModule(LightningDataModule):
    """DataModule for training, validation, test, and prediction using ExampleTorchDataset."""
    countInstance = 0
    def __init__(self, path:str =None, batch_size: int = 32, num_workers: int = 4) -> None:
        """Initialize the DataModule.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses used for data loading.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.device = "cuda:"
        self.path = path

  

        




    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Prepare datasets for all splits.

        Args:
            stage (str | None): One of None, 'fit', 'validate', 'test', or 'predict'.
                This argument is unused in this implementation, but you could use it to conditionally
                load specific datasets based on the stage of the model training or evaluation. For example:
                - 'fit': Load the training and validation datasets.
                - 'validate': Optionally, load validation data only.
                - 'test': Load test dataset for evaluation.
                - 'predict': Load prediction dataset for inference.
        """

        import glob
        import SimpleITK as sitk
        image_files_list2 = glob.glob(self.path,recursive=True)

        image_files_list2.extend(image_files_list2)
        image_files_list2.extend(image_files_list2)
        image_files_list2.extend(image_files_list2)

        print(len(image_files_list2))

        val_frac = 0.15
        test_frac = 0.15
        length = len(image_files_list2)
        indices = np.arange(length)
        np.random.shuffle(indices)

        test_split = int(test_frac * length)
        val_split = int(val_frac * length) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices #indices[val_split:]

        self.manager = Manager()
        self.train_x = self.manager.list([image_files_list2[i] for i in train_indices])

        self.val_x = self.manager.list([image_files_list2[i] for i in val_indices])

        self.test_x = [image_files_list2[i] for i in test_indices]
        self.train_transform_randCrop = Compose([RandAxisFlip(prob=0.5),RandRotate(range_x=1.0, range_y=1.0, range_z=1.0, prob = 1.0)])

        #self.setup_transformations()
    

    def setup_transformations(self, device_instance):
        self.train_transforms = Compose(
            [
                #LoadImage(image_only=True),
                
                EnsureChannelFirst(channel_dim='no_channel'),
                #ToDevice( device_instance),
                #Resize([256,256,256]),
                SpatialPad([384,384,384]),
                #SpatialPad([256,256,256]),
                #CenterSpatialCrop([256,256,256]),
                ScaleIntensity(),
                #RandFlip(prob=1.0),
                #RandGaussianNoise(),
                #RandCoarseDropout(1024,16)
                
            ]
        )

        return self.train_transforms

        # self.val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

        # self.y_pred_trans = Compose([Activations(softmax=True)])


    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        transform = self.setup_transformations(self.device+str(ExampleDataModule.countInstance))
        train_ds = VolumeDataset(self.train_x, transform, self.train_transform_randCrop)
        train_loader = ThreadDataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=2, persistent_workers=False, use_thread_workers=False, buffer_size=2, repeats=4, prefetch_factor=1)

        ExampleDataModule.countInstance += 1

        return train_loader


    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        transform = self.setup_transformations(self.device+str(1))
        val_ds = VolumeDataset(self.val_x, transform, self.train_transform_randCrop)
        val_loader = ThreadDataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=1,  pin_memory=False)

        return val_loader



    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        transform = self.setup_transformations(self.device+str(1))
        test_ds = VolumeDataset(self.test_x, transform, self.train_transform_randCrop)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=0)
        return test_loader

    def predict_dataloader(self) -> DataLoader:
        """Return the data loader for making predictions.

        In this simple implementation, we use the same `test_dataset` for both testing
        and prediction purposes. This can be extended in the future to use a different
        dataset or unlabeled data for inference if needed.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
