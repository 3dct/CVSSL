# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Torch Dataset wrapper for the MNIST dataset with normalization."""

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import torch 
import gc
from .hdf5_loader import load_hdf5, extract_random_block_3d

class VolumeDataset(Dataset):
    """Custom Dataset for loading MNIST with preprocessing."""

    def __init__(self, image_files, transforms, randCropTransform, useCache = False):
        self.image_files = image_files
        self.transforms = transforms
        #self.randCropTransform = randCropTransform

        #self.dataCache = defaultdict(dict)
        self.useCache = useCache
        torch.multiprocessing.set_sharing_strategy('file_system')
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        self.info =  torch.utils.data.get_worker_info()

        if self.useCache:
            if index in self.dataCache:
                None
                #image = self.dataCache[index] #.to(f"cuda:{self.deviceIndex}")
            else:
                #image = self.transforms(self.image_files[index]).to("cpu")
                image = load_hdf5(self.image_files[index])
                image = self.transforms(image).to("cpu")
                #self.dataCache[index] = image
                self.deviceIndex = image.device.index
        else:
            #image = self.transforms(self.image_files[index]).to("cpu")
            image = load_hdf5(self.image_files[index])

            image = self.transforms(image).to("cpu")
        if image.shape[0] == 2:
            image = image[0].unsqueeze( 0)  #.reshape((1,256,256,256))
        #return self.randCropTransform(image), self.randCropTransform(image)
        gc.collect()
        return image, image