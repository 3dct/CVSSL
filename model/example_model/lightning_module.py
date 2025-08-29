# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Example of LightningModule implementation for training a classification model."""

from collections.abc import Callable

import torch
from lightning import LightningModule
from monai.networks.nets.swin_unetr import SwinTransformer

from .torch_model import ExampleTorchModel

import monai

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

class ExampleLightningModel(LightningModule):
    """LightningModule for training, validating, and testing a classification model."""

    def __init__(
        self,
        num_classes: int,
        optimizer: Callable,
        loss_fn: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            num_classes (int): Number of output classes for the classification task.
            optimizer (Callable): A partial function that returns an optimizer when passed model parameters.
            loss_fn (Callable): Loss function used for training, validation, and testing.
            scheduler (Callable, optional): A scheduler function to adjust the learning rate.

        """
        super().__init__()
        self.model = model = SwinTransformer(
            spatial_dims=3,
            in_chans=1,
            #img_size=( 256, 256, 256),
            window_size =(7,7,7),
            patch_size=( 16, 16, 16),
            embed_dim=24,
            #mlp_dim=2048,
            depths = (2, 2, 2, 2 , 2 ),
            num_heads = (3, 6, 12, 24, 48),
            use_v2=True
        )
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam


        if loss_fn != None:
            self.loss_fn = loss_fn()
        else:
            self.loss_fn = monai.losses.ContrastiveLoss(temperature=0.2, batch_size=-1)

        self.scheduler = scheduler

        self.setupTransform()

    def setupTransform(self):
        self.is_transform_cuda = False
        self.train_transform_randCrop = Compose([ToDevice( self.device), RandAxisFlip(prob=0.5),RandRotate(range_x=1.0, range_y=1.0, range_z=1.0, prob = 1.0)]) #,RandCoarseDropout(1024,16), RandZoom(prob=0.8,min_zoom=0.5,max_zoom=1.25),

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """

        #move transform to gpu
        if(self.device.type == 'cuda' and not  self.is_transform_cuda ):
            self.setupTransform()
            self.is_transform_cuda = True

        y = []
        for image in range(x.shape[0]):
            y.append( self.train_transform_randCrop(x[image]))

        #x= self.train_transform_randCrop(x)
        x = torch.stack(y, dim=0)
        return self.model(x)

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Training loss.
        """

        inputs1,inputs2 = batch[0], batch[1]

        result = self.forward(inputs1)
        outputs1 = result[-1]
        outputs2 = self.forward(inputs2)[-1]

        outputs1 = outputs1.flatten(start_dim=1, end_dim=4)
        outputs2 = outputs2.flatten(start_dim=1, end_dim=4)
        loss =  self.loss_fn(outputs1,outputs2)

        self.log("train_loss", loss, prog_bar=True,  sync_dist=True)
        return loss

    def validation_step(self, batch: dict, _: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        inputs1,inputs2 = batch[0], batch[1]

        result = self.forward(inputs1)
        outputs1 = result[-1]
        outputs2 = self.forward(inputs2)[-1]

        outputs1 = outputs1.flatten(start_dim=1, end_dim=4)
        outputs2 = outputs2.flatten(start_dim=1, end_dim=4)
        loss =  self.loss_fn(outputs1,outputs2)
        self.log("val_loss", loss, prog_bar=True,  sync_dist=True)
        return loss

    def test_step(self, batch: dict, _: int) -> torch.Tensor:
        """Evaluate the model on the test set.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Test loss.
        """
        inputs1,inputs2 = batch[0], batch[1]

        result = self.forward(inputs1)
        outputs1 = result[-1]
        outputs2 = self.forward(inputs2)[-1]

        outputs1 = outputs1.flatten(start_dim=1, end_dim=4)
        outputs2 = outputs2.flatten(start_dim=1, end_dim=4)
        loss =  self.loss_fn(outputs1,outputs2)
        self.log("test_loss", loss, prog_bar=True,  sync_dist=True)
        return loss

    def predict_step(self, batch: dict, _: int) -> torch.Tensor:
        """Generate predictions for a given batch during inference.

        This method is used during `Trainer.predict()` to produce model outputs,
        such as logits or predicted class indices.

        Args:
            batch (tuple): A tuple of input, (optional) label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Model predictions (e.g., predicted class indices).
        """
        x, idx = batch["image"], batch["index"]
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return {"idx": idx, "pred": preds}

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        """Set up the optimizer and scheduler.

        Returns:
            torch.optim.Optimizer | dict: The optimizer instance if no scheduler is used,
            or a dictionary containing the optimizer and scheduler if a scheduler is provided.
        """
        optimizer = self.optimizer(self.model.parameters())

        # If scheduler is provided, return the optimizer and scheduler
        if self.scheduler:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # You can also use "step" if you want the scheduler to step per batch
                    "frequency": 1,
                },
            }

        return optimizer
