from RRDB.src.RRDB import LightGenerator
from data.Dataset import DataModule

import dask.array as da
import yaml
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

import torch
from lightning.pytorch.callbacks import Callback
import wandb
import matplotlib.pyplot as plt
import numpy as np

class LogSRImagesCallback(Callback):
    """
    Callback to log High-Resolution (HR) and Super-Resolved (SR) images with afmhot colormap to Weights & Biases during validation.
    """
    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Logs the SR and HR images with afmhot colormap at the end of a validation batch.
        """
        # Only log every n steps to avoid logging too many images
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        # Unpack the batch
        lr, hr = batch
        
        # Ensure model is in evaluation mode and no gradients are computed
        pl_module.eval()
        with torch.no_grad():
            sr = pl_module(lr)

        # Move images to CPU and convert to numpy for colormap application
        lr = lr.detach().cpu().numpy()
        hr = hr.detach().cpu().numpy()
        sr = sr.detach().cpu().numpy()

        # Apply afmhot colormap to the first image in the batch for each of LR, HR, and SR
        lr_image = self.apply_colormap(lr[0, 0], cmap='afmhot')
        hr_image = self.apply_colormap(hr[0, 0], cmap='afmhot')
        sr_image = self.apply_colormap(sr[0, 0], cmap='afmhot')

        # Log images to wandb
        trainer.logger.experiment.log({
            "Validation/LR Image": wandb.Image(lr_image, caption="Low Resolution (LR)"),
            "Validation/HR Image": wandb.Image(hr_image, caption="High Resolution (HR)"),
            "Validation/SR Image": wandb.Image(sr_image, caption="Super Resolved (SR)"),
            "global_step": trainer.global_step
        })

        # Set the model back to training mode if necessary
        pl_module.train()

    def apply_colormap(self, img, cmap='afmhot'):
        """
        Applies a colormap to a grayscale image.
        Args:
            img (numpy.ndarray): Grayscale image to apply the colormap to.
            cmap (str): Name of the colormap to apply.
        Returns:
            numpy.ndarray: RGB image with the colormap applied.
        """
        # Normalize image to [0, 1] range
        img_normalized = (img - img.min()) / (img.max() - img.min())

        # Apply colormap using matplotlib
        colormap = plt.get_cmap(cmap)
        colored_img = colormap(img_normalized)

        # Convert from RGBA to RGB and scale to [0, 255]
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

        return colored_img

    def state_dict(self):
        # Return an empty state dict if no state needs saving
        return {}

    def load_state_dict(self, state_dict):
        # Method needed for callback state loading compatibility
        pass



if __name__ == "__main__":
    # Run the CLI main function which handles the setup and execution
    wandb_logger = WandbLogger(log_model="all")

    generator = LightGenerator()
    datamodule = DataModule()
    wandb_logger.watch(generator, log='all', log_freq=5)
    trainer = Trainer(logger=wandb_logger,
                      callbacks=[LogSRImagesCallback()])
    trainer.fit(generator, datamodule)
