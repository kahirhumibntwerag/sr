import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L

class LightGenerator(L.LightningModule):
    """
    LightningModule that defines the Generator network for solar images with 1 channel
    and handles training, validation, and testing.
    """
    def __init__(self, in_channels=1, initial_channel=64, num_rrdb_blocks=4, upscale_factor=4, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, initial_channel, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # RRDB blocks
        self.rrdbs = nn.Sequential(*[self._make_rrdb(initial_channel) for _ in range(num_rrdb_blocks)])

        # Post-residual blocks
        self.post_rrdb = nn.Sequential(
            nn.Conv2d(initial_channel, initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            *[nn.Conv2d(initial_channel, 4*initial_channel, kernel_size=3, stride=1, padding=1),
              nn.PixelShuffle(2),
              nn.PReLU()] * int(np.log2(upscale_factor))
        )

        # Output layer
        self.output = nn.Conv2d(initial_channel, in_channels, kernel_size=9, stride=1, padding=4)

        # Example input for logging
        self.example_input_array = torch.Tensor(1, in_channels, 128, 128)

    def forward(self, x):
        # Forward pass through the network
        initial = self.initial(x)
        rrdbs = self.rrdbs(initial)
        post_rrdb = self.post_rrdb(rrdbs + initial)
        upsampled = self.upsampling(post_rrdb)
        return self.output(upsampled)

    def _make_rrdb(self, in_features, num_dense_layers=3):
        """
        Creates a Residual in Residual Dense Block (RRDB)
        """
        return nn.Sequential(*[self._make_residual_block(in_features) for _ in range(num_dense_layers)])

    def _make_residual_block(self, in_features):
        """
        Creates a Residual Block without Batch Normalization
        """
        return nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        )

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        train_loss = F.mse_loss(sr, hr)
        self.log('train_loss', train_loss, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        val_loss = F.mse_loss(sr, hr)
        self.log('validation_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        test_loss = F.mse_loss(sr, hr)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
