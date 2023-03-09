# Numerical
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, fourier_shift


# Pytorch
# import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError

# import torch._dynamo as dynamo
from torchsummary import summary

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


# Viz
from tqdm import tqdm
import seaborn as sns

# Local
from retina.apricot_data_module import ApricotData

# Builtin
from pathlib import Path
from datetime import datetime
import shutil
import pdb
from copy import deepcopy
from itertools import product
import os
import time
import subprocess
import itertools
from collections import OrderedDict
from sys import exit


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Apricot dataset class for Pytorch.

    The constructor reads the data from the ApricotData class and stores it as
    tensors of shape (n_cells, channels, height, width). While the constructor
    is called with particular gc_type and response_type, all data is retrieved
    and thus the __getitem__ method can be called with any index. This enables
    teaching the network with all data. The gc_type and response_type are, however,
    logged into the ApricotDataset instance object.
    """

    # TODO: Add noise and rectangular dropauts to the augmentation dictionary and pass it to the transforms.Compose() method.
    # RandomErasing([p, scale, ratio, value, inplace]) -> Randomly selects a rectangle region in an image and erases its pixels.

    def __init__(self, data, labels, resolution_hw, augmentation_dict=None):

        self.data = data
        self.labels = self._to_tensor(labels)

        self.augmentation_dict = augmentation_dict

        # Calculate mean and std of data
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)

        # Define transforms
        if self.augmentation_dict is None:
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(self._feature_scaling),
                    transforms.Lambda(self._to_tensor),
                    transforms.Resize(resolution_hw),
                ]
            )

        else:
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(self._feature_scaling),
                    transforms.Lambda(self._add_noise),
                    transforms.Lambda(self._random_rotate_image),
                    transforms.Lambda(self._random_shift_image),
                    transforms.Lambda(self._to_tensor),
                    transforms.Resize(resolution_hw),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Ideally, no data pre-processing steps should be coded anywhere in the whole model training pipeline but for this method.
        """

        image = self.data[idx]
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def _feature_scaling(self, data):
        """
        Scale data to range [0, 1]]

        Parameters
        ----------
        data : np.ndarray
            Data to be scaled

        Returns
        -------
        data_scaled : np.ndarray
            Scaled data
        """

        feature_range = (0, 1)
        feat_min, feat_max = feature_range
        data_std = (data - data.min()) / (data.max() - data.min())
        data_scaled = data_std * (feat_max - feat_min) + feat_min

        return data_scaled

    def _add_noise(self, image):
        """
        Add noise to the input images.

        Parameters
        ----------
        image : np.ndarray
            Input image
        noise_factor : float
            Noise factor

        Returns
        -------
        image_noise : np.ndarray

        """
        noise_factor = self.augmentation_dict["noise"]
        noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
        image_noise = np.clip(image + noise, -3.0, 3.0)

        return image_noise

    def _add_noise_t(self, image):
        """
        Add noise to the input images.

        Parameters
        ----------
        image : torch.Tensor
            Input image
        noise_factor : float
            Noise factor

        Returns
        -------
        image_noise : torch.Tensor
        """
        noise_factor = self.augmentation_dict["noise"]
        noise = torch.randn_like(image) * noise_factor
        image_noise = torch.clamp(image + noise, -3.0, 3.0)

        return image_noise

    def _random_rotate_image(self, image):
        """
        Rotate image by a random angle.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        image_rot : np.ndarray
            Rotated image
        """
        rot = self.augmentation_dict["rotation"]
        # Take random rot as float
        rot = np.random.uniform(-rot, rot)
        image_rot = rotate(image, rot, axes=(2, 1), reshape=False, mode="reflect")
        return image_rot

    def _random_shift_image(self, image):
        """
        Shift image by a random amount.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        image_shift : np.ndarray
            Shifted image
        """
        shift_proportions = self.augmentation_dict["translation"]

        shift_max = (
            int(image.shape[1] * shift_proportions[0]),
            int(image.shape[2] * shift_proportions[1]),
        )  # shift in pixels, tuple of (y, x) shift

        # Take random shift as float
        shift = (
            np.random.uniform(-shift_max[0], shift_max[0]),
            np.random.uniform(-shift_max[1], shift_max[1]),
        )

        input_ = np.fft.fft2(np.squeeze(image))
        result = fourier_shift(
            input_, shift=shift
        )  # shift in pixels, tuple of (y, x) shift

        result = np.fft.ifft2(result)
        image_shifted = result.real
        # Expand 0:th dimension
        image_shifted = np.expand_dims(image_shifted, axis=0)

        return image_shifted

    def _to_tensor(self, image):
        image_t = torch.from_numpy(deepcopy(image)).float()  # to float32
        return image_t


class VariationalEncoder(nn.Module):
    """
    Original implementation from Eugenia Anello (https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b)
    """

    def __init__(
        self,
        latent_dims,
        ksp=None,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        device=None,
    ):

        # super(VariationalEncoder, self).__init__()
        super().__init__()
        if ksp is None:
            ksp = {
                "kernel": 3,
                "stride": 1,
                "pad1": 1,
                "pad2": 1,
                "pad3": 0,
                "conv3_sidelen": 3,
            }

        self.device = device

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=ksp["kernel"],
                stride=ksp["stride"],
                padding=ksp["pad1"],
            ),
            nn.ReLU(True),
        )

        # Make an OrderedDict to feed into nn.Sequential containing the convolutional layers
        conv_layers_2toN = OrderedDict()
        for i in range(conv_layers - 1):
            n_channels = channels * 2**i
            conv_layers_2toN["conv" + str(i + 2)] = nn.Conv2d(
                n_channels,
                n_channels * 2,
                kernel_size=ksp["kernel"],
                stride=ksp["stride"],
                padding=ksp[f"pad{i + 2}"],
            )
            # Add one batch norm layer after second convolutional layer
            # parametrize 0 if need to put b-layer after other conv layers
            if batch_norm and i == 0:  # batch_norm is np.bool_ type, "is True" fails
                conv_layers_2toN["batch" + str(i + 2)] = nn.BatchNorm2d(n_channels * 2)

            conv_layers_2toN["relu" + str(i + 2)] = nn.ReLU(True)

        # OrderedDict works when it is the only argument to nn.Sequential
        self.encoder_conv2toN = nn.Sequential(conv_layers_2toN)

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Linear(
            int(
                ksp["conv3_sidelen"]
                * ksp["conv3_sidelen"]
                * channels
                * 2 ** (conv_layers - 1)
            ),
            128,
        )

        self.linear2 = nn.Linear(128, latent_dims)  # mu
        self.linear3 = nn.Linear(128, latent_dims)  # sigma

        self.N = torch.distributions.Normal(0, 1)
        if device is not None and device.type == "cpu":
            self.N.loc = self.N.loc.cpu()
            self.N.scale = self.N.scale.cpu()
        elif device is not None and device.type == "cuda":
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)

        x = self.encoder_conv1(x)
        x = self.encoder_conv2toN(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        mu = self.linear2(x)
        sigma = torch.exp(
            self.linear3(x)
        )  # The exp is an auxiliary activation to lin layer to ensure positive sigma
        z = mu + sigma * self.N.sample(mu.shape)
        # OLD self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum(), from orig ref@medium, see above
        # Ref Kingma_2014_arXiv
        self.kl = -0.5 * torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )

        return z


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        ksp=None,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        device=None,
    ):
        super().__init__()

        if ksp is None:
            ksp = {
                "kernel": 3,
                "stride": 1,
                "pad1": 1,
                "pad2": 1,
                "pad3": 0,
                "conv3_sidelen": 3,
            }

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(
                128,
                ksp["conv3_sidelen"]
                * ksp["conv3_sidelen"]
                * channels
                * 2 ** (conv_layers - 1),
            ),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(
                channels * 2 ** (conv_layers - 1),
                ksp["conv3_sidelen"],
                ksp["conv3_sidelen"],
            ),
        )

        # Make an OrderedDict to feed into nn.Sequential containing the deconvolutional layers
        deconv_layers_Nto2 = OrderedDict()
        conv_layers_list = list(range(conv_layers - 1))
        for i in conv_layers_list:
            n_channels = channels * 2 ** (conv_layers - i - 1)
            deconv_layers_Nto2["deconv" + str(conv_layers - i)] = nn.ConvTranspose2d(
                n_channels,
                n_channels // 2,
                kernel_size=ksp["kernel"],
                stride=ksp["stride"],
                padding=ksp[f"pad{conv_layers - i}"],
                output_padding=ksp[f"opad{conv_layers - i}"],
            )

            # parametrize the -1 if you want to change b-layer
            # batch_norm is np.bool_ type, "is True" fails
            if batch_norm and i == conv_layers_list[-1]:
                deconv_layers_Nto2["batch" + str(conv_layers - i)] = nn.BatchNorm2d(
                    n_channels // 2
                )
            deconv_layers_Nto2["relu" + str(conv_layers - i)] = nn.ReLU(True)

        self.decoder_convNto2 = nn.Sequential(deconv_layers_Nto2)

        self.decoder_end = nn.ConvTranspose2d(
            channels,
            1,
            kernel_size=ksp["kernel"],
            stride=ksp["stride"],
            padding=ksp["pad1"],
            output_padding=ksp["opad1"],
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_convNto2(x)
        x = self.decoder_end(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        ksp_key=None,
        channels=8,
        conv_layers=3,
        batch_norm=True,
        device=None,
    ):
        super().__init__()

        # Assert that conv_layers is an integer between 1 and 5
        assert isinstance(
            conv_layers, int
        ), "conv_layers must be an integer, aborting..."
        assert conv_layers >= 1, "conv_layers must be >= 1, aborting..."
        assert conv_layers <= 5, "conv_layers must be <= 5, aborting..."
        if conv_layers > 3:
            assert (
                "s2" not in ksp_key
            ), "stride 2 only supports conv_layers <= 3, check ksp, aborting..."

        self._set_ksp_key()
        ksp = self.ksp_keys[ksp_key]
        self.device = device
        self.encoder = VariationalEncoder(
            latent_dims=latent_dims,
            ksp=ksp,
            channels=channels,
            conv_layers=conv_layers,
            batch_norm=batch_norm,
            device=self.device,
        )
        self.decoder = Decoder(
            latent_dims,
            ksp,
            channels=channels,
            conv_layers=conv_layers,
            batch_norm=batch_norm,
            device=self.device,
        )

        # Consider moving for not to unnecessarily print the kid model
        self.mse = MeanSquaredError()
        # self.fid = FrechetInceptionDistance(
        #     n_features=64, reset_real_features=False, normalize=True
        # )
        # Allowed n_features: 64, 192, 768, 2048
        self.kid = KernelInceptionDistance(
            n_features=192, reset_real_features=False, normalize=True, subset_size=4
        )
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)
        z = self.encoder(x)

        return self.decoder(z)

    def _set_ksp_key(self):
        """
        Preset conv2D kernel, stride and padding values for kernel 3 and 5 and for
        reduction (28*28 => 3*3) and preservation (28*28 => 28*28) of representation size.
        """

        self.ksp_keys = {
            "k3s2": {
                "kernel": 3,
                "stride": 2,
                "pad1": 1,
                "pad2": 1,
                "pad3": 0,
                "conv3_sidelen": 3,
                "opad3": 0,
                "opad2": 1,
                "opad1": 1,
            },
            "k5s2": {
                "kernel": 5,
                "stride": 2,
                "pad1": 2,
                "pad2": 2,
                "pad3": 1,
                "conv3_sidelen": 3,
                "opad3": 0,
                "opad2": 1,
                "opad1": 1,
            },
            "k3s1": {
                "kernel": 3,
                "stride": 1,
                "pad1": 1,
                "pad2": 1,
                "pad3": 1,
                "pad4": 1,
                "pad5": 1,
                "conv3_sidelen": 28,
                "opad5": 0,
                "opad4": 0,
                "opad3": 0,
                "opad2": 0,
                "opad1": 0,
            },
            "k5s1": {
                "kernel": 5,
                "stride": 1,
                "pad1": 2,
                "pad2": 2,
                "pad3": 2,
                "pad4": 2,
                "pad5": 2,
                "conv3_sidelen": 28,
                "opad5": 0,
                "opad4": 0,
                "opad3": 0,
                "opad2": 0,
                "opad1": 0,
            },
            "k7s1": {
                "kernel": 7,
                "stride": 1,
                "pad1": 3,
                "pad2": 3,
                "pad3": 3,
                "pad4": 3,
                "pad5": 3,
                "conv3_sidelen": 28,
                "opad5": 0,
                "opad4": 0,
                "opad3": 0,
                "opad2": 0,
                "opad1": 0,
            },
        }


class TrainableVAE(tune.Trainable):
    """
    Tune will convert this class into a Ray actor, which runs on a separate process.
    By default, Tune will also change the current working directory of this process to
    its corresponding trial-level log directory self.logdir. This is designed so that
    different trials that run on the same physical node wont accidently write to the same
    location and overstep each other

    Generally you only need to implement setup, step, save_checkpoint, and load_checkpoint when subclassing Trainable.

    Accessing config through Trainable.setup

    Return metrics from Trainable.step

    https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-trainable-class-api
    """

    def setup(
        self,
        config,
        data_dict=None,
        device=None,
        methods=None,
    ):

        # Assert that none of the optional arguments are None
        assert data_dict is not None, "val_ds is None, aborting..."
        assert device is not None, "device is None, aborting..."
        assert methods is not None, "methods is None, aborting..."

        self.train_data = data_dict["train_data"]
        self.train_labels = data_dict["train_labels"]
        self.val_data = data_dict["val_data"]
        self.val_labels = data_dict["val_labels"]

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": config.get("rotation"),  # rotation in degrees
            "translation": (
                config.get("translation"),
                config.get("translation"),
            ),  # fraction of image, (x, y) -directions
            "noise": config.get(
                "noise"
            ),  # noise float in [0, 1] (noise is added to the image)
        }

        self._augment_and_get_dataloader = methods["_augment_and_get_dataloader"]

        self.train_loader = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=augmentation_dict,
            batch_size=config.get("batch_size"),
            shuffle=True,
        )

        self.val_loader = self._augment_and_get_dataloader(
            data_type="val",
            augmentation_dict=augmentation_dict,
            batch_size=config.get("batch_size"),
            shuffle=True,
        )

        self.device = device
        self._train_epoch = methods["_train_epoch"]
        self._validate_epoch = methods["_validate_epoch"]

        self.model = VariationalAutoencoder(
            latent_dims=config.get("latent_dim"),
            ksp_key=config.get("ksp"),
            channels=config.get("channels"),
            conv_layers=config.get("conv_layers"),
            batch_norm=config.get("batch_norm"),
            device=self.device,
        )
        self.model.to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr"), weight_decay=1e-5
        )

    def step(self):
        train_loss = self._train_epoch(
            self.model, self.device, self.train_loader, self.optim
        )

        (
            val_loss_epoch,
            mse_loss_epoch,
            ssim_loss_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        ) = self._validate_epoch(self.model, self.device, self.val_loader)

        # Convert to float, del & empty cache to free GPU memory
        train_loss_out = float(train_loss)
        val_loss_out = float(val_loss_epoch)
        mse_loss_out = float(mse_loss_epoch)
        ssim_loss_out = float(ssim_loss_epoch)
        kid_mean_out = float(kid_mean_epoch)
        kid_std_out = float(kid_std_epoch)

        del (
            train_loss,
            val_loss_epoch,
            mse_loss_epoch,
            ssim_loss_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        )
        torch.cuda.empty_cache()

        return {
            "iteration": self.iteration
            + 1,  # Do not remove, plus one for 0=>1 indexing
            "train_loss": train_loss_out,
            "val_loss": val_loss_out,
            "mse": mse_loss_out,
            "ssim": ssim_loss_out,
            "kid_mean": kid_mean_out,
            "kid_std": kid_std_out,
        }

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


class RetinaVAE:
    """
    Class to apply variational autoencoder to Apricot retina data and run single learning run
    or Ray[Tune] hyperparameter search.

    Refereces for validation metrics:
    FID : Heusel_2017_NIPS
    KID : Binkowski_2018_ICLR
    SSIM : Wang_2009_IEEESignProcMag, Wang_2004_IEEETransImProc
    """

    def __init__(self, apricot_data_folder, gc_type, response_type):
        super().__init__()

        self.apricot_data_folder = apricot_data_folder
        self.gc_type = gc_type
        self.response_type = response_type

        # Set common VAE model parameters
        self.latent_dim = 2
        self.channels = 16
        self.latent_space_plot_scale = 3.0  # Scale for plotting latent space
        self.lr = 0.0001

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.resolution_hw = (28, 28)

        self.batch_size = 64  # None will take the batch size from test_split size.
        self.epochs = 5
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)
        self.train_by = [["parasol"], ["on", "off"]]  # Train by these factors

        self.this_folder = self._get_this_folder()
        self.models_folder = self._set_models_folder()
        self.ray_dir = self.this_folder / "ray_results"
        self.dependent_variables = [
            "train_loss",
            "val_loss",
            "mse",
            "ssim",
            "kid_std",
            "kid_mean",
        ]

        self.ksp = "k7s1"  # "k3s1", "k3s2" # "k5s2" # "k5s1"
        self.conv_layers = 3
        self.batch_norm = True

        # TÄHÄN JÄIT:
        # implementoi tuneen uudet hyperparametrit
        # TARVITSEEKO LISÄTÄ PRECISION JA RECALL? IMPLEMENTAATIO.
        # tune until sun runs out of hydrogen

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": 15.0,  # rotation in degrees
            "translation": (0.0, 0.0),  # fraction of image, (x, y) -directions
            "noise": 0.0,  # noise float in [0, 1] (noise is added to the image)
        }
        self.augmentation_dict = augmentation_dict
        # self.augmentation_dict = None

        # Set the random seed for reproducible results for both torch and numpy
        self.random_seed = np.random.randint(1, 10000)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")

        # # Visualize the augmentation effects and exit
        # self._visualize_augmentation()

        # # Create datasets and dataloaders
        # # self._prep_minst_data()

        # # Create model and set optimizer and learning rate scheduler
        # self._prep_training()
        self._get_and_split_apricot_data()

        training_mode = "load_model"  # "train_model" or "tune_model" or "load_model"

        match training_mode:
            case "train_model":
                # Create datasets and dataloaders
                # self._prep_apricot_data()
                # self._prep_minst_data()

                self.train_loader = self._augment_and_get_dataloader(
                    data_type="train",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                self.val_loader = self._augment_and_get_dataloader(
                    data_type="val",
                    augmentation_dict=self.augmentation_dict,
                    batch_size=self.batch_size,
                    shuffle=True,
                )

                # Create model and set optimizer and learning rate scheduler
                self._prep_training()
                print(self.vae)

                # Init tensorboard
                self.tb_log_folder = "tb_logs"
                self._prep_tensorboard_logging()

                # Train
                self._train()
                self.writer.flush()
                self.writer.close()

                # Save model
                model_path = self._save_model()
                summary(
                    self.vae,
                    input_size=(1, self.resolution_hw[0], self.resolution_hw[1]),
                    batch_size=-1,
                )

            case "tune_model":

                # This will be captured at _set_ray_tuner
                # Search space of the tuning job. Both preprocessor and dataset can be tuned here.
                # Use grid search to try out all values for each parameter. values: iterable
                # Grid search: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#ray.tune.grid_search
                # Sampling: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs
                self.search_space = {
                    "lr": [0.0001],
                    "latent_dim": [2],
                    "ksp": [
                        "k5s1",
                    ],  # k3s2,k3s1,k5s2,k5s1,k7s1 Kernel-stride-padding for conv layers. NOTE you cannot use >3 conv layers with stride 2
                    "channels": [16],
                    "batch_size": [64],
                    "conv_layers": [2, 4],
                    "batch_norm": [True, False],  # becomes np.bool type
                    "rotation": [15],  # Augment: max rotation in degrees
                    "translation": [0],  # Augment: fract of im, max in (x, y)/[xy] dir
                    "noise": [0],  # Augment: noise float in [0, 1] (noise added)
                    "num_models": 1,  # repetitions of the same model
                }

                tuner = self._set_ray_tuner()
                self.result_grid = tuner.fit()

                results_df = self.result_grid.get_dataframe()
                print(
                    "Shortest training time:",
                    results_df["time_total_s"].min(),
                    "for config:",
                    results_df[
                        results_df["time_total_s"] == results_df["time_total_s"].min()
                    ].index.values,
                )
                print(
                    "Longest training time:",
                    results_df["time_total_s"].max(),
                    "for config:",
                    results_df[
                        results_df["time_total_s"] == results_df["time_total_s"].max()
                    ].index.values,
                )

                best_result = self.result_grid.get_best_result(
                    metric="val_loss", mode="min"
                )
                print("Best result:", best_result)
                result_df = best_result.metrics_dataframe
                result_df[["training_iteration", "val_loss", "time_total_s"]]

                # Load model state dict from checkpoint to new self.vae and return the state dict.
                state_dict = self._load_model(best_result=best_result)

                # Give one second to write the checkpoint to disk
                time.sleep(1)

            case "load_model":
                # Load previously calculated model for vizualization
                # Load model to self.vae and return state dict. The numbers are in the state dict.
                # my_model_path = "C:\Users\simov\Laskenta\GitRepos\MacaqueRetina\retina\models" # For single trials from "train_model"
                trial_name = "TrainableVAE_7a5cb_00000"  # From ray_results table/folder
                state_dict, result_grid, tb_dir = self._load_model(
                    model_path=None, trial_name=trial_name
                )
                # # Evoke new subprocess and run tensorboard at tb_dir folder
                # self._run_tensorboard(tb_dir=tb_dir)
                summary(
                    self.vae,
                    input_size=(1, self.resolution_hw[0], self.resolution_hw[1]),
                    batch_size=-1,
                )

                # Dep vars: train_loss, val_loss, mse, ssim, kid_std, kid_mean,
                self._plot_dependent_variables(
                    results_grid=result_grid,
                )

                # # Dep vars: train_loss, val_loss, mse, ssim, kid_std, kid_mean,
                # self._plot_dependent_variable_mean_std(
                #     results_grid=result_grid,
                # )

                print(result_grid)

        self.test_loader = self._augment_and_get_dataloader(
            data_type="test", shuffle=False
        )

        # Figure 1
        self._plot_ae_outputs(
            self.vae.encoder,
            self.vae.decoder,
            ds_name="test_ds",
            sample_start_stop=[10, 25],
        )

        if training_mode == "train_model":
            self._plot_ae_outputs(
                self.vae.encoder, self.vae.decoder, ds_name="train_ds"
            )
            self._plot_ae_outputs(
                self.vae.encoder, self.vae.decoder, ds_name="valid_ds"
            )

        self.vae.eval()

        # Figure 2
        self._reconstruct_random_images()

        self._reconstruct_grid_images()

        encoded_samples = self._get_encoded_samples(ds_name="test_ds")

        # Figure 3
        self._plot_latent_space(encoded_samples)

        # Figure 4
        self._plot_tsne_space(encoded_samples)

        if training_mode == "train_model":
            encoded_samples = self._get_encoded_samples(ds_name="train_ds")
            self._plot_latent_space(encoded_samples)
            self._plot_tsne_space(encoded_samples)

    def _plot_dependent_variables(self, results_grid):
        """Plot results from ray tune"""

        df = results_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        # # remove "model_id" from the varied columns
        # varied_cols.remove("model_id")

        num_colors = len(results_grid.get_dataframe())
        colors = plt.cm.get_cmap("tab20", num_colors).colors

        # Make one subplot for each dependent variable
        # List of dependent variables
        dep_vars = self.dependent_variables

        nrows = 2
        ncols = len(dep_vars) // 2
        plt.figure(figsize=(ncols * 5, nrows * 5))

        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = plt.subplot(nrows, ncols, idx + 1)

            for result in results_grid:
                if idx == 0:
                    label = ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                    legend = True
                else:
                    legend = False

                result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=legend,
                )
                color_idx += 1
            ax.set_title(f"{dep_var}")
            ax.set_ylabel(dep_var)
            ax.grid(True)

    def _run_tensorboard(self, tb_dir):
        """Run tensorboard in a new subprocess"""
        subprocess.run(
            [
                "tensorboard",
                "--logdir",
                tb_dir,
                "--host",
                "localhost",
                "--port",
                "6006",
            ]
        )

    def _set_ray_tuner(self):
        """Set ray tuner"""

        # List of strings from the self.search_space dictionary which should be reported.
        # Include only the parameters which have more than one item listed in the search space.
        parameters_to_report = []
        for key, value in self.search_space.items():
            if key == "num_models":
                continue
            if len(value) > 1:
                parameters_to_report.append(key)

        print(f"parameters_to_report: {parameters_to_report}")
        reporter = CLIReporter(
            metric_columns=[
                "time_total_s",
                "iteration",
                "train_loss",
                "val_loss",
                "mse",
                "ssim",
                "kid_mean",
                "kid_std",
            ],
            parameter_columns=parameters_to_report,
        )

        trainable = tune.with_resources(TrainableVAE, {"gpu": 0.25})
        trainable_with_parameters = tune.with_parameters(
            trainable,
            data_dict={
                "train_data": self.train_data,
                "train_labels": self.train_labels,
                "val_data": self.val_data,
                "val_labels": self.val_labels,
            },
            device=self.device,
            methods={
                "_train_epoch": self._train_epoch,
                "_validate_epoch": self._validate_epoch,
                "_augment_and_get_dataloader": self._augment_and_get_dataloader,
            },
        )

        # NUM_MODELS = 2
        param_space = {
            "lr": tune.grid_search(self.search_space["lr"]),
            "latent_dim": tune.grid_search(self.search_space["latent_dim"]),
            "ksp": tune.grid_search(self.search_space["ksp"]),
            "channels": tune.grid_search(self.search_space["channels"]),
            "batch_size": tune.grid_search(self.search_space["batch_size"]),
            "conv_layers": tune.grid_search(self.search_space["conv_layers"]),
            "batch_norm": tune.grid_search(self.search_space["batch_norm"]),
            "rotation": tune.grid_search(self.search_space["rotation"]),
            "translation": tune.grid_search(self.search_space["translation"]),
            "noise": tune.grid_search(self.search_space["noise"]),
            "model_id": tune.grid_search(
                ["model_{}".format(i) for i in range(self.search_space["num_models"])]
            ),
        }

        # Efficient hyperparameter selection. Search Algorithms are wrappers around open-source
        # optimization libraries. Each library has a
        # specific way of defining the search space.
        # https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.tune.tune_config.TuneConfig
        tune_config = tune.TuneConfig(
            search_alg=tune.search.basic_variant.BasicVariantGenerator(
                constant_grid_search=True,
            ),
            # metric="kid_std",
            # mode="max",
        )

        # Runtime configuration that is specific to individual trials. Will overwrite the run config passed to the Trainer.
        # for API, see https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.RunConfig
        run_config = (
            air.RunConfig(
                stop={"training_iteration": self.epochs},
                progress_reporter=reporter,
                local_dir=self.ray_dir,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=True,
                    num_to_keep=1,  # Keep only the best checkpoint
                ),
                verbose=2,
            ),
        )

        tuner = tune.Tuner(
            trainable_with_parameters,
            param_space=param_space,
            run_config=run_config[0],
            tune_config=tune_config,
        )

        return tuner

    def _get_this_folder(self):
        """Get the folder where this module file is located"""
        from retina import vae_module as vv

        this_folder = Path(vv.__file__).parent
        return this_folder

    def _set_models_folder(self):
        """Set the folder where models are saved"""
        models_folder = self.this_folder / "models"
        Path(models_folder).mkdir(parents=True, exist_ok=True)

        return models_folder

    def _save_model(self):
        """
        Save model for single trial, a.k.a. 'train_model' training_mode
        """
        model_path = f"{self.models_folder}/model_{self.timestamp}.pt"
        # Create models folder if it does not exist using pathlib

        print(f"Saving model to {model_path}")
        torch.save(self.vae.state_dict(), model_path)
        return model_path

    def _load_model(self, model_path=None, best_result=None, trial_name=None):
        """Load model if exists. Use either model_path, best_result, or trial_name to load model"""

        # if not hasattr(self, "vae"):
        #     # Note that if you start parametrically vary the model architecture, you need to save the model architecture as well or
        #     # rebuild it here (c.f. latent_dims, ksp_key)
        #     self.vae = VariationalAutoencoder(
        #         latent_dims=self.latent_dim,
        #         ksp_key="k3s2",  # kernel size 3, stride 2
        #         channels=8,
        #         device=self.device,
        #     )

        if best_result is not None:
            # ref https://medium.com/distributed-computing-with-ray/simple-end-to-end-ml-from-selection-to-serving-with-ray-tune-and-ray-serve-10f5564d33ba
            log_dir = best_result.log_dir
            checkpoint_dir = [
                d for d in os.listdir(log_dir) if d.startswith("checkpoint")
            ][0]
            checkpoint_path = os.path.join(log_dir, checkpoint_dir, "model.pth")

            latent_dim = best_result.config["latent_dim"]
            ksp = best_result.config["ksp"]
            channels = best_result.config["channels"]
            conv_layers = best_result.config["conv_layers"]
            batch_norm = best_result.config["batch_norm"]
            # Get model with correct layer dimensions
            model = VariationalAutoencoder(
                latent_dims=latent_dim,
                ksp_key=ksp,
                channels=channels,
                conv_layers=conv_layers,
                batch_norm=batch_norm,
                device=self.device,
            )
            model.load_state_dict(torch.load(checkpoint_path))
            self.latent_dim = latent_dim
            self.channels = channels
            self.vae = model.to(self.device)

        elif model_path is not None:

            self.vae = VariationalAutoencoder(
                latent_dims=self.latent_dim,
                ksp_key=self.ksp,
                channels=self.channels,
                conv_layers=self.conv_layers,
                batch_norm=self.batch_norm,
                device=self.device,
            )

            if not Path(model_path).exists():

                # Get the most recent model. Max recognizes the timestamp with the largest value
                try:
                    model_path = max(Path(self.models_folder).glob("*.pt"))
                    print(f"Most recent model is {model_path}.")
                except ValueError:
                    raise FileNotFoundError("No model files found. Aborting...")

            else:
                model_path = Path(model_path)

            if Path.exists(model_path):
                print(
                    f"Loading model from {model_path}. \nWARNING: This will replace the current model in-place."
                )
                self.vae.load_state_dict(torch.load(model_path))
            else:
                print(f"Model {model_path} does not exist.")

        elif trial_name is not None:
            # trial_name = "TrainableVAE_XXX" from ray.tune results table.
            # Search under self.ray_dir for folder with the trial name. Under that folder,
            # there should be a checkpoint folder which contains the model.pth file.
            try:
                correct_trial_folder = [
                    p for p in Path(self.ray_dir).glob(f"**/") if trial_name in p.stem
                ][0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find trial with name {trial_name}. Aborting..."
                )
            # However, we need to first check that the model dimensionality is correct.
            # This will be hard coded for checking and changing only latent_dim.
            # More versatile version is necessary if other dimensions are searched.

            # Get the results as dataframe from the ray directory / correct run
            results_folder = correct_trial_folder.parents[0]
            tuner = tune.Tuner.restore(str(results_folder))
            results = tuner.get_results()
            df = results.get_dataframe()

            # Check the latent_dim, change if necessary and update the model
            new_latent_dim = df[df["logdir"] == str(correct_trial_folder)][
                "config/latent_dim"
            ].values[0]
            # check new ksp and update
            new_ksp = df[df["logdir"] == str(correct_trial_folder)][
                "config/ksp"
            ].values[0]
            new_channels = int(
                df[df["logdir"] == str(correct_trial_folder)]["config/channels"].values[
                    0
                ]
            )
            new_conv_layers = int(
                df[df["logdir"] == str(correct_trial_folder)][
                    "config/conv_layers"
                ].values[0]
            )
            new_batch_norm = df[df["logdir"] == str(correct_trial_folder)][
                "config/batch_norm"
            ].values[0]

            self.latent_dim = new_latent_dim
            self.vae = VariationalAutoencoder(
                latent_dims=self.latent_dim,
                ksp_key=new_ksp,
                channels=new_channels,
                conv_layers=new_conv_layers,
                batch_norm=new_batch_norm,
                device=self.device,
            )

            # Load the model from the checkpoint folder.
            checkpoint_folder_name = [
                p for p in Path(correct_trial_folder).glob("checkpoint_*")
            ][0]
            model_path = Path.joinpath(checkpoint_folder_name, "model.pth")

            # pdb.set_trace()
            self.vae.load_state_dict(torch.load(model_path))

            # Move new model to same device as the input data
            self.vae.to(self.device)
            return self.vae.state_dict(), results, correct_trial_folder

        return self.vae.state_dict()

    def _visualize_augmentation(self):
        """
        Visualize the augmentation effects and exit
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = self._get_spatial_apricot_data()

        # Split to training, validation and testing
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data_np,
            labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels_np,
        )

        # Augment training and validation data
        train_val_ds = AugmentedDataset(
            train_val_data, train_val_labels, augmentation_dict=self.augmentation_dict
        )

        # Do not augment test data
        test_ds = AugmentedDataset(test_data, test_labels, augmentation_dict=None)

        # Split into train and validation
        train_ds, val_ds = random_split(
            train_val_ds,
            [
                int(np.round(len(train_val_ds) * (1 - self.test_split))),
                int(np.round(len(train_val_ds) * self.test_split)),
            ],
        )

        # Get n items for the three sets
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)
        self.n_test = len(test_ds)

        # Make a figure with 2 rows and 5 columns, with upper row containing 5 original and the lower row 5 augmented images.
        # The original images are in the train_val_data (numpy array with dims (N x C x H x W)), and the augmented images are in the train_val_ds
        # (torch dataset with dims (N x C x H x W)). The labels are in train_val_labels (numpy array with dims (N, 1)).
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))
        for i in range(5):
            axs[0, i].imshow(train_val_data[i, 0, :, :], cmap="gray")
            axs[0, i].axis("off")
            axs[1, i].imshow(train_val_ds[i][0][0, :, :], cmap="gray")
            axs[1, i].axis("off")

        # Set the labels as text upper left inside the images of the upper row
        for i in range(5):
            axs[0, i].text(
                0.05,
                0.85,
                self.apricot_data.data_labels2names_dict[train_val_labels[i][0]],
                fontsize=10,
                color="blue",
                transform=axs[0, i].transAxes,
            )

        # Set subtitle "Original" for the first row
        axs[0, 0].set_title("Original", fontsize=14)
        # Set subtitle "Augmented" for the second row
        axs[1, 0].set_title("Augmented", fontsize=14)

        plt.show()
        exit()

    def _get_and_split_apricot_data(self):
        """
        Load data
        Split into training, validation and testing
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = self._get_spatial_apricot_data()

        # Split to training+validation and testing
        (
            train_val_data_np,
            test_data_np,
            train_val_labels_np,
            test_labels_np,
        ) = train_test_split(
            data_np,
            labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels_np,
        )

        # Split into train and validation
        train_data_np, val_data_np, train_labels_np, val_labels_np = train_test_split(
            train_val_data_np,
            train_val_labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=train_val_labels_np,
        )

        # These are all numpy arrays
        self.train_data = train_data_np
        self.train_labels = train_labels_np
        self.val_data = val_data_np
        self.val_labels = val_labels_np
        self.test_data = test_data_np
        self.test_labels = test_labels_np

    def _augment_and_get_dataloader(
        self, data_type="train", augmentation_dict=None, batch_size=32, shuffle=True
    ):
        """
        Augmenting data
        Creating dataloaders

        Parameters:
            data_type (str): "train", "val" or "test"
            augmentation_dict (dict): augmentation dictionary
            batch_size (int): batch size
            shuffle (bool): shuffle data
        """

        # Assert that data_type is "train", "val" or "test"
        assert data_type in [
            "train",
            "val",
            "test",
        ], "data_type must be 'train', 'val' or 'test', aborting..."

        # Assert that self. has attribute "train_data", "val_data" or "test_data"
        assert hasattr(self, data_type + "_data"), (
            "\nself has no attribute '" + data_type + "_data', aborting...\n"
        )

        data = getattr(self, data_type + "_data")
        labels = getattr(self, data_type + "_labels")

        # Augment training and validation data
        data_ds = AugmentedDataset(
            data,
            labels,
            self.resolution_hw,
            augmentation_dict=augmentation_dict,
        )

        # set self. attribute "n_train", "n_val" or "n_test"
        setattr(self, "n_" + data_type, len(data_ds))

        # set self. attribute "train_ds", "val_ds" or "test_ds"
        setattr(self, data_type + "_ds", data_ds)

        data_loader = DataLoader(data_ds, batch_size=batch_size, shuffle=shuffle)

        # # set self. attribute "train_loader", "val_loader" or "test_loader"
        # setattr(self, data_type + "_loader", data_loader)

        return data_loader

    def _prep_apricot_data(self, tune_augmentation=False):
        """
        OBSOLETE, REPLACED BY _get_and_split_apricot_data AND _augment_and_get_dataloader
        Prep apricot data for training. This includes:
        - Loading data
        - Splitting into training, validation and testing
        - Augmenting data
        - Preprocessing data
        - Creating dataloaders
        """

        # Get numpy data
        data_np, labels_np, data_names2labels_dict = self._get_spatial_apricot_data()

        # Split to training, validation and testing
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data_np,
            labels_np,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels_np,
        )

        if tune_augmentation is True:
            self.train_val_data = train_val_data
            self.train_val_labels = train_val_labels

        # Augment training and validation data
        train_val_ds = AugmentedDataset(
            train_val_data,
            train_val_labels,
            self.resolution_hw,
            augmentation_dict=self.augmentation_dict,
        )

        # Do not augment test data
        test_ds = AugmentedDataset(
            test_data,
            test_labels,
            self.resolution_hw,
            augmentation_dict=None,
        )

        test_ds.labels = test_ds.labels  # MNIST uses targets instead of labels
        self.test_ds = test_ds

        # Split into train and validation
        train_ds, val_ds = random_split(
            train_val_ds,
            [
                int(np.round(len(train_val_ds) * (1 - self.test_split))),
                int(np.round(len(train_val_ds) * self.test_split)),
            ],
        )

        # Save for later use, this may be a problem with bigger datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.train_ds.labels = train_val_ds.labels[self.train_ds.indices]
        self.val_ds.labels = train_val_ds.labels[self.val_ds.indices]

        # Get n items for the three sets
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)
        self.n_test = len(test_ds)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _get_spatial_apricot_data(self):
        """
        Get spatial ganglion cell data from file using the apricot_data method read_spatial_filter_data().
        All data is returned, the requested data is looged in the class attributes gc_type and response_type.

        Returns
        -------
        collated_gc_spatial_data_np : np.ndarray
            Spatial data with shape (n_gc, 1, ydim, xdim)
        collated_gc_spatial_labels_np : np.ndarray
            Labels with shape (n_gc, 1)
        data_names2labels_dict : dict
            Dictionary with gc names as keys and labels as values
        """

        # Get requested data
        self.apricot_data = ApricotData(
            self.apricot_data_folder, self.gc_type, self.response_type
        )

        # Log requested label
        self.gc_label = self.apricot_data.data_names2labels_dict[
            f"{self.gc_type}_{self.response_type}"
        ]

        # We train by more data, however.
        # Build a list of combinations of gc types and response types from self.train_by
        train_by_combinations = [
            f"{gc}_{response}"
            for (gc, response) in product(self.train_by[0], self.train_by[1])
        ]

        response_labels = [
            self.apricot_data.data_names2labels_dict[key]
            for key in train_by_combinations
        ]

        # Log trained_by labels
        self.train_by_labels = response_labels

        # Get all available gc types and response types
        gc_types = [key[: key.find("_")] for key in train_by_combinations]
        response_types = [key[key.find("_") + 1 :] for key in train_by_combinations]

        # Initialise numpy arrays to store data
        collated_gc_spatial_data_np = np.empty(
            (
                0,
                1,
                self.apricot_data.metadata["data_spatialfilter_height"],
                self.apricot_data.metadata["data_spatialfilter_width"],
            )
        )
        collated_labels_np = np.empty((0, 1), dtype=int)

        # Get data for learning
        for gc_type, response_type, label in zip(
            gc_types, response_types, response_labels
        ):
            print(f"Loading data for {gc_type}_{response_type} (label {label})")
            apricot_data = ApricotData(self.apricot_data_folder, gc_type, response_type)

            (
                gc_spatial_data_np_orig,
                _,
                bad_data_indices,
            ) = apricot_data.read_spatial_filter_data()

            # Drop bad data
            gc_spatial_data_np = np.delete(
                gc_spatial_data_np_orig, bad_data_indices, axis=2
            )

            # Reshape  pytorch (n_samples, 1, xdim, ydim)
            gc_spatial_data_np = np.moveaxis(gc_spatial_data_np, 2, 0)
            gc_spatial_data_np = np.expand_dims(gc_spatial_data_np, axis=1)

            # Collate data
            collated_gc_spatial_data_np = np.concatenate(
                (collated_gc_spatial_data_np, gc_spatial_data_np), axis=0
            )
            labels = np.full((gc_spatial_data_np.shape[0], 1), label)
            collated_labels_np = np.concatenate((collated_labels_np, labels), axis=0)

        return (
            collated_gc_spatial_data_np,
            collated_labels_np,
            apricot_data.data_names2labels_dict,
        )

    def _prep_minst_data(self):
        data_dir = "dataset"

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        _train_ds = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        _test_ds = torchvision.datasets.MNIST(
            data_dir, train=False, download=True, transform=test_transform
        )

        # Cut for testing. It is disappointing how complex this needs to be.
        train_indices = torch.arange(6000)
        test_indices = torch.arange(1000)
        train_val_ds = Subset(_train_ds, train_indices)
        test_ds = Subset(_test_ds, test_indices)
        test_ds.labels = torch.from_numpy(
            np.fromiter((_test_ds.targets[i] for i in test_indices), int)
        )  # Add targets for the plotting

        self.test_ds = test_ds

        m = len(train_val_ds)

        train_ds, val_ds = random_split(train_val_ds, [int(m - m * 0.2), int(m * 0.2)])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _prep_training(self):

        self.vae = VariationalAutoencoder(
            latent_dims=self.latent_dim,
            ksp_key=self.ksp,
            channels=self.channels,
            conv_layers=self.conv_layers,
            batch_norm=self.batch_norm,
            device=self.device,
        )

        self.optim = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, weight_decay=1e-5
        )

        print(f"Selected device: {self.device}")
        self.vae.to(self.device)

    def _prep_tensorboard_logging(self):
        """
        Prepare local folder environment for tensorboard logging and model building

        Note that the tensoboard reset takes place only when quitting the terminal call
        to tensorboard. You will see old graph, old scalars, if they are not overwritten
        by new ones.
        """

        # Create a folder for the experiment tensorboard logs
        exp_folder = self.this_folder.joinpath(self.tb_log_folder)
        Path.mkdir(exp_folder, parents=True, exist_ok=True)

        # Clear files and folders under exp_folder
        for f in exp_folder.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

        # This creates new scalar/time series line in tensorboard
        self.writer = SummaryWriter(str(exp_folder), max_queue=100)

    ### Training function
    def _train_epoch(self, vae, device, dataloader, optimizer):
        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)

        # for x, _ in dataloader: # MNIST
        for x, _ in dataloader:  # Apricot
            # Move tensor to the proper device
            x = x.to(device)
            x_hat = vae(x)
            # Evaluate loss
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # # Print batch loss
            # print("\t partial train loss (single batch): %f" % (loss.item()))
            train_loss += loss.item()

        return train_loss / len(dataloader.dataset)

    ### Testing function
    def _validate_epoch(self, vae, device, dataloader):
        # Set evaluation mode for encoder and decoder
        vae.eval()
        val_loss = 0.0
        vae.mse.reset()
        vae.ssim.reset()
        vae.kid.reset()

        with torch.no_grad():  # No need to track the gradients
            for x, _ in dataloader:
                # Move tensor to the proper device
                x = x.to(device)
                x_hat = vae(x)
                loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
                val_loss += loss.item()

                vae.mse.update(x_hat, x)  # MSE
                vae.ssim.update(x_hat, x)  # SSIM

                # Expand dim 1 to 3 for x and x_hat
                x_expanded = x.expand(-1, 3, -1, -1)
                x_hat_expanded = x_hat.expand(-1, 3, -1, -1)

                vae.kid.update(x_hat_expanded, real=False)  # KID
                vae.kid.update(x_hat_expanded, real=True)  # KID

        n_samples = len(dataloader.dataset)
        val_loss_epoch = val_loss / n_samples
        mse_loss_epoch = vae.mse.compute()
        ssim_loss_epoch = vae.ssim.compute()
        kid_mean_epoch, kid_std_epoch = vae.kid.compute()

        # Test all output variables for type, and covert to value if needed
        if isinstance(val_loss_epoch, torch.Tensor):
            val_loss_epoch = val_loss_epoch.item()
        if isinstance(mse_loss_epoch, torch.Tensor):
            mse_loss_epoch = mse_loss_epoch.item()
        if isinstance(ssim_loss_epoch, torch.Tensor):
            ssim_loss_epoch = ssim_loss_epoch.item()
        if isinstance(kid_mean_epoch, torch.Tensor):
            kid_mean_epoch = kid_mean_epoch.item()
        if isinstance(kid_std_epoch, torch.Tensor):
            kid_std_epoch = kid_std_epoch.item()

        return (
            val_loss_epoch,
            mse_loss_epoch,
            ssim_loss_epoch,
            kid_mean_epoch,
            kid_std_epoch,
        )

    def _plot_ae_outputs(
        self, encoder, decoder, ds_name="test_ds", sample_start_stop=[0, 10]
    ):
        """
        Plot the outputs of the autoencoder, one for each label.
        """

        if ds_name == "train_ds":
            ds = self.train_ds
        elif ds_name == "valid_ds":
            ds = self.val_ds
        else:
            ds = self.test_ds

        plt.figure(figsize=(16, 4.5))
        targets = ds.labels.numpy()
        # t_idx = {i: np.where(targets == i)[0][0] for i in self.train_by_labels}
        t_idx = {i: np.where(targets == i)[0][:] for i in self.train_by_labels}
        encoder.eval()
        decoder.eval()

        n_cell_types = len(self.train_by_labels)
        samples = np.arange(sample_start_stop[0], sample_start_stop[1])

        for pos_idx, sample_idx in enumerate(samples):
            # this_idx = t_idx[self.train_by_labels[i]]

            ax = plt.subplot(2, len(samples), pos_idx + 1)
            img = ds[sample_idx][0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                rec_img = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.text(
                0.05,
                0.85,
                self.apricot_data.data_labels2names_dict[ds[sample_idx][1].item()],
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )
            if pos_idx == 0:
                ax.set_title("Original images")

            ax = plt.subplot(2, len(samples), len(samples) + pos_idx + 1)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                ax.set_title("Reconstructed images")

        # Set the whole figure title as ds_name
        plt.suptitle(ds_name)

    def _train(self):
        """
        Train for training_mode = train_model
        """
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(
                self.vae, self.device, self.train_loader, self.optim
            )
            (
                val_loss_epoch,
                mse_loss_epoch,
                ssim_loss_epoch,
                kid_mean_epoch,
                kid_std_epoch,
            ) = self._validate_epoch(self.vae, self.device, self.val_loader)

            # For every 100th epoch, print the outputs of the autoencoder
            # if epoch == 0 or epoch % 100 == 0:
            print(
                f""" 
                EPOCH {epoch + 1}/{self.epochs} \t train_loss {train_loss:.3f} \t val loss {val_loss_epoch:.3f}
                mse {mse_loss_epoch:.3f} \t ssim {ssim_loss_epoch:.3f} \t kid mean {kid_mean_epoch:.3f} \t kid std {kid_std_epoch:.3f}
                """
            )

            # Convert to float, del & empty cache to free GPU memory
            train_loss_out = float(train_loss)
            val_loss_out = float(val_loss_epoch)
            mse_loss_out = float(mse_loss_epoch)
            ssim_loss_out = float(ssim_loss_epoch)
            kid_mean_out = float(kid_mean_epoch)
            kid_std_out = float(kid_std_epoch)

            del (
                train_loss,
                val_loss_epoch,
                mse_loss_epoch,
                ssim_loss_epoch,
                kid_mean_epoch,
                kid_std_epoch,
            )
            torch.cuda.empty_cache()

            # Add train loss and val loss to tensorboard SummaryWriter
            self.writer.add_scalars(
                f"Training_{self.timestamp}",
                {
                    "loss/train": train_loss_out,
                    "loss/val": val_loss_out,
                    "mse/val": mse_loss_out,
                    "ssim/val": ssim_loss_out,
                    "kid_mean/val": kid_mean_out,
                    "kid_std/val": kid_std_out,
                },
                epoch,
            )

    def _reconstruct_random_images(self):
        with torch.no_grad():

            # # sample latent vectors from the normal distribution
            # latent = torch.randn(128, self.latent_dim, device=self.device)

            scale = self.latent_space_plot_scale
            # sample latent vectors from the uniform distribution between -scale and scale
            latent = (
                torch.rand(128, self.latent_dim, device=self.device) * 2 * scale - scale
            )

            # reconstruct images from the latent vectors
            img_recon = self.vae.decoder(latent)
            img_recon = img_recon.cpu()
            latent = latent.cpu()

            fig, ax = plt.subplots(figsize=(20, 8.5))
            # plot 100 images in 10x10 grid with 1 pixel padding in-between.
            self._show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 1))
            ax.set_title("Decoded images from a random sample of latent space")

    def _reconstruct_grid_images(self):

        if self.latent_dim == 2:
            with torch.no_grad():
                scale = self.latent_space_plot_scale
                # sample grid of vectors between -1.5 and 1.5 in both dimensions
                grid = torch.linspace(-scale, scale, 10)
                latent = (
                    torch.stack(torch.meshgrid(grid, grid))
                    .reshape(2, -1)
                    .T.to(self.device)
                )

                # reconstruct images from the latent vectors
                img_recon = self.vae.decoder(latent)
                img_recon = img_recon.cpu()
                latent = latent.cpu()

                fig, ax = plt.subplots(figsize=(20, 8.5))
                # plot 100 images in 10x10 grid with 1 pixel padding in-between.
                self._show_image(
                    torchvision.utils.make_grid(
                        img_recon.data[:100], nrow=10, padding=1
                    ),
                    latent,
                )
                ax.set_title("Decoded images from a grid of samples of latent space")
        else:
            print("Latent dimension is not 2. Multidim grid plot is not implemented.")

    def _show_image(self, img, latent=None):
        npimg = img.numpy()
        # Enc 0 as x-axis, 1 as y-axis
        npimg_transposed = np.transpose(npimg, (2, 1, 0))
        sidelength = int(npimg_transposed.shape[0] / 10)
        npimg_transposed = np.flip(npimg_transposed, 0)  # flip the image ud
        plt.imshow(npimg_transposed)
        plt.xticks([])
        plt.yticks([])
        if latent is not None:
            # Make x and y ticks from latent space (rows, cols) values
            x_ticks = np.linspace(latent[:, 1].min(), latent[:, 1].max(), 10)
            y_ticks = np.linspace(latent[:, 0].max(), latent[:, 0].min(), 10)
            # Limit both x and y ticks to 2 significant digits
            x_ticks = np.around(x_ticks, 2)
            y_ticks = np.around(y_ticks, 2)
            plt.xticks(
                np.arange(0 + sidelength / 2, sidelength * 10, sidelength), x_ticks
            )
            plt.yticks(
                np.arange(0 + sidelength / 2, sidelength * 10, sidelength), y_ticks
            )
            # X label and Y label
            plt.xlabel("Enc. Variable 0")
            plt.ylabel("Enc. Variable 1")

    def _get_encoded_samples(self, ds_name="test_ds"):

        if ds_name == "train_ds":
            ds = self.train_ds
        elif ds_name == "valid_ds":
            ds = self.valid_ds
        else:
            ds = self.test_ds

        encoded_samples = []
        for sample in tqdm(ds):
            img = sample[0].unsqueeze(0).to(self.device)
            label = self.apricot_data.data_labels2names_dict[sample[1].item()]
            # Encode image
            self.vae.eval()
            with torch.no_grad():
                encoded_img = self.vae.encoder(img)
            # Append to list
            encoded_img = encoded_img.flatten().cpu().numpy()
            encoded_sample = {
                f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)
            }
            encoded_sample["label"] = label
            encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)

        return encoded_samples

    def _plot_latent_space(self, encoded_samples):
        sns.relplot(
            data=encoded_samples,
            x="Enc. Variable 0",
            y="Enc. Variable 1",
            hue=encoded_samples.label.astype(str),
        )
        plt.title("Encoded samples")

    def _plot_tsne_space(self, encoded_samples):
        tsne = TSNE(n_components=2)

        if encoded_samples.shape[0] < tsne.perplexity:
            tsne.perplexity = encoded_samples.shape[0] - 1

        tsne_results = tsne.fit_transform(encoded_samples.drop(["label"], axis=1))

        ax0 = sns.relplot(
            # data=tsne_results,
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=encoded_samples.label.astype(str),
        )
        ax0.set(xlabel="tsne-2d-one", ylabel="tsne-2d-two")
        plt.title("TSNE plot of encoded samples")


if __name__ == "__main__":

    pass
