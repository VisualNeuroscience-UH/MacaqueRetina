# Numerical
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, fourier_shift

# System
import psutil

# Machine learning and hyperparameter optimization
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

# import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
from torchsummary import summary

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Callback

from optuna.samplers import TPESampler


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
from collections import OrderedDict
from sys import exit

import warnings

warnings.filterwarnings(
    "ignore",
    message="Metric `Kernel Inception Distance`",
)
warnings.filterwarnings("ignore", message="FutureWarning:")


class HardDiskWatchDog(Callback):
    def __init__(self, output_path, disk_usage_threshold=90):
        self.disk_usage_threshold = disk_usage_threshold
        self.output_path = output_path

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if result["training_iteration"] % 100 == 0:
            disk_usage = psutil.disk_usage(str(self.output_path))
            if disk_usage.percent > self.disk_usage_threshold:
                print(
                    f"""
                    WARNING: disk_usage_threshold exceeded ({disk_usage.percent:.2f}%
                    Shutting down ray and exiting.
                    """
                )
                ray.shutdown()


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

    def __init__(
        self, data, labels, resolution_hw, augmentation_dict=None, data_multiplier=1.0
    ):
        if augmentation_dict is not None:
            # Multiply the amount of images by the data_multiplier. Take random samples from the data
            len_data = data.shape[0]

            # Get the number of images to be added
            n_images_to_add = int(data_multiplier * len_data) - len_data

            # Get the indices of the images to be added
            indices_to_add = np.random.choice(len_data, n_images_to_add, replace=True)

            # Get the images to be added
            data_to_add = data[indices_to_add]

            # Get the labels to be added
            labels_to_add = labels[indices_to_add]

            # Concatenate the data and labels
            data = np.concatenate((data, data_to_add), axis=0)
            labels = np.concatenate((labels, labels_to_add), axis=0)

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
                    transforms.Resize(resolution_hw, antialias=True),
                ]
            )

        else:
            self.transform = transforms.Compose(
                [transforms.Lambda(self._feature_scaling)]
            )

            if self.augmentation_dict["noise"] > 0:
                self.transform.transforms.append(transforms.Lambda(self._add_noise))

            if self.augmentation_dict["rotation"] > 0:
                self.transform.transforms.append(
                    transforms.Lambda(self._random_rotate_image)
                )

            if np.sum(self.augmentation_dict["translation"]) > 0:
                self.transform.transforms.append(
                    transforms.Lambda(self._random_shift_image)
                )

            self.transform.transforms.append(transforms.Lambda(self._to_tensor))

            if self.augmentation_dict["flip"] > 0:
                self.transform.transforms.append(
                    transforms.RandomHorizontalFlip(self.augmentation_dict["flip"])
                )
                self.transform.transforms.append(
                    transforms.RandomVerticalFlip(self.augmentation_dict["flip"])
                )

            self.transform.transforms.append(
                transforms.Resize(resolution_hw, antialias=True)
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
        Scale data to range [0, 1]]. Before scaling the data, the abs median value is close to 0.0,
        consistently =< 0.02 for both parasol and midget data.

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

        # Define the range of valid values for latent_dims
        min_dim = 2
        max_dim = 128

        # Check if latent_dims is a power of 2 bwteen min_dim and max_dim
        assert (
            (latent_dims & (latent_dims - 1) == 0)
            and (latent_dims >= min_dim)
            and (latent_dims <= max_dim)
        ), "Latent_dims must be a power of 2 between 2 and 128, aborting..."

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
            n_features=2048,
            reset_real_features=False,
            normalize=True,
            subset_size=16,
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
            "k9s1": {
                "kernel": 9,
                "stride": 1,
                "pad1": 4,
                "pad2": 4,
                "pad3": 4,
                "pad4": 4,
                "pad5": 4,
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
        fixed_params=None,
    ):
        # Assert that none of the optional arguments are None
        assert data_dict is not None, "val_ds is None, aborting..."
        assert device is not None, "device is None, aborting..."
        assert methods is not None, "methods is None, aborting..."

        self.train_data = data_dict["train_data"]
        self.train_labels = data_dict["train_labels"]
        self.val_data = data_dict["val_data"]
        self.val_labels = data_dict["val_labels"]
        self.test_data = data_dict["test_data"]
        self.test_labels = data_dict["test_labels"]

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": config.get("rotation"),
            "translation": (
                config.get("translation"),
                config.get("translation"),
            ),
            "noise": config.get("noise"),
            "flip": config.get("flip"),
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

        # Will be saved with checkpoint model
        self.model.test_data = self.test_data
        self.model.test_labels = self.test_labels

        self.model.to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr"), weight_decay=1e-5
        )
        # Define the scheduler with a step size and gamma factor
        self.scheduler = lr_scheduler.StepLR(
            self.optim,
            step_size=fixed_params["lr_step_size"],
            gamma=fixed_params["lr_gamma"],
        )

    def step(self):
        train_loss = self._train_epoch(
            self.model, self.device, self.train_loader, self.optim, self.scheduler
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
        # torch.save(self.model.state_dict(), checkpoint_path)
        torch.save(self.model, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = torch.load(checkpoint_path)


class RetinaVAE:
    """
    Class to apply variational autoencoder to Apricot retina data and run single learning run
    or Ray[Tune] hyperparameter search.

    Refereces for validation metrics:
    FID : Heusel_2017_NIPS
    KID : Binkowski_2018_ICLR
    SSIM : Wang_2009_IEEESignProcMag, Wang_2004_IEEETransImProc
    """

    def __init__(
        self,
        gc_type,
        response_type,
        training_mode,
        apricot_data_folder,
        output_folder=None,
        save_tuned_models=False,
    ):
        super().__init__()

        self.apricot_data_folder = apricot_data_folder
        self.gc_type = gc_type
        self.response_type = response_type

        # Fixed values for both single training and ray tune runs
        self.epochs = 5
        self.lr_step_size = 10  # Learning rate decay step size (in epochs)
        self.lr_gamma = 0.9  # Learning rate decay (multiplier for learning rate)
        # how many times to get the data, applied only if augmentation_dict is not None
        self.data_multiplier = 4

        # For ray tune only
        # If grid_search is True, time_budget is ignored
        self.time_budget = 60 * 60 * 24 * 4  # in seconds
        self.grid_search = True  # False for tune by Optuna, True for grid search
        self.grace_period = 50  # epochs. ASHA stops earliest at grace period.

        # TÄHÄN JÄIT: OPETTELE PENKOMAAN EXPRIMENT JSON. KANNATTANEE TUUNATA ILMAN CHECKPOINTTEJA ISOSTI. SEN JÄLKEEN EHKÄ
        # CHECKPOINTIT TAI YKSITTÄISET AJOT.

        #######################
        # Single run parameters
        #######################
        # Set common VAE model parameters
        self.latent_dim = 8  # 2**1 - 2**6, use powers of 2 btw 2 and 128
        self.channels = 8
        # lr will be reduced by scheduler down to lr * gamma ** (epochs/step_size)
        self.lr = 0.001
        # self._show_lr_decay(self.lr, self.lr_gamma, self.lr_step_size, self.epochs)

        self.batch_size = 128  # None will take the batch size from test_split size.
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)
        self.train_by = [["parasol"], ["on", "off"]]  # Train by these factors
        # self.train_by = [["midget"], ["on", "off"]]  # Train by these factors

        self.ksp = "k9s1"  # "k3s1", "k3s2" # "k5s2" # "k5s1"
        self.conv_layers = 3  # 1 - 5
        self.batch_norm = False

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": 0,  # rotation in degrees
            "translation": (
                0.07692307692307693,
                0.07692307692307693,
            ),  # fraction of image, (x, y) -directions
            "noise": 0.005,  # noise float in [0, 1] (noise is added to the image)
            "flip": 0.5,  # flip probability, both horizontal and vertical
        }
        self.augmentation_dict = augmentation_dict
        # self.augmentation_dict = None

        ####################
        # Utility parameters
        ####################

        # Set the random seed for reproducible results for both torch and numpy
        self.random_seed = np.random.randint(1, 10000)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.latent_space_plot_scale = 3.0  # Scale for plotting latent space

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.resolution_hw = (28, 28)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.this_folder = self._get_this_folder()
        self.models_folder = self._set_models_folder(output_folder=output_folder)
        self.ray_dir = self.models_folder / "ray_results"
        self.tb_log_folder = self.models_folder / "tb_logs"

        self.dependent_variables = [
            "train_loss",
            "val_loss",
            "mse",
            "ssim",
            "kid_std",
            "kid_mean",
        ]

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # # Visualize the augmentation effects and exit
        # self._visualize_augmentation()

        # # Create model and set optimizer and learning rate scheduler
        # self._prep_training()
        self._get_and_split_apricot_data()

        # # KID comparison btw real and fake images
        # self.check_kid_and_exit()

        match training_mode:
            case "train_model":
                # Create datasets and dataloaders

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
                # print(self.vae)

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
                # Note that initial_params under _set_ray_tuner MUST be included in the search space.
                # Grid search: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#ray.tune.grid_search
                # Sampling: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs
                self.search_space = {
                    "lr": [0.001],
                    "latent_dim": [2, 4, 8, 16, 32],
                    # k3s2,k3s1,k5s2,k5s1,k7s1, k9s1 Kernel-stride-padding for conv layers. NOTE you cannot use >3 conv layers with stride 2
                    "ksp": ["k7s1", "k9s1"],
                    "channels": [4, 8, 16],
                    "batch_size": [128],
                    "conv_layers": [1, 2, 3],
                    "batch_norm": [False],
                    "rotation": [0],  # Augment: max rotation in degrees
                    # Augment: fract of im, max in (x, y)/[xy] dir
                    "translation": [0.07692307692307693],  # 1/13 pixels
                    "noise": [0.005],  # Augment: noise added, btw [0., 1.]
                    "flip": [0.5],  # Augment: flip prob, both horiz and vert
                    "num_models": 4,  # repetitions of the same model
                }

                # The first metric is the one that will be used to prioritize the checkpoints and pruning.
                self.multi_objective = {
                    "metric": ["val_loss"],
                    "mode": ["min"],
                }

                # Fraction of GPU per trial. 0.25 for smaller models is enough. Larger may need 0.33 or 0.5.
                # Increase if you get CUDA out of memory errors.
                self.gpu_fraction = 1.0

                self.disk_usage_threshold = 90  # %, stops training if exceeded

                # Save tuned models. > 100 MB / model.
                self.save_tuned_models = save_tuned_models

                tuner = self._set_ray_tuner(grid_search=self.grid_search)
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

                self.best_result = self.result_grid.get_best_result(
                    metric="val_loss", mode="min"
                )
                print("Best result:", self.best_result)
                result_df = self.best_result.metrics_dataframe
                result_df[["training_iteration", "val_loss", "time_total_s"]]

                # Load model state dict from checkpoint to new self.vae and return the state dict.
                self.vae = self._load_model(best_result=self.best_result)

                self._update_vae_to_match_best_model(self.best_result)

                # Give one second to write the checkpoint to disk
                time.sleep(1)

            case "load_model":
                # Load previously calculated model for vizualization
                # Load model to self.vae and return state dict. The numbers are in the state dict.

                if hasattr(self, "trial_name"):
                    self.vae, result_grid, tb_dir = self._load_model(
                        model_path=None, trial_name=self.trial_name
                    )
                    # Dep vars: train_loss, val_loss, mse, ssim, kid_std, kid_mean,
                    self._plot_dependent_variables(
                        results_grid=result_grid,
                    )

                elif hasattr(self, "models_folder"):
                    self.vae = self._load_model(
                        model_path=self.models_folder, trial_name=None
                    )
                    # Get datasets for RF generation and vizualization
                    self.train_loader = self._augment_and_get_dataloader(
                        data_type="train"
                    )
                    self.val_loader = self._augment_and_get_dataloader(data_type="val")
                    self.test_loader = self._augment_and_get_dataloader(
                        data_type="test"
                    )
                else:
                    raise ValueError(
                        "No output path (models_folder) or trial name given, cannot load model, aborting..."
                    )

                # # Evoke new subprocess and run tensorboard at tb_dir folder
                # self._run_tensorboard(tb_dir=tb_dir)
                summary(
                    self.vae.to(self.device),
                    input_size=(1, self.resolution_hw[0], self.resolution_hw[1]),
                    batch_size=-1,
                )

                # # Dep vars: train_loss, val_loss, mse, ssim, kid_std, kid_mean,
                # self._plot_dependent_variable_mean_std(
                #     results_grid=result_grid,
                # )

        # This attaches test data to the model.
        self.test_loader = self._augment_and_get_dataloader(
            data_type="test", shuffle=False
        )
        if 1:
            # Figure 1
            self._plot_ae_outputs(
                self.vae.encoder,
                self.vae.decoder,
                ds_name="test_ds",
                sample_start_stop=[10, 25],
            )

            if training_mode in ["train_model"]:
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

            encoded_samples = self.get_encoded_samples(ds_name="test_ds")

            # Figure 3
            self._plot_latent_space(encoded_samples)

            # Figure 4
            self._plot_tsne_space(encoded_samples)

            if training_mode == "train_model":
                encoded_samples = self.get_encoded_samples(ds_name="train_ds")
                self._plot_latent_space(encoded_samples)
                self._plot_tsne_space(encoded_samples)

    def _show_lr_decay(self, lr, gamma, step_size, epochs):
        lrs = np.zeros(epochs)
        for this_epoch in range(epochs):
            lrs[this_epoch] = lr * gamma ** np.floor(this_epoch / step_size)
        plt.plot(lrs)
        plt.show()
        exit()

    def _update_vae_to_match_best_model(self, best_result):
        """
        Update the VAE to match the best model found by the tuner.
        """

        self.latent_dim = best_result.config["latent_dim"]
        self.channels = best_result.config["channels"]
        self.lr = best_result.config["lr"]

        self.batch_size = best_result.config["batch_size"]
        self.ksp = best_result.config["ksp"]
        self.conv_layers = best_result.config["conv_layers"]
        self.batch_norm = best_result.config["batch_norm"]

        self.augmentation_dict = {
            "rotation": best_result.config["rotation"],
            "translation": (
                best_result.config["translation"],
                best_result.config["translation"],
            ),
            "noise": best_result.config["noise"],
            "flip": best_result.config["flip"],
        }

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
                    try:
                        label = ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                    except:
                        raise NotImplementedError("Not implemented yet, aborting...")
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

                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
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

    def _set_ray_tuner(self, grid_search=True):
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

        trainable = tune.with_resources(TrainableVAE, {"gpu": self.gpu_fraction})
        trainable_with_parameters = tune.with_parameters(
            trainable,
            data_dict={
                "train_data": self.train_data,
                "train_labels": self.train_labels,
                "val_data": self.val_data,
                "val_labels": self.val_labels,
                "test_data": self.test_data,  # For later evaluation and viz
                "test_labels": self.test_labels,
            },
            device=self.device,
            methods={
                "_train_epoch": self._train_epoch,
                "_validate_epoch": self._validate_epoch,
                "_augment_and_get_dataloader": self._augment_and_get_dataloader,
            },
            fixed_params={
                "lr_step_size": self.lr_step_size,
                "lr_gamma": self.lr_gamma,
            },
        )

        if grid_search:
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
                "flip": tune.grid_search(self.search_space["flip"]),
                "model_id": tune.grid_search(
                    [
                        "model_{}".format(i)
                        for i in range(self.search_space["num_models"])
                    ]
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
            )
        else:
            # Note that the initial parameters must be included in the search space
            initial_params = [
                {
                    "lr": 0.001,
                    "latent_dim": 16,
                    "ksp": "k7s1",
                    "channels": 16,
                    "batch_size": 128,
                    "conv_layers": 2,
                    "batch_norm": False,
                    "rotation": 0,
                    "translation": 0,
                    "noise": 0.0,
                    "flip": 0.5,
                    "model_id": "model_0",
                }
            ]

            # tune (log)uniform etc require two positional arguments, so we need to unpack the list
            param_space = {
                "lr": tune.loguniform(
                    self.search_space["lr"][0], self.search_space["lr"][-1]
                ),
                "latent_dim": tune.choice(self.search_space["latent_dim"]),
                "ksp": tune.choice(self.search_space["ksp"]),
                "channels": tune.choice(self.search_space["channels"]),
                "batch_size": tune.choice(self.search_space["batch_size"]),
                "conv_layers": tune.choice(self.search_space["conv_layers"]),
                "batch_norm": tune.choice(self.search_space["batch_norm"]),
                "rotation": tune.uniform(
                    self.search_space["rotation"][0], self.search_space["rotation"][-1]
                ),
                "translation": tune.uniform(
                    self.search_space["translation"][0],
                    self.search_space["translation"][-1],
                ),
                "noise": tune.uniform(
                    self.search_space["noise"][0], self.search_space["noise"][-1]
                ),
                "flip": tune.uniform(
                    self.search_space["flip"][0], self.search_space["flip"][-1]
                ),
                "model_id": tune.choice(
                    [
                        "model_{}".format(i)
                        for i in range(self.search_space["num_models"])
                    ]
                ),
            }

            # Efficient hyperparameter selection. Search Algorithms are wrappers around open-source
            # optimization libraries. Each library has a specific way of defining the search space.
            # https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.tune.tune_config.TuneConfig
            tune_config = tune.TuneConfig(
                # Local optuna search will generate study name "optuna" indicating in-memory storage
                search_alg=OptunaSearch(
                    sampler=TPESampler(),
                    metric=self.multi_objective["metric"],
                    mode=self.multi_objective["mode"],
                    points_to_evaluate=initial_params,
                ),
                scheduler=ASHAScheduler(
                    time_attr="training_iteration",
                    metric=self.multi_objective["metric"][
                        0
                    ],  # Only 1st metric used for pruning
                    mode=self.multi_objective["mode"][0],
                    max_t=self.epochs,
                    grace_period=self.grace_period,
                    reduction_factor=2,
                ),
                time_budget_s=self.time_budget,
                num_samples=-1,
            )

        # Runtime configuration that is specific to individual trials. Will overwrite the run config passed to the
        # Trainer. for API, see https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.RunConfig

        if self.save_tuned_models is True:
            hard_disk_watchdog = [
                HardDiskWatchDog(
                    self.ray_dir, disk_usage_threshold=self.disk_usage_threshold
                )
            ]
        else:
            hard_disk_watchdog = None

        run_config = (
            air.RunConfig(
                stop={"training_iteration": self.epochs},
                progress_reporter=reporter,
                local_dir=self.ray_dir,
                callbacks=hard_disk_watchdog,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute=self.multi_objective["metric"][0],
                    checkpoint_score_order=self.multi_objective["mode"][0],
                    num_to_keep=1,
                    checkpoint_at_end=self.save_tuned_models,
                    checkpoint_frequency=0,
                ),
                verbose=1,
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

    def _set_models_folder(self, output_folder=None):
        """Set the folder where models are saved"""

        # If output_folder is Path instance or string, use it as models_folder
        if isinstance(output_folder, Path) or isinstance(output_folder, str):
            models_folder = output_folder
        else:
            models_folder = self.this_folder / "models"
        Path(models_folder).mkdir(parents=True, exist_ok=True)

        return models_folder

    def _save_model(self):
        """
        Save model for single trial, a.k.a. 'train_model' training_mode
        """

        model_path = f"{self.models_folder}/model_{self.timestamp}.pt"
        # Create models folder if it does not exist using pathlib

        # Get key VAE structural parameters and save them with the full model
        self.vae.config = {
            "latent_dims": self.latent_dim,
            "ksp_key": self.ksp,
            "channels": self.channels,
            "conv_layers": self.conv_layers,
        }
        print(f"Saving model to {model_path}")
        # torch.save(self.vae.state_dict(), model_path)
        torch.save(self.vae, model_path)

        return model_path

    def _load_model(self, model_path=None, best_result=None, trial_name=None):
        """Load model if exists. Use either model_path, best_result, or trial_name to load model"""

        if best_result is not None:
            # ref https://medium.com/distributed-computing-with-ray/simple-end-to-end-ml-from-selection-to-serving-with-ray-tune-and-ray-serve-10f5564d33ba
            log_dir = best_result.log_dir
            checkpoint_dir = [
                d for d in os.listdir(log_dir) if d.startswith("checkpoint")
            ][0]
            checkpoint_path = os.path.join(log_dir, checkpoint_dir, "model.pth")

            vae_model = torch.load(checkpoint_path)

        elif model_path is not None:
            model_path = Path(model_path)
            if Path.exists(model_path) and model_path.is_file():
                print(
                    f"Loading model from {model_path}. \nWARNING: This will replace the current model in-place."
                )
                vae_model = torch.load(model_path)
            elif Path.exists(model_path) and model_path.is_dir():
                try:
                    model_path = max(Path(self.models_folder).glob("*.pt"))
                    vae_model = torch.load(model_path)
                    print(f"Most recent model is {model_path}.")
                except ValueError:
                    raise FileNotFoundError("No model files found. Aborting...")

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

            # Load the model from the checkpoint folder.
            try:
                checkpoint_folder_name = [
                    p for p in Path(correct_trial_folder).glob("checkpoint_*")
                ][0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find checkpoint folder in {correct_trial_folder}. Aborting..."
                )
            model_path = Path.joinpath(checkpoint_folder_name, "model.pth")
            vae_model = torch.load(model_path)

            # Move new model to same device as the input data
            vae_model.to(self.device)
            return vae_model, results, correct_trial_folder

        else:
            # Get the most recent model. Max recognizes the timestamp with the largest value
            try:
                model_path = max(Path(self.models_folder).glob("*.pt"))
                print(f"Most recent model is {model_path}.")
            except ValueError:
                raise FileNotFoundError("No model files found. Aborting...")
            vae_model = torch.load(model_path)

        vae_model.to(self.device)

        return vae_model

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
            train_val_data,
            train_val_labels,
            self.resolution_hw,
            augmentation_dict=self.augmentation_dict,
        )

        # Do not augment test data
        test_ds = AugmentedDataset(
            test_data, test_labels, self.resolution_hw, augmentation_dict=None
        )

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

        Returns:
            dataloader (torch.utils.data.DataLoader): dataloader
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
            data_multiplier=self.data_multiplier,
        )

        # set self. attribute "n_train", "n_val" or "n_test"
        setattr(self, "n_" + data_type, len(data_ds))

        # set self. attribute "train_ds", "val_ds" or "test_ds"
        setattr(self, data_type + "_ds", data_ds)

        data_loader = DataLoader(data_ds, batch_size=batch_size, shuffle=shuffle)

        return data_loader

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
            bad_data_idx = apricot_data.manually_picked_bad_data_idx
            (
                gc_spatial_data_np_orig,
                _,
            ) = apricot_data.read_spatial_filter_data()

            # Drop bad data
            gc_spatial_data_np = np.delete(
                gc_spatial_data_np_orig, bad_data_idx, axis=0
            )

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

    def _prep_training(self):
        self.vae = VariationalAutoencoder(
            latent_dims=self.latent_dim,
            ksp_key=self.ksp,
            channels=self.channels,
            conv_layers=self.conv_layers,
            batch_norm=self.batch_norm,
            device=self.device,
        )

        # Will be saved with model for later eval and viz
        self.vae.test_data = self.test_data
        self.vae.test_labels = self.test_labels

        self.optim = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Define the scheduler with a step size and gamma factor
        self.scheduler = lr_scheduler.StepLR(
            self.optim, step_size=self.lr_step_size, gamma=self.lr_gamma
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
        self.tb_log_folder = Path(self.tb_log_folder)

        # Create a folder for the experiment tensorboard logs
        Path.mkdir(self.tb_log_folder, parents=True, exist_ok=True)

        # Clear files and folders under exp_folder
        for f in self.tb_log_folder.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

        # This creates new scalar/time series line in tensorboard
        self.writer = SummaryWriter(str(self.tb_log_folder), max_queue=100)

    ### Training function
    def _train_epoch(self, vae, device, dataloader, optimizer, scheduler):
        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)

        for x, _ in dataloader:
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

        # Update the learning rate at the end of each epoch
        scheduler.step()

        train_loss_out = float(train_loss)
        del train_loss, loss, x, x_hat

        return train_loss_out / len(dataloader.dataset)

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

                vae.kid.update(x_expanded, real=True)  # KID
                vae.kid.update(x_hat_expanded, real=False)  # KID

        # # CUDA memory management
        # del x, _, x_hat, loss, x_expanded, x_hat_expanded
        # # # Delete attribute kl from vae.encoder.kl
        # # delattr(vae.encoder, "kl")

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
        Plot the outputs of the autoencoder.
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
                self.vae, self.device, self.train_loader, self.optim, self.scheduler
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
                Learning rate: {self.optim.param_groups[0]['lr']:.3e}
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
                    torch.stack(torch.meshgrid(grid, grid, indexing="xy"))
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
            plt.xlabel("EncVariable 0")
            plt.ylabel("EncVariable 1")

    def get_encoded_samples(self, ds_name="test_ds"):
        """Get encoded samples from a dataset.

        Parameters
        ----------
        ds_name : str, optional
            Dataset name, by default "test_ds"

        Returns
        -------
        pd.DataFrame
            Encoded samples
        """

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
                f"EncVariable {i}": enc for i, enc in enumerate(encoded_img)
            }
            encoded_sample["label"] = label
            encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)

        return encoded_samples

    def _plot_latent_space(self, encoded_samples):
        sns.relplot(
            data=encoded_samples,
            x="EncVariable 0",
            y="EncVariable 1",
            hue=encoded_samples.label.astype(str),
        )
        plt.title("Encoded samples")

    def _plot_tsne_space(self, encoded_samples):
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)

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

    def check_kid_and_exit(self):
        """
        Check KernelInceptionDistance between real and fake data and exit.
        """

        def kid_compare(dataloader_real, dataloader_fake, n_features=64):
            # Set evaluation mode for encoder and decoder
            kid = KernelInceptionDistance(
                n_features=n_features,
                reset_real_features=True,
                normalize=True,
                subset_size=16,
            )

            kid.reset()
            kid.to(self.device)

            with torch.no_grad():  # No need to track the gradients
                # for x, _ in dataloader_real:
                for real_batch, fake_batch in zip(dataloader_real, dataloader_fake):
                    # Move tensor to the proper device
                    real_img_batch = real_batch[0].to(self.device)
                    fake_img_batch = fake_batch[0].to(self.device)
                    # Expand dim 1 to 3 for x and x_hat
                    real_img_batch_expanded = real_img_batch.expand(-1, 3, -1, -1)
                    fake_img_batch_hat_expanded = fake_img_batch.expand(-1, 3, -1, -1)

                    kid.update(real_img_batch_expanded, real=True)  # KID
                    kid.update(fake_img_batch_hat_expanded, real=False)  # KID

            kid_mean_epoch, kid_std_epoch = kid.compute()

            return kid_mean_epoch, kid_std_epoch

        # self.train_by = [["parasol"], ["on", "off"]]
        # self._get_and_split_apricot_data()
        dataloader_real = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=None,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # self.train_by = [["midget"], ["on", "off"]]
        # self._get_and_split_apricot_data()
        dataloader_fake = self._augment_and_get_dataloader(
            data_type="train",
            augmentation_dict=self.augmentation_dict,
            # augmentation_dict=None,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # dataloader_fake = dataloader_real

        kid_mean, kid_std = kid_compare(dataloader_real, dataloader_fake, n_features=64)
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 64 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features=192
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 192 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features=768
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 768 features")

        kid_mean, kid_std = kid_compare(
            dataloader_real, dataloader_fake, n_features="2048"
        )
        print(f"KID mean: {kid_mean}, KID std: {kid_std} for 2048 features")

        exit()


if __name__ == "__main__":
    pass
