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

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler

# import torch.utils.data.Subset as Subset

# import torch.optim as optim

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


# TÄHÄN JÄIT. TEE FID, FIT STATS, SSIM SEKÄ MEAN SD , HIST MEASURES KUTEN KL DIVERGENCE.
# NÄIDEN AVULLA NÄET KVANTITATIIVISESTI MITEN HYVIN ML TOIMII


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

        # # Define transforms
        # if self.augmentation_dict is None:
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.Lambda(self._to_tensor),
        #             transforms.Normalize(mean=data_mean, std=data_std),
        #             transforms.Resize(resolution_hw),
        #         ]
        #     )
        # else:
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.Lambda(self._random_rotate_image),
        #             transforms.Lambda(self._random_shift_image),
        #             transforms.Lambda(self._to_tensor),
        #             transforms.Normalize(mean=data_mean, std=data_std),
        #             transforms.Resize(resolution_hw),
        #             transforms.Lambda(self._add_noise_t),
        #         ]
        #     )
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
        # print(f"rot={rot:.2f}")
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
        # print(f"shift={shift[0]:.2f}, {shift[1]:.2f}")

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
        image_t = torch.from_numpy(image).float()  # to float32
        return image_t


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        # super(VariationalEncoder, self).__init__()
        super().__init__()

        self.device = device
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3 * 3 * 32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        # self.N.loc = self.N.loc.cpu()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cpu()
        self.kl = 0

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super().__init__()

        self.device = device
        self.encoder = VariationalEncoder(latent_dims=latent_dims, device=self.device)
        # self.encoder = self.variational_encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        # self.decoder = self.variational_decoder(latent_dims)

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)


def objective(x, a, b):
    return a * (x**0.5) + b


class Trainable(tune.Trainable):
    def setup(self, config: dict):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def step(self):  # This is called iteratively.
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}


class MyTrainableClass(tune.Trainable):
    def setup(self, config):
        self.model = nn.Sequential(
            nn.Linear(config.get("input_size", 32), 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def step(self):
        return {}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


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
    """

    def setup(self, config, data=None):
        # self.model = nn.Sequential(
        #     nn.Linear(config.get("input_size", 32), 32), nn.ReLU(), nn.Linear(32, 10)
        # )
        self.model = nn.Sequential(
            nn.Linear(config.get("input_size", 32), 32), nn.ReLU(), nn.Linear(32, 10)
        )
        self.data = data
        self.iter = iter(self.data)
        self.next_sample = next(self.iter)

    def step(self):
        loss = update_model(self.next_sample)
        try:
            self.next_sample = next(self.iter)
        except StopIteration:
            return {"loss": loss, done: True}
        return {"loss": loss}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


class RetinaVAE(nn.Module):
    """Variational Autoencoder class"""

    # TODO BASE CLASS JOHON INTERFACE?

    def __init__(self, apricot_data_folder, gc_type, response_type):
        # def __init__(self):
        super().__init__()

        self.apricot_data_folder = apricot_data_folder
        self.gc_type = gc_type
        self.response_type = response_type

        # Set common VAE model parameters
        self.latent_dim = 4
        self.latent_space_plot_scale = 3.0  # Scale for plotting latent space
        self.lr = 0.0001

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.resolution_hw = (28, 28)

        self.batch_size = 128  # None will take the batch size from test_split size.
        self.epochs = 400
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)
        self.train_by = [["parasol"], ["on", "off"]]  # Train by these factors

        self.this_folder = self._get_this_folder()
        self.models_folder = self._set_models_folder()
        self.ray_dir = self.this_folder / "ray_results"

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": 45.0,  # rotation in degrees
            "translation": (0.1, 0.1),  # fraction of image, (x, y) -directions
            "noise": 0.05,  # noise float in [0, 1] (noise is added to the image)
        }
        self.augmentation_dict = augmentation_dict
        # self.augmentation_dict = None

        self.random_seed = 42
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")

        # # Set the random seed for reproducible results for both torch and numpy
        # torch.manual_seed(self.random_seed)
        # np.random.seed(self.random_seed)

        # # Visualize the augmentation effects and exit
        # self._visualize_augmentation()

        # Create datasets and dataloaders
        self._prep_apricot_data()
        # self._prep_minst_data()

        # Create model and set optimizer and learning rate scheduler
        self._prep_training()

        print(self.vae)
        self.debug = True

        # TÄHÄN JÄIT. TÄMÄHÄN EI SIIS MENE NÄIN. KÄYTÄ
        # https://docs.ray.io/en/master/tune/examples/includes/mnist_pytorch_trainable.html

        tuner = self._set_ray_tuner()
        tuner.fit()
        exit()
        training = True

        if training:
            # Init tensorboard
            self.tb_log_folder = "tb_logs"
            self._prep_tensorboard_logging()

            # Train
            self._train()
            self.writer.flush()
            self.writer.close()

            # Save model
            model_path = self._save_model()
        else:
            # Load previously calculated model for vizualization
            my_model_path = (
                "/opt2/Git_Repos/MacaqueRetina/retina/models/model_20230126-173032.pt"
            )
            # Load model to self.vae and return state dict. The numbers are in the state dict.
            state_dict = self._load_model(model_path=None)

        # Figure 1
        self._plot_ae_outputs(self.vae.encoder, self.vae.decoder, ds_name="test_ds")

        # Figure 1
        self._plot_ae_outputs(self.vae.encoder, self.vae.decoder, ds_name="train_ds")

        # Figure 1
        self._plot_ae_outputs(self.vae.encoder, self.vae.decoder, ds_name="valid_ds")

        self.vae.eval()

        # Figure 2
        self._reconstruct_random_images()

        self._reconstruct_grid_images()

        encoded_samples = self._get_encoded_samples(ds_name="test_ds")

        # Figure 3
        self._plot_latent_space(encoded_samples)

        # Figure 4
        self._plot_tsne_space(encoded_samples)

        encoded_samples = self._get_encoded_samples(ds_name="train_ds")

        # Figure 3
        self._plot_latent_space(encoded_samples)

        # Figure 4
        self._plot_tsne_space(encoded_samples)

    def _set_ray_tuner(self):
        """Set ray tuner"""

        trainable = tune.with_resources(MyTrainableClass, {"cpu": 2, "gpu": 0.25})

        # Search space of the tuning job. Both preprocessor and dataset can be tuned here.
        # Use grid search to try out all values for each parameter. values: iterable
        # Grid search: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#ray.tune.grid_search
        # Sampling: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs
        param_space = {"lr": tune.grid_search([0.001, 0.01, 0.1])}

        # Tuning algorithm specific configs.
        tune_config = None

        # Runtime configuration that is specific to individual trials. Will overwrite the run config passed to the Trainer.
        # for API, see https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.RunConfig
        run_config = (
            air.RunConfig(
                name="my_run",
                stop={"training_iteration": 60},
                local_dir=self.ray_dir,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=True,
                    num_to_keep=1,  # Keep only the best checkpoint
                ),
                verbose=2,
            ),
        )

        tuner = tune.Tuner(
            trainable,
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
        """Save model"""
        model_path = f"{self.models_folder}/model_{self.timestamp}.pt"
        # Create models folder if it does not exist using pathlib

        print(f"Saving model to {model_path}")
        torch.save(self.vae.state_dict(), model_path)
        return model_path

    def _load_model(self, model_path=None):
        """Load model if exists"""

        if model_path is None or not Path(model_path).exists():
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

    def _prep_apricot_data(self):
        """
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

        test_ds.targets = test_ds.labels  # MNIST uses targets instead of labels
        self.test_ds = test_ds

        # Split into train and validation
        train_ds, val_ds = random_split(
            train_val_ds,
            [
                int(np.round(len(train_val_ds) * (1 - self.test_split))),
                int(np.round(len(train_val_ds) * self.test_split)),
            ],
        )

        # Save for later use, this may be a proble with bigger datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.train_ds.targets = train_val_ds.labels[self.train_ds.indices]
        self.val_ds.targets = train_val_ds.labels[self.val_ds.indices]

        # Get n items for the three sets
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)
        self.n_test = len(test_ds)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
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
        test_ds.targets = torch.from_numpy(
            np.fromiter((_test_ds.targets[i] for i in test_indices), int)
        )  # Add targets for the plotting

        self.test_ds = test_ds

        m = len(train_val_ds)

        train_ds, val_ds = random_split(train_val_ds, [int(m - m * 0.2), int(m * 0.2)])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def _prep_training(self):

        self.vae = VariationalAutoencoder(
            latent_dims=self.latent_dim, device=self.device
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
    def _test_epoch(self, vae, device, dataloader):
        # Set evaluation mode for encoder and decoder
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():  # No need to track the gradients
            for x, _ in dataloader:
                # Move tensor to the proper device
                x = x.to(device)
                # Encode data
                encoded_data = vae.encoder(x)
                # Decode data
                x_hat = vae(x)
                loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
                val_loss += loss.item()

        return val_loss / len(dataloader.dataset)

    def _plot_ae_outputs(self, encoder, decoder, ds_name="test_ds"):
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
        targets = ds.targets.numpy()
        t_idx = {i: np.where(targets == i)[0][0] for i in self.train_by_labels}
        encoder.eval()
        decoder.eval()

        n = len(self.train_by_labels)

        for i in range(n):
            t_idx_i = t_idx[self.train_by_labels[i]]
            ax = plt.subplot(2, n, i + 1)
            img = ds[t_idx_i][0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                rec_img = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.text(
                0.05,
                0.85,
                self.apricot_data.data_labels2names_dict[ds[t_idx_i][1].item()],
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.set_title("Original images")

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Reconstructed images")

        # Set the whole figure title as ds_name
        plt.suptitle(ds_name)

    def _train(self):

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(
                self.vae, self.device, self.train_loader, self.optim
            )
            val_loss = self._test_epoch(self.vae, self.device, self.valid_loader)

            # For every 100th epoch, print the outputs of the autoencoder
            if epoch == 0 or epoch % 100 == 0:
                print(
                    f" EPOCH {epoch + 1}/{self.epochs} \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}"
                )

                # Add train loss and val loss to tensorboard SummaryWriter
                self.writer.add_scalars(
                    f"Training_{self.timestamp}",
                    {
                        "loss/train": train_loss,
                        "loss/val": val_loss,
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
