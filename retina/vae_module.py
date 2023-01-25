# Numerical
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, fourier_shift
from torch.utils.tensorboard import SummaryWriter


# Pytorch
# import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn
import torch.nn.functional as F

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

    def __init__(self, data, labels, augmentation_dict=None):

        self.data = data
        self.labels = self._to_tensor(labels)

        self.augmentation_dict = augmentation_dict

        # Define transforms
        if self.augmentation_dict is None:
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(self._feature_scaling),
                    transforms.Lambda(self._to_tensor),
                    transforms.Resize((28, 28)),
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
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
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
        noisy : np.ndarray

        """
        noise_factor = self.augmentation_dict["noise"]

        noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
        noisy = np.clip(image + noise, 0.0, 1.0)
        # print(f"noise_factor={noise_factor}")
        return noisy

    def _random_rotate_image(self, image):

        rot = self.augmentation_dict["rotation"]
        # Take random rot as float
        rot = np.random.uniform(-rot, rot)
        # print(f"rot={rot:.2f}")
        image_rot = rotate(image, rot, axes=(2, 1), reshape=False, mode="reflect")
        return image_rot

    def _random_shift_image(self, image):

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


class VAE(nn.Module):
    """Variational Autoencoder class"""

    # TODO KANNATTAAKO TEHDÄ KAKSIVAIHEINEN OPETUS? ENSIN KAIKKI JA SITTEN HALUTTU LUOKKA?

    def __init__(self, apricot_data_folder, gc_type, response_type):
        # def __init__(self):
        super().__init__()

        self.apricot_data_folder = apricot_data_folder
        self.gc_type = gc_type
        self.response_type = response_type

        # Set common VAE model parameters
        self.latent_dim = 32
        self.latent_space_plot_scale = 2  # Scale for plotting latent space
        self.lr = 0.001

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.image_shape = (
            28,
            28,
            1,
        )

        self.batch_size = 128  # None will take the batch size from test_split size.
        self.epochs = 20000
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)

        # # Preprocessing parameters
        # self.gaussian_filter_size = None  # None or 0.5 or ... # Denoising gaussian filter size (in pixels). None does not apply filter
        # self.n_pca_components = 16  # None or 32 # Number of PCA components to use for denoising. None does not apply PCA

        # Augment training and validation data.
        augmentation_dict = {
            "rotation": 20.0,  # rotation in degrees
            "translation": (0.2, 0.2),  # fraction of image, (x, y) -directions
            "noise": 0.15,  # noise float in [0, 1] (noise is added to the image)
        }
        self.augmentation_dict = augmentation_dict
        # self.augmentation_dict = None

        self.random_seed = 42
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")

        # Set the random seed for reproducible results for both torch and numpy
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Create datasets and dataloaders
        self._prep_apricot_data(apricot_data_folder, gc_type, response_type)
        # self._prep_minst_data()

        # Create model and set optimizer and learning rate scheduler
        self._prep_training()

        print(self.vae)

        # Init tensorboard
        self.tb_log_folder = "tb_logs"
        self._prep_tensorboard_logging()

        # Train
        self._train()
        # self.writer.close()

        self._plot_ae_outputs(self.vae.encoder, self.vae.decoder, n=4)

        self.vae.eval()

        self._reconstruct_random_images()

        encoded_samples = self._get_encoded_samples()

        self._plot_latent_space(encoded_samples)

        self._plot_tsne_space(encoded_samples)

    def _prep_apricot_data(self, apricot_data_folder, gc_type, response_type):
        """
        Prep apricot data for training. This includes:
        - Loading data
        - Splitting into training, validation and testing
        - Augmenting data
        - Preprocessing data
        - Creating dataloaders

        Parameters
        ----------
        apricot_data_folder : str
            Path to apricot data folder
        gc_type : str
            Type of ganglion cell to use. Options are 'on' or 'off'
        response_type : str
            Type of response to use. Options are 'mean' or 'peak'

        Returns
        -------

        """

        # Get numpy data
        data_np, labels_np, data_labels = self._get_spatial_apricot_data()

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
            data_np, labels_np, augmentation_dict=self.augmentation_dict
        )

        # Do not augment test data
        test_ds = AugmentedDataset(test_data, test_labels, augmentation_dict=None)

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

        # Get n items for the three sets
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)
        self.n_test = len(test_ds)

        if 0:
            # Plot one example from each set
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            plt.colorbar(axes[0].imshow(train_ds[0][0].squeeze()))
            plt.colorbar(axes[1].imshow(val_ds[0][0].squeeze()))
            plt.colorbar(axes[2].imshow(test_ds[0][0].squeeze()))
            plt.show()

        train_loader = DataLoader(train_ds, batch_size=self.batch_size)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def _get_spatial_apricot_data(self):
        """
        Get spatial ganglion cell data from file using the apricot_data method read_spatial_filter_data().
        All data is returned, the requested data is looged in the class attributes gc_type and response_type.

        Returns
        -------
        gc_spatial_data_np : np.ndarray
            Spatial data with shape (n_gc, 1, ydim, xdim), pytorch format
        """

        self.apricot_data = ApricotData(
            self.apricot_data_folder, self.gc_type, self.response_type
        )

        # Get all available gc types and response types
        gc_types = [
            key[: key.find("_")] for key in self.apricot_data.data_labels.keys()
        ]
        response_types = [
            key[key.find("_") + 1 :] for key in self.apricot_data.data_labels.keys()
        ]
        # Get the integer labels for each gc type and response type
        response_labels = [value for value in self.apricot_data.data_labels.values()]

        # Log requested label
        self.gc_label = self.apricot_data.data_labels[
            f"{self.gc_type}_{self.response_type}"
        ]

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

        # Get all data for learning
        for gc_type, response_type, label in zip(
            gc_types, response_types, response_labels
        ):

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

            # # Covert to tensors
            # collated_gc_spatial_data_t = torch.from_numpy(collated_gc_spatial_data_np)
            # collated_labels_t = torch.from_numpy(collated_labels_np).type(torch.uint8)

        return collated_gc_spatial_data_np, collated_labels_np, apricot_data.data_labels

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

        # Get the folder where this module file is located
        from retina import vae_module as vv

        this_folder = Path(vv.__file__).parent

        # Create a folder for the experiment tensorboard logs
        exp_folder = this_folder.joinpath(self.tb_log_folder)
        Path.mkdir(exp_folder, parents=True, exist_ok=True)

        # Clear files and folders under exp_folder
        for f in exp_folder.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

        # This creates new scalar/time series line in tensorboard
        self.writer = SummaryWriter(str(exp_folder), max_queue=5)

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
            # Print batch loss
            print("\t partial train loss (single batch): %f" % (loss.item()))
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

    def _plot_ae_outputs(self, encoder, decoder, n=10):
        plt.figure(figsize=(16, 4.5))
        targets = self.test_ds.targets.numpy()
        t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
        encoder.eval()
        decoder.eval()
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = self.test_ds[t_idx[i]][0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                rec_img = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Original images")
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Reconstructed images")

    def _train(self):

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(
                self.vae, self.device, self.train_loader, self.optim
            )
            val_loss = self._test_epoch(self.vae, self.device, self.valid_loader)
            print(
                f"\n EPOCH {epoch + 1}/{self.epochs} \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}"
            )
            # # Add train loss and val loss to tensorboard SummaryWriter
            # self.writer.add_scalar("Loss/train", train_loss, epoch)
            # self.writer.add_scalar("Loss/val", val_loss, epoch)
            # Add train loss and val loss to tensorboard SummaryWriter
            with self.writer as writer:
                writer.add_scalars(
                    f"Training_{self.timestamp}",
                    {
                        "loss/train": train_loss,
                        "loss/val": val_loss,
                    },
                    epoch,
                )
            # self.writer.add_scalar("Loss/val", val_loss, epoch)

    def _reconstruct_random_images(self):
        with torch.no_grad():

            # sample latent vectors from the normal distribution
            latent = torch.randn(128, self.latent_dim, device=self.device)

            # reconstruct images from the latent vectors
            img_recon = self.vae.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(20, 8.5))
            # plot 100 images in 10x10 grid with 1 pixel padding in-between.
            self._show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 1))
            ax.set_title("Decoded images from a random sample of latent space")

    def _show_image(self, img):
        npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def _get_encoded_samples(self):
        encoded_samples = []
        for sample in tqdm(self.test_ds):
            img = sample[0].unsqueeze(0).to(self.device)
            label = sample[1]
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
