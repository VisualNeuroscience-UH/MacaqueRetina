# Numerical
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd

# Pytorch
# import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F

# import torch.optim as optim

# Viz
from tqdm import tqdm

# Local
from retina.apricot_data_module import ApricotData

# Builtin
from pathlib import Path
import pdb


class VAE(nn.Module):
    """Variational Autoencoder class"""

    # def __init__(self, apricot_data_folder, gc_type, response_type):
    def __init__(self):

        # Set common VAE model parameters
        self.latent_dim = 4
        self.latent_space_plot_scale = 2  # Scale for plotting latent space
        self.lr = 0.001

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.image_shape = (
            28,
            28,
            1,
        )

        self.batch_size = 256  # None will take the batch size from test_split size.
        self.epochs = 50
        self.test_split = 0.2  # Split data for validation and testing (both will take this fraction of data)

        # Preprocessing parameters
        self.gaussian_filter_size = None  # None or 0.5 or ... # Denoising gaussian filter size (in pixels). None does not apply filter
        self.n_pca_components = 16  # None or 32 # Number of PCA components to use for denoising. None does not apply PCA

        # Augment data. Final n samples =  n * (1 + n_repeats * 2): n is the original number of samples, 2 is the rot & shift
        self.n_repeats = 100  # 10  # Each repeated array of images will have only one transformation applied to it (same rot or same shift).
        self.angle_min = -10  # 30 # rotation int in degrees
        self.angle_max = 10  # 30
        self.shift_min = (
            -3
        )  # 5 # shift int in pixels (upsampled space, rand int for both x and y)
        self.shift_max = 3  # 5

        self.random_seed = 42
        # n_threads = 30
        # self._set_n_cpus(n_threads)

        # self._fit_all()
        # self._save_metadata()

        # # Get experimental data
        # self.apricot_data = ApricotData(apricot_data_folder, gc_type, response_type)
        # gc_spatial_data_np = self._get_spatial_apricot_data()

        # # Transform to tensor
        # gc_spatial_data_tensor = torch.from_numpy(gc_spatial_data_np).float()

        # # Split into train and test
        # gc_train, gc_test = random_split(
        #     gc_spatial_data_tensor,
        #     [
        #         int(len(gc_spatial_data_tensor) * (1 - self.test_split)),
        #         int(len(gc_spatial_data_tensor) * self.test_split),
        #     ],
        # )
        # # Create dataloaders

        # # pdb.set_trace()
        # # Create model
        # # Create optimizer
        # # Create loss function
        # # Train
        # # Save model to self.model
        # # Save latent space to self.latent_space
        # # Save reconstruction to self.reconstruction

    # def __call__(self, *args, **kwargs):
    #     return self

    # def __getattr__(self, *args, **kwargs):
    #     return self

    def _get_spatial_apricot_data(self):
        """
        Get spatial ganglion cell data from file using the apricot_data method read_spatial_filter_data()

        Returns
        -------
        gc_spatial_data_np : np.ndarray
            Spatial data with shape (n_gc, 1, ydim, xdim), pytorch format
        """
        (
            gc_spatial_data_np_orig,
            _,
            bad_data_indices,
        ) = self.apricot_data.read_spatial_filter_data()

        # drop bad data
        gc_spatial_data_np = np.delete(
            gc_spatial_data_np_orig, bad_data_indices, axis=2
        )

        # reshape  pytorch (n_samples, 1, xdim, ydim)
        gc_spatial_data_np = np.moveaxis(gc_spatial_data_np, 2, 0)
        gc_spatial_data_np = np.expand_dims(gc_spatial_data_np, axis=1)

        return gc_spatial_data_np


if __name__ == "__main__":

    tmpself = VAE()
    data_dir = "dataset"

    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

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

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m = len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    batch_size = tmpself.batch_size  # 256

    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    class VariationalEncoder(nn.Module):
        def __init__(self, latent_dims):
            super(VariationalEncoder, self).__init__()
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
            self.kl = 0

        def forward(self, x):
            x = x.to(device)
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
        def __init__(self, latent_dims):
            super(VariationalAutoencoder, self).__init__()
            self.encoder = VariationalEncoder(latent_dims)
            self.decoder = Decoder(latent_dims)

        def forward(self, x):
            x = x.to(device)
            z = self.encoder(x)
            return self.decoder(z)

    ### Set the random seed for reproducible results
    torch.manual_seed(tmpself.random_seed)

    vae = VariationalAutoencoder(latent_dims=tmpself.latent_dim)

    optim = torch.optim.Adam(vae.parameters(), lr=tmpself.lr, weight_decay=1e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}")
    vae.to(device)

    ### Training function
    def train_epoch(vae, device, dataloader, optimizer):
        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for x, _ in dataloader:
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

    def test_epoch(vae, device, dataloader):
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

    def plot_ae_outputs(encoder, decoder, n=10):
        plt.figure(figsize=(16, 4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                rec_img = decoder(encoder(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Original images")
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Reconstructed images")
        # plt.show()

    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss = train_epoch(vae, device, train_loader, optim)
        val_loss = test_epoch(vae, device, valid_loader)
        print(
            "\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}".format(
                epoch + 1, num_epochs, train_loss, val_loss
            )
        )
    plot_ae_outputs(vae.encoder, vae.decoder, n=10)

    def show_image(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    vae.eval()

    with torch.no_grad():

        # sample latent vectors from the normal distribution
        latent = torch.randn(128, tmpself.latent_dim, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
        # plt.show()

    encoded_samples = []
    for sample in tqdm(test_dataset):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        vae.eval()
        with torch.no_grad():
            encoded_img = vae.encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {
            f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)
        }
        encoded_sample["label"] = label
        encoded_samples.append(encoded_sample)

    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples

    from sklearn.manifold import TSNE
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    sns.relplot(
        data=encoded_samples,
        x="Enc. Variable 0",
        y="Enc. Variable 1",
        hue=encoded_samples.label.astype(str),
    )

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(["label"], axis=1))

    ax0 = sns.relplot(
        # data=tsne_results,
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=encoded_samples.label.astype(str),
    )
    ax0.set(xlabel="tsne-2d-one", ylabel="tsne-2d-two")
    plt.show()
