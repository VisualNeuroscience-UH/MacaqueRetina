# Numerical
import numpy as np
# import scipy.optimize as opt
# import scipy.io as sio
# from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline
import pandas as pd

# Machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Viz
# from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
# from retina.retina_math_module import RetinaMath
from retina.apricot_fitter_module import ApricotData

# Builtin
import sys
import pdb


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, input_shape=None, latent_dim=None, **kwargs):
        super(VAE, self).__init__(**kwargs)

        assert input_shape is not None, 'Argument input_shape  must be specified, aborting...'
        assert latent_dim is not None, 'Argument latent_dim must be specified, aborting...'


        # self.encoder = encoder
        # self.decoder = decoder

        """
        Build encoder
        """
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

        '''
        Build decoder
        '''

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()


        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var, z

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }




class ApricotVAE(ApricotData, VAE):
    """
    Class for creating model for variational autoencoder from  Apricot data 
    """

    # Load temporal and spatial data from Apricot data
    def __init__(self, apricot_data_folder, gc_type, response_type):
        super().__init__(apricot_data_folder, gc_type, response_type)

        self.temporal_filter_data = self.read_temporal_filter_data()
        self.spatial_filter_data = self.read_spatial_filter_data()
        # self.spatial_filter_sums = self.compute_spatial_filter_sums()
        # self.temporal_filter_sums = self.compute_temporal_filter_sums()
        self.bad_data_indices = self.spatial_filter_data[2]


        self._fit_all()

    def plot_latent_space(self, vae, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        
        plt.figure()
        plt.hist(figure.flatten(), bins=30, density=True)

    def plot_comparison_image_samples(self, array1, array2):
        """
        Displays ten random images from each one of the supplied arrays.
        """

        n = 10

        indices = np.random.randint(len(array1), size=n)
        images1 = array1[indices, :]
        images2 = array2[indices, :]

        plt.figure(figsize=(20, 4))
        for i, (image1, image2) in enumerate(zip(images1, images2)):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(image1.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(image2.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    def _prep_data(self, data):
        """
        Prep data for training
        """
        
        # rf_data should have shape N samples, xdim, ydim, 1
        upsampled_data = np.zeros((data.shape[2], 28, 28, 1))
        xx = np.array(range(data.shape[1]))
        yy = np.array(range(data.shape[0]))
        xnew = np.linspace(0, 13, num=28)
        ynew = np.linspace(0, 13, num=28)

        # Temporary upsampling to build vae model
        for this_sample in range(data.shape[2]):
            aa = data[:,:,this_sample] # 2-D array of data with shape (x.size,y.size)
            f = RectBivariateSpline(xx, yy, aa)
            upsampled_data[this_sample, :, :, 0] = f(xnew, ynew)


        # RectBivariateSpline
        # # Divide data into train and test sets
        # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        # (train_data, _), (test_data, _) = keras.datasets.mnist.load_data()
        # rf_data = np.concatenate([train_data, test_data], axis=0)
        # rf_data = np.expand_dims(rf_data, -1).astype("float32") / 25

        rf_data = upsampled_data

        return rf_data

    def _fit_spatial_vae(self):

        gc_spatial_data_array, initial_center_values, _ \
            = self.spatial_filter_data

        # input_shape = gc_spatial_data_array.shape[:2] + (1,)
        input_shape = (28, 28, 1) # tmp for buildup
        latent_dim = 2

        # # Build spatial VAE
        # self.encoder = self.build_encoder(input_shape=input_shape)
        # self.decoder = self.build_decoder()

        # vae = VAE(self.encoder, self.decoder)
        vae = VAE(input_shape=input_shape, latent_dim=latent_dim)
        vae.compile(optimizer=keras.optimizers.Adam())

        rf_data = self._prep_data(gc_spatial_data_array)

        vae.fit(rf_data, epochs=200, batch_size=16)

        return vae, rf_data

    def _fit_all(self):

        spatial_vae, rf_data = self._fit_spatial_vae()

        # Quality of fit
        # self.plot_latent_space(spatial_vae)
        predictions, z_mean, z_log_var, z = spatial_vae.predict(rf_data)
        self.plot_comparison_image_samples(rf_data, predictions)

        plt.show()
        sys.exit()

