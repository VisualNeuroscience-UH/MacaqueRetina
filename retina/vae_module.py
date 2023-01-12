# Numerical
# from cv2 import BFMatcher_create
import numpy as np
from scipy.ndimage import rotate, fourier_shift
from scipy.interpolate import RectBivariateSpline
# from scipy import linalg

# Machine learning
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers
from skimage.filters import gaussian
from sklearn.decomposition import PCA

layers = keras.layers

# Viz
import matplotlib.pyplot as plt

# Local
from retina.apricot_fitter_module import ApricotData
from retina.fid_module import FrechetInceptionDistance

# Builtin
# import sys
import pdb
import os
# import time
import datetime
from pathlib import Path
import shutil
# import logging
import json


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        z_dist = z_mean + tf.exp(0.5 * z_log_var) * epsilon  # tsvae compatible

        return z_dist


class VAE(keras.Model):
    def __init__(
        self,
        image_shape=None,
        latent_dim=None,
        val_data=None,
        batch_normalization=False,
        **kwargs,
    ):
        # super(VAE, self).__init__(**kwargs)
        super().__init__(**kwargs)

        assert (
            image_shape is not None
        ), "Argument image_shape  must be specified, aborting..."
        assert (
            latent_dim is not None
        ), "Argument latent_dim must be specified, aborting..."

        # self.beta = 1.0
        # Init attribute for validation. We lack custom fit() method, so we need to pass validation data to train_step()
        self.mystep = 0
        
        self.val_data = val_data
        self.batch_normalization = batch_normalization

        self.encoder = self._build_encoder(
            latent_dim=latent_dim, image_shape=image_shape
        )

        self.decoder = self._build_decoder(latent_dim)

        self.sampler = Sampler()

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    def _build_encoder(self, image_shape=None, latent_dim=None):
        """
        Build encoder
        """
        assert (
            image_shape is not None
        ), "Argument image_shape  must be specified, aborting..."
        assert (
            latent_dim is not None
        ), "Argument latent_dim must be specified, aborting..."

        encoder_inputs = keras.Input(shape=image_shape)

        if self.batch_normalization is True:
            x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(
                encoder_inputs
            )
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        elif self.batch_normalization is False:
            x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(
                encoder_inputs
            )
            x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
            # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
            #     encoder_inputs
            # )
            # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        encoder.summary()

        return encoder

    def _build_decoder(self, latent_dim=None):
        """
        Build decoder
        """
        assert (
            latent_dim is not None
        ), "Argument latent_dim must be specified, aborting..."

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 32))(x)
        # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)

        if self.batch_normalization is True:
            x = layers.Conv2DTranspose(
                32, 3, strides=2, padding="same", use_bias=False
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2DTranspose(
                16, 3, strides=2, padding="same", use_bias=False
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        elif self.batch_normalization is False:
            x = layers.Conv2DTranspose(
                32, 3, activation="relu", strides=2, padding="same"
            )(x)
            x = layers.Conv2DTranspose(
                16, 3, activation="relu", strides=2, padding="same"
            )(x)
            # x = layers.Conv2DTranspose(
            #     64, 3, activation="relu", strides=2, padding="same"
            # )(x)
            # x = layers.Conv2DTranspose(
            #     32, 3, activation="relu", strides=2, padding="same"
            # )(x)

        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z_random = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z_random)
        return reconstruction, z_mean, z_log_var

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_loss_tracker,
        ]

    # @tf.function
    def train_step(self, data):
        mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
            

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = mse(data, reconstruction)

            # reconstruction_loss = tf.reduce_mean(
            #         tf.reduce_sum(
            #             keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #         )
            # )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + self.beta * tf.reduce_mean(kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        total_loss = self.total_loss_tracker.result()
        reconstruction_loss = self.reconstruction_loss_tracker.result()
        kl_loss = self.kl_loss_tracker.result()

        # log for tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=self.optimizer.iterations)
            tf.summary.scalar(
                "reconstruction_loss",
                reconstruction_loss,
                step=self.optimizer.iterations,
            )
            tf.summary.scalar("kl_loss", kl_loss, step=self.optimizer.iterations)

        if self.val_data is not None:
            val_z_mean, _ = self.encoder(self.val_data)
            val_reconstruction = self.decoder(val_z_mean)
            val_loss = mse(self.val_data, val_reconstruction)
            # val_loss = tf.reduce_mean(
            #         tf.reduce_sum(
            #             keras.losses.binary_crossentropy(self.val_data, val_reconstruction), axis=(1, 2)
            #         )
            # )
            self.val_loss_tracker.update_state(val_loss)
        else:
            val_loss = 0.0

        # Returning validation here evokes tensorboard validation callback
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "val_reconstruction_loss": val_loss,
        }


class TwoStageVAE(keras.Model):
    """
    Constructing code following the two-stage VAE model proposed in ICLR 2019 paper
    Dai, B. and Wipf, D. Diagnosing and enhancing VAE models. Not functional in current status
    Model available at https://github.com/daib13/TwoStageVAE
    """

    def __init__(
        self,
        image_shape=None,
        latent_dim=None,
        val_data=None,
        batch_normalization=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            image_shape is not None
        ), "Argument image_shape  must be specified, aborting..."
        assert (
            latent_dim is not None
        ), "Argument latent_dim must be specified, aborting..."

        # Init attribute for validation. We lack custom fit() method, so we need to pass validation data to train_step()
        self.val_data = val_data
        self.batch_normalization = batch_normalization

        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.second_depth = 3
        self.second_dim = 1024

        self._build_encoder1()
        self._build_decoder1()
        self._build_encoder2()
        self._build_decoder2()
        # self._build_loss()

        self.sampler = Sampler()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        self.total_loss_stage2_tracker = keras.metrics.Mean(name="total_loss_stage2")
        self.reconstruction_loss_stage2_tracker = keras.metrics.Mean(
            name="reconstruction_loss_stage2"
        )
        self.kl_loss_stage2_tracker = keras.metrics.Mean(name="kl_loss_stage2")
        self.val_loss_stage2_tracker = keras.metrics.Mean(name="val_loss_stage2")

    def _build_encoder1(self):
        encoder_inputs = keras.Input(shape=self.image_shape)

        if self.batch_normalization is True:
            x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(
                encoder_inputs
            )
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        elif self.batch_normalization is False:
            x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(
                encoder_inputs
            )
            x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        self.encoder_stage1 = keras.Model(
            inputs=encoder_inputs, outputs=[z_mean, z_log_var], name="encoder_stage1"
        )
        self.encoder_stage1.summary()

    def _build_encoder2(self):
        encoder_inputs = keras.Input(shape=self.latent_dim)
        x = layers.Dense(16, activation="relu", name="fc1")(encoder_inputs)
        x = layers.Dense(16, activation="relu", name="fc2")(x)
        x = layers.Dense(16, activation="relu", name="fc3")(x)
        u_mean = layers.Dense(self.latent_dim, name="u_mean")(x)
        u_log_var = layers.Dense(self.latent_dim, name="u_log_var")(x)
        self.encoder_stage2 = keras.Model(
            inputs=encoder_inputs, outputs=[u_mean, u_log_var], name="encoder_stage2"
        )
        self.encoder_stage2.summary()

    def _build_decoder1(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(
            x
        )
        x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(
            x
        )
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder_stage1 = keras.Model(
            inputs=latent_inputs, outputs=decoder_outputs, name="decoder_stage1"
        )
        self.decoder_stage1.summary()

        self.decoder_stage1.loggamma_x = tf.Variable(
            initial_value=0.0,
            trainable=True,
            name="loggamma_x",
            dtype=tf.float32,
            shape=[],
        )

    def _build_decoder2(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(16, activation="relu", name="fc1")(latent_inputs)
        x = layers.Dense(16, activation="relu", name="fc2")(x)
        x = layers.Dense(16, activation="relu", name="fc3")(x)
        z_hat = layers.Dense(self.latent_dim, name="z_hat")(x)
        self.decoder_stage2 = keras.Model(
            inputs=latent_inputs, outputs=z_hat, name="decoder_stage2"
        )
        self.decoder_stage2.summary()

        self.decoder_stage2.loggamma_z = tf.Variable(
            initial_value=0.0,
            trainable=False,
            name="loggamma_z",
            dtype=tf.float32,
            shape=[],
        )

    def call(self, inputs):
        if self.model_stage == "Stage1":
            z_mean, z_log_var = self.encoder_stage1(inputs)
            z_random = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder_stage1(z_random)
            return reconstruction, z_mean, z_log_var
        elif self.model_stage == "Stage2":
            u_mean, u_log_var = self.encoder_stage2(inputs)
            u_random = self.sampler(u_mean, u_log_var)
            z_hat = self.decoder_stage2(u_random)
            return z_hat, u_mean, u_log_var

    @property
    def metrics(self):
        if self.model_stage == "Stage1":
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.val_loss_tracker,
            ]
        elif self.model_stage == "Stage2":
            return [
                self.total_loss_stage2_tracker,
                self.reconstruction_loss_stage2_tracker,
                self.kl_loss_stage2_tracker,
                self.val_loss_stage2_tracker,
            ]

    # @tf.function
    def train_step(self, data):
        def val_loss_stage1():
            if self.val_data is not None:
                val_z_mean, _ = self.encoder_stage1(self.val_data)
                val_reconstruction = self.decoder_stage1(val_z_mean)
                val_loss = mse(self.val_data, val_reconstruction)
                self.val_loss_tracker.update_state(val_loss)
                return self.val_loss_tracker.result()
            else:
                return 0.0

        def val_loss_stage2():
            # For the second stage, validation data equals the posterior
            # of the image validation data from the first stage.
            if self.val_data is not None:
                val_z_mean, _ = self.encoder_stage1(self.val_data)
                val_u_mean, _ = self.encoder_stage2(val_z_mean)
                val_reconstruction2 = self.decoder_stage2(val_u_mean)
                val_loss2 = mse(val_z_mean, val_reconstruction2)
                self.val_loss_stage2_tracker.update_state(val_loss2)
                return self.val_loss_stage2_tracker.result()
            else:
                return 0.0

        HALF_LOG_TWO_PI = 0.91893  # np.log(2*np.pi)*0.5
        mse = keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )  # mse : loss = square(y_true - y_pred)
        batch_size = tf.cast(tf.shape(data)[0], tf.float32)
        if self.model_stage == "Stage1":
            self.gamma_x = tf.exp(self.decoder_stage1.loggamma_x, name="gamma_x")
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encoder_stage1(data)
                z = self.sampler(z_mean, z_log_var)  # tsvae compatible
                reconstruction = self.decoder_stage1(z)

                reconstruction_loss = (
                    tf.reduce_sum(
                        tf.square((data - reconstruction) / self.gamma_x) / 2.0
                        + self.decoder_stage1.loggamma_x
                        + HALF_LOG_TWO_PI
                    )
                    / batch_size
                )

                kl_loss = (
                    -0.5
                    * tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    / batch_size
                )  # tsvae compatible
                total_loss = reconstruction_loss + self.beta * kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            total_loss = self.total_loss_tracker.result()
            reconstruction_loss = self.reconstruction_loss_tracker.result()
            kl_loss = self.kl_loss_tracker.result()

            # log for tensorboard
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "total_loss", total_loss, step=self.optimizer.iterations
                )
                tf.summary.scalar(
                    "reconstruction_loss",
                    reconstruction_loss,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar("kl_loss", kl_loss, step=self.optimizer.iterations)
                tf.summary.scalar(
                    "loggamma_x",
                    self.decoder_stage1.loggamma_x,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar(
                    "gamma_x", self.gamma_x, step=self.optimizer.iterations
                )
                tf.summary.histogram("z_mean", z_mean, step=self.optimizer.iterations)
                tf.summary.histogram(
                    "z_log_var", z_log_var, step=self.optimizer.iterations
                )
                tf.summary.histogram("z", z, step=self.optimizer.iterations)

            return {
                "total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "val_reconstruction_loss": val_loss_stage1(),
            }

        elif self.model_stage == "Stage2":

            self.gamma_z = tf.exp(self.decoder_stage2.loggamma_z, name="gamma_z")

            with tf.GradientTape() as tape:
                u_mean, u_log_var = self.encoder_stage2(data)
                u = self.sampler(u_mean, u_log_var)
                reconstruction = self.decoder_stage2(u)
                reconstruction_loss_stage2 = (
                    tf.reduce_sum(
                        (
                            tf.square((data - reconstruction) / self.gamma_z) / 2.0
                            + self.decoder_stage2.loggamma_z
                            + HALF_LOG_TWO_PI
                        )
                    )
                    / batch_size
                )

                # tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
                kl_loss_stage2 = (
                    -0.5
                    * tf.reduce_sum(
                        1 + u_log_var - tf.square(u_mean) - tf.exp(u_log_var)
                    )
                    / batch_size
                )
                total_loss_stage2 = (
                    reconstruction_loss_stage2 + self.beta * kl_loss_stage2
                )

            grads = tape.gradient(total_loss_stage2, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_stage2_tracker.update_state(total_loss_stage2)
            self.reconstruction_loss_stage2_tracker.update_state(
                reconstruction_loss_stage2
            )
            self.kl_loss_stage2_tracker.update_state(kl_loss_stage2)
            total_loss_stage2 = self.total_loss_stage2_tracker.result()
            reconstruction_loss_stage2 = (
                self.reconstruction_loss_stage2_tracker.result()
            )
            kl_loss_stage2 = self.kl_loss_stage2_tracker.result()

            # log for tensorboard
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "total_loss_stage2",
                    total_loss_stage2,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar(
                    "reconstruction_loss_stage2",
                    reconstruction_loss_stage2,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar(
                    "kl_loss_stage2",
                    kl_loss_stage2,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar(
                    "loggamma_z",
                    self.decoder_stage2.loggamma_z,
                    step=self.optimizer.iterations,
                )
                tf.summary.scalar(
                    "gamma_z", self.gamma_z, step=self.optimizer.iterations
                )
                tf.summary.histogram("u_mean", u_mean, step=self.optimizer.iterations)
                tf.summary.histogram(
                    "u_log_var", u_log_var, step=self.optimizer.iterations
                )
                tf.summary.histogram("u", u, step=self.optimizer.iterations)

            return {
                "total_loss_stage2": total_loss_stage2,
                "reconstruction_loss_stage2": reconstruction_loss_stage2,
                "kl_loss_stage2": kl_loss_stage2,
                # "val_reconstruction_loss": val_loss_stage1(),
                "val_reconstruction_loss_stage2": val_loss_stage2(),
                # "val_reconstruction_loss_stage2": val_loss_stage1(),
            }


class ApricotVAE(ApricotData):
    """
    Class for creating model for variational autoencoder from  Apricot data
    """

    # Load temporal and spatial data from Apricot data
    def __init__(self, apricot_data_folder, gc_type, response_type):
        super().__init__(apricot_data_folder, gc_type, response_type)

        # self.temporal_filter_data = self.read_temporal_filter_data()
        # self.spatial_filter_data = self.read_spatial_filter_data()
        # self.spatial_filter_sums = self.compute_spatial_filter_sums()
        # self.temporal_filter_sums = self.compute_temporal_filter_sums()
        # self.bad_data_indices = self.spatial_filter_data[2]

        # Set common VAE model parameters
        self.latent_dim = 4  # 2 32
        self.latent_space_plot_scale = 2  # Scale for plotting latent space
        self.beta = 1  # Beta parameter for KL loss. Overrides VAE class beta parameter

        self.model_type = "VAE"  # TwoStageVAE or VAE
        self.batch_normalization = False
        self.lr_epochs = 5 # 150 at these epoch intervals, learning rate will be divided by half. Applies to TwoStageVae only
        self.lr = 0.001 # 0.0001
        self.optimizer_stage1 = keras.optimizers.Adam(
            learning_rate=self.lr
        )  # default lr = 0.001
        self.optimizer_stage2 = keras.optimizers.Adam(
            learning_rate=self.lr
        )  # Only used for TwoStageVAE

        # lr rate change from tf1
        # lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(float(epoch) / float(args.lr_epochs)))

        # Images will be sampled to this space. If you change this you need to change layers, too, for consistent output shape
        self.image_shape = (28, 28, 1,)  
        # self.image_shape = (299, 299, 1) 
        self.batch_size = 128 # 512  # None will take the batch size from test_split size. Note that the batch size affects training speed and loss values
        self.batch_size_stage2 = 512 # 512  # Only used for TwoStageVAE

        # TÄHÄN JÄIT:
        # Yksinkertaista mallia 64=> 32
        # TSEKKAA KOODI, TSEKKAA TB PARAMETRIT RISTIIN TF1 VERSION KANSSA
        # KATSO SAATKO YLEISTETTYÄ AIKAAN

        self.epochs = 300
        self.epochs_stage2 = 0 # Only used for TwoStageVAE
        self.test_split = 0.2  # None or 0.2  # Split data for validation and testing (both will take this fraction of data)
        self.verbose = 2  #  1 or 'auto' necessary for graph creation. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        # Preprocessing parameters
        self.gaussian_filter_size = None  # None or 0.5 or ... # Denoising gaussian filter size (in pixels). None does not apply filter
        self.n_pca_components = 16  # None or 32 # Number of PCA components to use for denoising. None does not apply PCA

        # Augment data. Final n samples =  n * (1 + n_repeats * 2): n is the original number of samples, 2 is the rot & shift
        self.n_repeats = 100 # 10  # Each repeated array of images will have only one transformation applied to it (same rot or same shift).
        self.angle_min = -10  # 30 # rotation int in degrees
        self.angle_max = 10  # 30
        self.shift_min = (
            -3
        )  # 5 # shift int in pixels (upsampled space, rand int for both x and y)
        self.shift_max = 3  # 5

        self.random_seed = 42
        tf.keras.utils.set_random_seed(self.random_seed)

        n_threads = 30
        self._set_n_cpus(n_threads)

        # Eager mode works like normal python code. You can access variables better.
        # Graph mode postpones computations, but is more efficient.
        # Graph mode also forces tensorflow to produce graph for tensorboard
        tf.config.run_functions_eagerly(False)

        self.output_path = Path("./retina/output")  # move  later to io
        self.exp_folder = Path("vae")
        self.metadata_folder = Path("/home/simo/Documents/Analysis/")

        self.tensorboard_callback = []
        self._prep_tensorboard_logging()  # sets tensorboard_callback
        self._fit_all()

        self._save_metadata()

    def _save_metadata(self):
        """
        From the self object save all string, scalar or None attributes to
        a text file at the metadata_folder.
        """
        metadata = {}
        for attr in dir(self):
            if not attr.startswith("_"):
                try:
                    value = getattr(self, attr)
                except:
                    pass
                if isinstance(value, (str, int, float, type(None))):
                    metadata[attr] = value
        # Append short time stamp to metadata file name
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = f"metadata_{time_stamp}.txt"
        metadata_path = self.metadata_folder / self.exp_folder / metadata_file
        # Create folder if it does not exist
        if not metadata_path.parent.exists():
            metadata_path.parent.mkdir(parents=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _prep_tensorboard_logging(self):
        """
        Prepare local folder environment for tensorboard logging and model building

        Note that the tensoboard reset takes place only when quitting the terminal call
        to tensorboard. You will see old graph, old scalars, if they are not overwritten
        by new ones.

        Scalar logger x-axis is (Ndata/batch_size) * epochs
        """

        # Folders
        exp_folder = Path(self.output_path).joinpath(self.exp_folder)
        Path.mkdir(exp_folder, parents=True, exist_ok=True)

        # Clear files and folders under exp_folder
        for f in exp_folder.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

        self.tensorboard_callback.append(tf.keras.callbacks.TensorBoard(
            log_dir=exp_folder,
            histogram_freq=1,
            write_graph=True,
        ))

        # This creates new scalar/time series line in tensorboard
        self.summary_writer = tf.summary.create_file_writer(str(exp_folder))

    def _prep_lr_scheduler(self):
        """
        Prepare learning rate scheduler for the 2-stage vae.
        """
        def lr_scheduler(epoch, lr):
            # Learning rate scheduler for the 2-stage vae.
            # This cuts learning rate by half after lr_epochs number of epochs is reached
            # This is reflected onto loggamma
            lr_updated = (
                self.lr
                if self.lr_epochs is None
                else self.lr * np.power(0.5, np.floor(float(epoch) / float(self.lr_epochs)))
            )

            return lr_updated
        
        self.tensorboard_callback.append(tf.keras.callbacks.LearningRateScheduler(
            lr_scheduler, verbose=0)
        )

    def _set_n_cpus(self, n_threads):
        # Set number of CPU cores to use for parallel processing
        # NOTE Not sure this has any effect
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(n_threads)

        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        tf.config.set_soft_device_placement(True)

    def plot_latent_space(self, vae, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = self.image_shape[0]  # side length of the digits
        scale = self.latent_space_plot_scale
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        if self.model_type == "VAE":
            predictor = vae.decoder.predict
        elif self.model_type == "TwoStageVAE":
            predictor = vae.decoder_stage1.predict

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                if self.latent_dim > 2:
                    # Add the extra dimensions as zeros to the z_sample
                    z_sample = np.concatenate(
                        (z_sample, np.zeros((1, self.latent_dim - 2))), axis=1
                    )
                    # z_sample = np.concatenate((z_sample, np.ones((1, self.latent_dim - 2))), axis=1)
                # print(f"z_sample: {z_sample}")
                x_decoded = predictor(z_sample)
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
        # plt.imshow(figure)

        plt.figure()
        plt.hist(figure.flatten(), bins=30, density=True)
        # set title as "latent space"
        plt.title("latent space")

    def plot_label_clusters(self, vae, data, labels=None):
        '''Display a 2D plot of the digit classes in the latent space'''

        # z_mean, _, _ = vae.encoder.predict(data)
        z_mean, _ = vae.encoder.predict(data)
        plt.figure(figsize=(12, 10))
        if labels is not None:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap="tab10")
        else:
            plt.scatter(z_mean[:, 0], z_mean[:, 1])
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

    def plot_sample_images(self, image_array, labels=None, n=10, sample=None):
        """
        Displays n random images from each one of the supplied arrays.
        """

        if sample is not None:
            n = len(sample)

        if isinstance(image_array, list):
            nrows = len(image_array)
            # Assert that all supplied arrays are of type numpy.ndarray and are of the same length
            for image_array_i in image_array:
                assert isinstance(
                    image_array_i, np.ndarray
                ), "Supplied array is not of type numpy.ndarray, aborting..."
        else:
            nrows = 1
            image_array = [image_array]

        if not len(image_array_i) == len(image_array[0]):
            print(
                """Supplied arrays are of different length. 
                        If no sample list is provided, 
                        indices will be drawn randomly from the first array."""
            )

        if sample is None:
            indices = np.random.randint(len(image_array[0]), size=n)
        else:
            indices = sample

        image1_shape = image_array[0][0].squeeze().shape
        images1 = image_array[0][indices, :]

        images1_min = np.min(images1.flatten())
        images1_max = np.max(images1.flatten())

        plt.figure(figsize=(2 * n, 2 * nrows))

        for i, image1 in enumerate(images1):
            ax = plt.subplot(nrows, n, i + 1)
            plt.imshow(image1.reshape(image1_shape), vmin=images1_min, vmax=images1_max)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if labels is not None and i == 0:
                plt.title(labels[0])
        plt.colorbar()

        if nrows > 1:
            for i, image_array_i in enumerate(image_array[1:]):
                images2 = image_array_i[indices, :]
                images2_min = np.min(images2.flatten())
                images2_max = np.max(images2.flatten())
                image2_shape = image_array_i[0].squeeze().shape

                for j, image2 in enumerate(images2):
                    ax = plt.subplot(nrows, n, i * n + j + 1 + n)
                    plt.imshow(
                        image2.reshape(image2_shape), vmin=images2_min, vmax=images2_max
                    )
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if labels is not None and j == 0:
                        plt.title(labels[i + 1])
                plt.colorbar()

    def _resample_data(self, data, resampled_size):
        """
        Up- or downsample data
        """

        assert (
            len(resampled_size) == 2
        ), "Resampled_size must be a 2-element array, aborting..."

        n_samples = data.shape[0]

        x_orig_dim = data.shape[1]
        y_orig_dim = data.shape[2]
        x_orig_grid = np.array(range(x_orig_dim))
        y_orig_grid = np.array(range(y_orig_dim))

        # rf_data should have shape N samples, xdim, ydim, 1
        x_new_dim = resampled_size[0]
        y_new_dim = resampled_size[1]
        upsampled_data = np.zeros((n_samples, x_new_dim, y_new_dim, 1))

        x_new_grid = np.linspace(0, x_orig_dim, num=x_new_dim)
        y_new_grid = np.linspace(0, y_orig_dim, num=y_new_dim)

        # Temporary upsampling to build vae model
        for this_sample in range(n_samples):
            aa = data[this_sample, :, :]  # 2-D array of data with shape (x.size,y.size)
            f = RectBivariateSpline(x_orig_grid, y_orig_grid, aa)
            upsampled_data[this_sample, :, :, 0] = f(x_new_grid, y_new_grid)

        return upsampled_data

    def _get_spatial_vae_model(self, val_data):
        """
        Builds a model for the spatial VAE
        """

        # Build model
        if self.model_type == "VAE":
            vae = VAE(
                image_shape=self.image_shape,
                latent_dim=self.latent_dim,
                val_data=val_data,
                batch_normalization=self.batch_normalization,
            )
        elif self.model_type == "TwoStageVAE":
            vae = TwoStageVAE(
                image_shape=self.image_shape,
                latent_dim=self.latent_dim,
                val_data=val_data,
                batch_normalization=self.batch_normalization,
            )
        else:
            raise ValueError("Model type not recognized, aborting...")

        # change beta
        vae.beta = self.beta

        return vae

    def _fit_spatial_vae(self, data, val_data):

        vae = self._get_spatial_vae_model(val_data=val_data)

        # Attach summary file writer to model object
        vae.summary_writer = self.summary_writer

        if self.model_type == "TwoStageVAE":
            vae.model_stage = "Stage1"
            # Set encoder_stage2 and decoder_stage2 weights to non-trainable status
            vae.encoder_stage2.trainable = False
            vae.decoder_stage2.trainable = False
            self._prep_lr_scheduler() # appends lr_scheduler to fit callbacks

        # Compile model
        vae.compile(optimizer=self.optimizer_stage1)

        # Fit model
        fit_history = vae.fit(
            data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            callbacks=self.tensorboard_callback,
        )

        # Extract posterior. For tf2, model.predict -method handles the batch size.
        # Thus no separate loop is needed as in tf1 version in method extract_posterior
        if self.model_type == "TwoStageVAE":
            vae.model_stage = "Stage2"
            z_mean, z_log_var = vae.encoder_stage1.predict(
                data, batch_size=self.batch_size
            )

            # We take a normally distributed sample from the latent space
            # Note that SD = np.exp(0.5 * np.log(VAR))
            z_N01_sample = Sampler()(z_mean, z_log_var)

            # Switch trainable weights to stage2
            vae.encoder_stage1.trainable = False
            vae.decoder_stage1.trainable = False
            vae.encoder_stage2.trainable = True
            vae.decoder_stage2.trainable = True

            # Compile model
            vae.compile(optimizer=self.optimizer_stage2)

            # Fit model, second stage. Now the data
            # comprises the normally distributed sample from the
            # latent space of the first stage
            fit_history_stage2 = vae.fit(
                z_N01_sample,
                epochs=self.epochs_stage2,
                batch_size=self.batch_size_stage2,
                shuffle=True,
                verbose=self.verbose,
                callbacks=self.tensorboard_callback,
            )

            # Set model stage back to 1 for evaluation
            vae.model_stage = "Stage1"

            return vae, fit_history, fit_history_stage2

        return vae, fit_history

    def _get_spatial_apricot_data(self):
        """
        Get spatial data from file using the inherited method read_spatial_filter_data()
        """
        gc_spatial_data_np_orig, _, bad_data_indices = self.read_spatial_filter_data()

        # drop bad data
        gc_spatial_data_np = np.delete(
            gc_spatial_data_np_orig, bad_data_indices, axis=2
        )

        # reshape to (n_samples, xdim, ydim, 1)
        gc_spatial_data_np = np.moveaxis(gc_spatial_data_np, 2, 0)
        gc_spatial_data_np = np.expand_dims(gc_spatial_data_np, axis=3)

        return gc_spatial_data_np

    def _plot_batch_sizes(self, ds):

        plt.figure()
        batch_sizes = [batch.shape[0] for batch in ds]
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.xlabel("Batch number")
        plt.ylabel("Batch size")

    def _augment_data(self, np_array, n_repeats):
        """
        Rotate and repeat datasets
        """

        if n_repeats <= 1 or n_repeats is None:
            return np_array

        def _random_rotate_image(image, rot_angle):

            assert (
                image.ndim == 4
            ), "Image array must be 4-dimensional for rotation axes to match, aborting..."

            image_rot = rotate(
                image, rot_angle, axes=(2, 1), reshape=False, mode="reflect"
            )
            return image_rot

        def _random_shift_image(image, shift):

            assert (
                image.ndim == 4
            ), "Image array must be 4-dimensional for shift axes to match, aborting..."
            image_shifted = image.copy()
            for this_image_idx in range(image.shape[0]):
                this_image = image[this_image_idx, :, :, 0]
                input_ = np.fft.fft2(this_image)
                result = fourier_shift(
                    input_, shift=shift
                )  # shift in pixels, tuple of (y, x) shift
                # result = fourier_shift(input_, shift=(-5, 0)) # shift in pixels, tuple of (y, x) shift
                result = np.fft.ifft2(result)
                image_shifted[this_image_idx, :, :, 0] = result.real

            return image_shifted

        def _rotate_and_shift_image_QA(image, image_transf, this_ima=0, this_transf=0):

            n_ima = image.shape[0]

            plt.figure()
            plt.imshow(image[this_ima, :, :, 0])
            plt.title("Original image")
            plt.figure()
            plt.imshow(image_transf[this_ima + n_ima * this_transf, :, :, 0])
            plt.title("Transformed image")
            plt.show()

        n_orig_images = np_array.shape[0]

        # Augment with rotations
        # n_repeats which will be rotated
        np_array_rep = np.repeat(np_array, n_repeats, axis=0)
        rot = np.random.uniform(self.angle_min, self.angle_max, size=n_repeats)

        for rot_idx, this_rot in enumerate(rot):
            np_array_rep[
                rot_idx * n_orig_images : (rot_idx + 1) * n_orig_images, :, :
            ] = _random_rotate_image(np_array, this_rot)
            # rot_all = rot_all + [this_rot] * n_orig_images # append this_rot to list rot_all n_orig_images times

        np_array_augmented = np.vstack((np_array, np_array_rep))

        # Augment with shifts
        # n_repeats which will be shifted
        np_array_rep = np.repeat(np_array, n_repeats, axis=0)
        shift = np.random.uniform(self.shift_min, self.shift_max, size=(n_repeats, 2))

        for shift_idx, this_shift in enumerate(shift):
            np_array_rep[
                shift_idx * n_orig_images : (shift_idx + 1) * n_orig_images, :, :
            ] = _random_shift_image(np_array, this_shift)

        if 0:
            _rotate_and_shift_image_QA(
                np_array, np_array_rep, this_ima=0, this_transf=2
            )

        np_array_augmented = np.vstack((np_array_augmented, np_array_rep))

        return np_array_augmented

    def _to_numpy_array(self, ds):
        """
        Convert tf.data.Dataset to numpy array e.g. for easier visualization and analysis
        This function is on hold, because I am working directly with numpy arrays at the moment (small datasets)
        """

        if isinstance(ds, tf.data.Dataset):
            ds_np = np.stack(list(ds))
            ds_np = np.squeeze(ds_np)
            dims = ds_np.shape
            ds_np = ds_np.reshape(dims[0] * dims[1], dims[2], dims[3])

        else:
            row_length = len(ds)
            dims = [row_length] + ds.element_spec.shape.as_list()
            ds_np = np.zeros(dims)

            for idx, element in enumerate(ds):
                ds_np[idx, :] = element.numpy()

        return ds_np

    def _normalize_data(
        self, data, scale_type="standard", scale_min=None, scale_max=None
    ):
        """
        Normalize data to either mean 0, std 1 or [scale_min, scale_max]

        :param data: numpy array
        :param type: 'standard' or 'minmax'
        :param scale_min: minimum value of scaled data
        :param scale_max: maximum value of scaled data
        """

        if scale_type == "standard":
            data_mean = data.mean()
            data_std = data.std()
            data = (data - data_mean) / data_std

            # Get data transfer parameters
            self.data_mean = data_mean
            self.data_std = data_std
        elif scale_type == "minmax":
            if scale_min is None:
                scale_min = data.min()
            if scale_max is None:
                scale_max = data.max()
            data = (data - scale_min) / (scale_max - scale_min)

            # Get data transfer parameters
            self.data_min = scale_min
            self.data_max = scale_max
        else:
            raise ValueError('scale_type must be either "standard" or "minmax"')

        return data.astype("float32")

    def _filter_data_gaussian(self, data):
        """
        Filter data with Gaussian filter
        :param data: numpy array
        """
        filter_size_pixels = self.gaussian_filter_size

        # If filter_size_pixels = None, do not filter
        if filter_size_pixels is None:
            print("No filtering with Gaussian filter")
            return data

        dataf = np.zeros(data.shape)
        for idx in range(data.shape[0]):

            # Apply gaussian filter
            dataf[idx, :, :, 0] = gaussian(
                data[idx, :, :, 0],
                sigma=[filter_size_pixels, filter_size_pixels],
                channel_axis=-1,
                mode="reflect",
                truncate=2.0,
            )

        return dataf.astype("float32")

    def _filter_data_PCA(self, data):
        """
        Filter data with PCA
        :param data: numpy array
        """

        n_pca_components = self.n_pca_components

        if n_pca_components is None:
            print("No filtering with PCA")
            return data, None

        # PCA without Gaussian filter
        # Flatten images to provide 2D input for PCA. Each row is an image, each column a pixel
        data_2D_np = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

        # Apply PCA to data
        pca = PCA(n_components=n_pca_components)
        pca.fit(data_2D_np)

        # Get inverse PCA transformation
        pca_inv = pca.inverse_transform(pca.transform(data_2D_np))

        # Reshape to 3D
        data_pca = pca_inv.reshape(data.shape[0], data.shape[1], data.shape[2])

        return data_pca.astype("float32"), pca

    def _show_gaussian_filtered_data(self, data, dataf, example_image=[0]):
        """
        Show original and filtered data
        """

        if self.gaussian_filter_size is None:
            return

        if isinstance(example_image, int):
            example_image = [example_image]

        for example_image_idx in example_image:

            plt.figure()
            # Create one subplt for each plot below

            # Show exaple original and filtered image
            plt.subplot(1, 2, 1)
            plt.imshow(data[example_image_idx, :, :])
            plt.colorbar()
            plt.title("Original image")

            plt.subplot(1, 2, 2)
            plt.imshow(dataf[example_image_idx, :, :])
            plt.colorbar()
            plt.title("Gaussian filtered image")

    def _show_pca_components(self, data, data_pca, pca, example_image=[0]):
        """
        Show PCA components
        """
        if pca is None:
            return

        if isinstance(example_image, int):
            example_image = [example_image]

        for example_image_idx in example_image:
            plt.figure()
            # Create one subplt for each plot below

            # Show PCA components
            plt.subplot(2, 2, 1)
            plt.bar(
                range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_
            )
            plt.title("PCA explained variance ratio")

            # Plot first 2 PCA components as scatter plot
            plt.subplot(2, 2, 2)
            plt.scatter(pca.components_[0, :], pca.components_[1, :])
            plt.title("PCA components 1 and 2")

            # From the previous subplot, color the example_image red
            plt.scatter(
                pca.components_[0, example_image_idx],
                pca.components_[1, example_image_idx],
                c="r",
            )

            # Show exaple original and filtered image
            plt.subplot(2, 2, 3)
            plt.imshow(data[example_image_idx, :, :])
            plt.colorbar()
            plt.title("Original image")

            plt.subplot(2, 2, 4)
            plt.imshow(data_pca[example_image_idx, :, :])
            plt.colorbar()
            plt.title("PCA reconstructed image")

    def _input_processing_pipe(self, data_np):
        """
        Process input data for training and validation
        """
        # Filter data
        data_npf = self._filter_data_gaussian(data_np)
        # self._show_gaussian_filtered_data(data_np, data_npf, example_image=[0, 69, 87])

        data_npf, pca = self._filter_data_PCA(data_npf)
        # self._show_pca_components(data_np, data_npf, pca, example_image=[0, 69, 87])
        # plt.show()

        # Up or downsample 2D image data
        data_us = self._resample_data(data_npf, self.image_shape[:2])

        # Normalize data
        data_usn = self._normalize_data(
            data_us, scale_type="minmax", scale_min=None, scale_max=None
        )

        if self.test_split is not None:
            # split data into train, validation and test sets using proportion of 60%, 20%, 20%
            skip_size = int(data_us.shape[0] * self.test_split)
            data_train_np = data_usn[skip_size * 2 :]
            data_val_np = data_usn[:skip_size]
            data_test_np = data_usn[skip_size : skip_size * 2]
            # Report split sizes
            print(
                f"The train sample size before augmentation is {data_train_np.shape[0]}"
            )
            print(f"The validation sample size is {data_val_np.shape[0]}")
            print(f"The test sample size is {data_test_np.shape[0]}")
            # Augment data with random rotation and shift
            data_train_np_reps = self._augment_data(data_train_np, self.n_repeats)

            # Report split sizes
            print(
                f"The train sample size after augmentation is {data_train_np_reps.shape[0]}"
            )

            # Set batch size as the number of samples in the smallest dataset
            if self.batch_size == None:
                self.batch_size = min(
                    data_train_np_reps.shape[0],
                    data_val_np.shape[0],
                    data_test_np.shape[0],
                )
                assert (
                    data_val_np.shape[0] != 0
                ), "Validation set is empty, you must determine batch_size manually, aborting..."
                print(f"Batch size is {self.batch_size}")
            elif self.batch_size > min(
                data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0]
            ):
                self.batch_size = min(
                    data_train_np_reps.shape[0],
                    data_val_np.shape[0],
                    data_test_np.shape[0],
                )
                print(
                    f"Batch size is set to {self.batch_size} to match the smallest dataset"
                )
            elif self.batch_size < min(
                data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0]
            ):
                print(
                    f"Validation and test data size is set to {self.batch_size} to match the batch size"
                )
                data_val_np = data_val_np[: self.batch_size]
                data_test_np = data_test_np[: self.batch_size]
            print("\n")
            data_train_np = data_train_np_reps

        else:
            data_train_np = data_usn
            data_val_np = None
            data_test_np = None

        return data_train_np, data_val_np, data_test_np

    def _plot_fit_history(self, fit_history):
        """
        Plot the fit history
        """

        plt.figure()
        keys = [k for k in fit_history.history.keys()]
        for key in keys:
            plt.plot(fit_history.history[key])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(keys, loc="upper left")

        [v_l_key] = [v for v in keys if "val_reconstruction_loss" in v]
        val_min = min(fit_history.history[v_l_key])
        # plot horizontal dashed red line at the minimum of val_reconstruction_loss
        plt.axhline(val_min, color="r", linestyle="dashed", linewidth=1)

        # get the max value of x-axis
        x_max = plt.xlim()[1]
        # add the minimum value as text
        plt.text(x_max, val_min, round(val_min, 1), ha="right", va="bottom", color="r")

    def _plot_z_mean_in_2D(self, z_mean):
        plt.figure()
        plt.scatter(z_mean[:, 0], z_mean[:, 1])
        plt.title("Latent space")

    def _get_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

        mnist_targets = np.concatenate([y_train, y_test], axis=0)

        return mnist_digits, mnist_targets

    def _k_fold_cross_validation(self, data_np, n_folds=3):
        """
        Perform k-fold cross validation on the data
        """
        # loss for validation reconstruction
        mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        num_validation_samples = len(data_np) // n_folds
        np.random.shuffle(data_np)

        validation_scores = []
        reconstruction_losses = []

        for fold in range(n_folds):

            validation_data = data_np[
                num_validation_samples * fold : num_validation_samples * (fold + 1)
            ]
            training_data = np.vstack(
                (
                    data_np[: num_validation_samples * fold],
                    data_np[num_validation_samples * (fold + 1) :],
                )
            )

            model = self._get_spatial_vae_model()
            model.fit(
                training_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=self.verbose,
            )

            # Evaluate model on validation data
            (this_val_rf_reconstructed, _, _) = model.predict(validation_data)
            loss = mse(validation_data, this_val_rf_reconstructed)
            n_val = validation_data.shape[0]
            reconstruction_losses.append(loss / n_val)

            validation_scores.append(np.average(np.array(reconstruction_losses)))

        validation_score = np.average(validation_scores)
        validation_score_std = np.std(validation_scores)

        # Final model fit with all training data
        model = self._get_spatial_vae_model()
        fit_history = model.fit(
            data_np,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
        )

        return model, (validation_score, validation_score_std), fit_history

    def _get_fid_score(self, model, fid_data, image_range):
        """
        Calculate the FID score for the model

        """
        # Construct inception model
        fid = FrechetInceptionDistance(
            image_range=image_range, generator_postprocessing=None
        )

        # Get the predicted image given the data as input
        generated_val, _, _ = model.predict(fid_data)

        # Up or downsample 2D image data
        fid_data_us = self._resample_data(fid_data, (75, 75))
        generated_val_us = self._resample_data(generated_val, (75, 75))

        # Augment data by multiplying the last dimension to 3
        fid_data_3 = np.repeat(fid_data_us, 3, axis=-1)
        generated_val_3 = np.repeat(generated_val_us, 3, axis=-1)

        # 	Arguments to call, see FrechetInceptionDistance class in fid_module for more details
        n_samples = fid_data_3.shape[0]
        fid_score = fid(
            fid_data_3,
            generated_val_3,
            batch_size=self.batch_size,
            num_batches_real=int(np.floor(n_samples / self.batch_size)),
            num_batches_gen=None,
            shuffle=True,
            seed=self.random_seed,
        )

        return fid_score

    def _get_ssim_score(self, model, ssim_data, image_range):
        """
        Calculate the structural similarity score (SSIM) for the model.
        The SSIM is a function of differences between luminance, contrast, and structure.
        The structure is a measure of the local correlation between pixels. 
        
        This function is based on the standard SSIM implementation from: 
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004) 
        Image quality assessment: from error visibility to structural similarity. 
        IEEE transactions on image processing. 
        """
        # Get the predicted image given the data as input
        generated_val, _, _ = model.predict(ssim_data)

        # Up or downsample 2D image data
        ssim_data_us = self._resample_data(ssim_data, (75, 75))
        generated_val_us = self._resample_data(generated_val, (75, 75))

        # Calculate SSIM
        ssim_score = tf.image.ssim(
            ssim_data_us, generated_val_us, max_val=np.ptp(image_range)
        )

        return ssim_score.numpy().mean()

    def _generate_from_latent_space(self, model, n_samples=10):

        if self.model_type == "VAE":
            z_random = np.random.normal(0, 1, [n_samples, self.latent_dim])
            reconstruction = model.decoder(z_random)
        elif self.model_type == "TwoStageVAE":
            model.model_stage = "Stage2"
            # u ~ N(0, I)
            u_random = np.random.normal(0, 1, [n_samples, self.latent_dim])
            # z ~ N(f_2(u), \gamma_z I)
            z_hat = model.decoder_stage2.predict(u_random)
            gamma_z = tf.exp(model.decoder_stage2.loggamma_z, name="gamma_z")

            # z, gamma_z = sess.run([self.z_hat, self.gamma_z], feed_dict={self.u: u, self.is_training: False})
            z_random = z_hat + gamma_z * np.random.normal(
                0, 1, [n_samples, self.latent_dim]
            )
            reconstruction = model.decoder_stage1(z_random)

        return reconstruction

    def _fit_all(self):

        # Get numpy array of data in correct dimensions
        gc_spatial_data_np = self._get_spatial_apricot_data()

        # # Process input data for training and validation dataset objects
        rf_training, rf_val, rf_test = self._input_processing_pipe(gc_spatial_data_np)

        # rf_all, mnist_targets = self._get_mnist_data()
        # rf_training = rf_all[:60000]
        # rf_target = mnist_targets[:60000]
        # rf_test = rf_all[60000:65000]  # None
        # rf_val = rf_all[65000:70000]  # None

        if self.model_type == "VAE":
            spatial_vae, fit_history = self._fit_spatial_vae(rf_training, rf_val)
            # spatial_vae, validation_data, fit_history = self._k_fold_cross_validation(rf_training, n_folds=5)

            # Plot history of training and validation loss
            self._plot_fit_history(fit_history)

        elif self.model_type == "TwoStageVAE":
            spatial_vae, fit_history, fit_history_stage2 = self._fit_spatial_vae(
                rf_training, rf_val
            )
            self._plot_fit_history(fit_history)
            self._plot_fit_history(fit_history_stage2)

        if rf_test is not None:
            # Get fid score
            fid_score_test = self._get_fid_score(
                spatial_vae, rf_test, image_range=(0, 1)
            )
            print(f"FID score on test data: {fid_score_test}")

            # get ssim score
            ssim_score_test = self._get_ssim_score(
                spatial_vae, rf_test, image_range=(0, 1)
            )
            print(f"SSIM score on test data: {ssim_score_test}")

        # Quality of fit
        self.plot_latent_space(spatial_vae, n=20)

        # Plot latent space, previously plot_z_mean_in_2D(z_mean)
        # self.plot_label_clusters(spatial_vae, rf_training, rf_target)
        self.plot_label_clusters(spatial_vae, rf_training, labels=None)

        # Get a random sample of size n_samples from the data
        n_samples = 1000

        random_sample = np.random.choice(rf_training.shape[0], n_samples, replace=False)
        rf_sample = rf_training[random_sample]

        predictions, z_mean, z_log_var = spatial_vae.predict(rf_sample)

        # # Random sample from latent space
        reconstruction = self._generate_from_latent_space(
            model=spatial_vae, n_samples=n_samples
        )

        # rf_validation_ds_np = self._to_numpy_array(rf_validation_ds)
        self.plot_sample_images(
            [rf_sample, predictions, reconstruction.numpy()],
            labels=["test", "pred", "randreco"],
            n=10,
        )

        plt.show()
