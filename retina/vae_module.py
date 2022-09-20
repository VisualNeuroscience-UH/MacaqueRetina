# Numerical
import numpy as np
from scipy.ndimage import rotate, fourier_shift 
from scipy.interpolate import RectBivariateSpline
from scipy import linalg

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

# Builtin
import sys
import pdb
import os


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        z_dist = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # z_dist = z_mean # my autoencoder
        return z_dist


class VAE(keras.Model):
    def __init__(self, image_shape=None, latent_dim=None, val_data=None, **kwargs):
        # super(VAE, self).__init__(**kwargs)
        super().__init__(**kwargs)

        assert image_shape is not None, 'Argument image_shape  must be specified, aborting...'
        assert latent_dim is not None, 'Argument latent_dim must be specified, aborting...'

        # self.beta = 1.0
        # Init attribute for validation. We lack custom fit() method, so we need to pass validation data to train_step()
        self.val_data = val_data

        """
        Build encoder
        """

        encoder_inputs = keras.Input(shape=image_shape)
        # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(16, activation="relu")(x)
        x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        self.encoder.summary()

        '''
        Build decoder
        '''
        latent_inputs = keras.Input(shape=(latent_dim,))
        # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

        self.sampler = Sampler()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

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
            self.val_loss_tracker
            ]

    def train_step(self, data):
        mse=keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = mse(data, reconstruction)

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

        if self.val_data is not None:
            val_z_mean, _ = self.encoder(self.val_data)
            val_reconstruction = self.decoder(val_z_mean)
            val_loss = mse(self.val_data, val_reconstruction)
            self.val_loss_tracker.update_state(val_loss)
            val_loss = self.val_loss_tracker.result()
        else:
            val_loss = 0.0

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "val_reconstruction_loss": val_loss,
        }

class TwoStageVAE(keras.Model):
    def __init__(self, image_shape=None, latent_dim=None, val_data=None, **kwargs):
        super().__init__(**kwargs)

        assert image_shape is not None, 'Argument image_shape  must be specified, aborting...'
        assert latent_dim is not None, 'Argument latent_dim must be specified, aborting...'

        # Init attribute for validation. We lack custom fit() method, so we need to pass validation data to train_step()
        self.val_data = val_data

        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.second_depth = 3
        self.second_dim=1024

        self._build_encoder1()
        self._build_decoder1()
        self._build_encoder2()
        self._build_decoder2()
        self._build_loss()

        pdb.set_trace()
        self.sampler = Sampler()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    def _build_encoder1(self):
        encoder_inputs = keras.Input(shape=self.image_shape)
        x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        self.encoder.summary()

    def _build_encoder2(self):
        t = self.z 
        for i in range(self.second_depth):
            t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
        t = tf.concat([self.z, t], -1)
    
        self.mu_u = tf.layers.dense(t, self.latent_dim, name='mu_u')
        self.logsd_u = tf.layers.dense(t, self.latent_dim, name='logsd_u')
        self.sd_u = tf.exp(self.logsd_u)
        self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim])

    def _build_decoder1(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    def _build_decoder2(self):
        t = self.u 
        for i in range(self.second_depth):
            t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
        t = tf.concat([self.u, t], -1)

        self.z_hat = tf.layers.dense(t, self.latent_dim, name='z_hat')
        self.loggamma_z = tf.get_variable('loggamma_z', [], tf.float32, tf.zeros_initializer())
        self.gamma_z = tf.exp(self.loggamma_z)

    def _build_loss(self):
        HALF_LOG_TWO_PI = 0.91893 # np.log(2*np.pi)*0.5

        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        if not self.cross_entropy_loss:
            self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        else:
            self.gen_loss1 = -tf.reduce_sum(self.x * tf.log(tf.maximum(self.x_hat, 1e-8)) + (1-self.x) * tf.log(tf.maximum(1-self.x_hat, 1e-8))) / float(self.batch_size)
        self.loss1 = self.kl_loss1 + self.gen_loss1 

        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 


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
            self.val_loss_tracker
            ]

    def train_step(self, data):
        mse=keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = mse(data, reconstruction)

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

        if self.val_data is not None:
            val_z_mean, _ = self.encoder(self.val_data)
            val_reconstruction = self.decoder(val_z_mean)
            val_loss = mse(self.val_data, val_reconstruction)
            self.val_loss_tracker.update_state(val_loss)
            val_loss = self.val_loss_tracker.result()
        else:
            val_loss = 0.0

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "val_reconstruction_loss": val_loss,
        }


class ApricotVAE(ApricotData, VAE):
    """
    Class for creating model for variational autoencoder from  Apricot data 
    """

    # Load temporal and spatial data from Apricot data
    def __init__(self, apricot_data_folder, gc_type, response_type):
        super().__init__(apricot_data_folder, gc_type, response_type)

        self.temporal_filter_data = self.read_temporal_filter_data()
        # self.spatial_filter_data = self.read_spatial_filter_data()
        # self.spatial_filter_sums = self.compute_spatial_filter_sums()
        # self.temporal_filter_sums = self.compute_temporal_filter_sums()
        # self.bad_data_indices = self.spatial_filter_data[2]

        # Set common VAE model parameters
        self.latent_dim = 32 # 2
        self.image_shape = (28, 28, 1) # Images will be smapled to this space. If you change this you need to change layers, too, for consistent output shape
        self.latent_space_plot_scale = 4 # Scale for plotting latent space

        self.batch_size = 16 # None will take the batch size from test_split size. Note that the batch size affects training speed and loss values
        self.epochs = 200
        self.test_split = 0.2   # Split data for validation and testing (both will take this fraction of data)
        self.verbose = 2 #  'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
        
        # Preprocessing parameters
        self.gaussian_filter_size = None # None or 0.5 or ... # Denoising gaussian filter size (in pixels). None does not apply filter
        self.n_pca_components = 32 # None or 32 # Number of PCA components to use for denoising. None does not apply PCA
        
        # Augment data. Final n samples =  n * (1 + n_repeats * 2): n is the original number of samples, 2 is the rot & shift 
        self.n_repeats = 10 # Each repeated array of images will have only one transformation applied to it (same rot or same shift).
        self.angle_min = -30 #30 # rotation int in degrees
        self.angle_max = 30 #30
        self.shift_min = -5 #5 # shift int in pixels (upsampled space, rand int for both x and y)
        self.shift_max = 5 #5

        self.random_seed = 42
        tf.keras.utils.set_random_seed(self.random_seed)

        self.beta = 1 # Beta parameter for KL loss. Overrides VAE class beta parameter

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)  # default lr = 0.001

        n_threads = 30
        self._set_n_cpus(n_threads)

        # Eager mode works like normal python code. You can access variables better. 
        # Graph mode postpones computations, but is more efficient.
        tf.config.run_functions_eagerly(False) 

        self._fit_all()
    
    def _set_n_cpus(self, n_threads):
        # Set number of CPU cores to use for parallel processing
        # NOTE Not sure this has any effect
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(n_threads)

        tf.config.threading.set_inter_op_parallelism_threads(
            n_threads
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            n_threads
        )
        tf.config.set_soft_device_placement(True)
    
    def plot_latent_space(self, vae, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = self.image_shape[0] # side length of the digits
        scale = self.latent_space_plot_scale
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                if self.latent_dim > 2:
                    # Add the extra dimensions as zeros to the z_sample
                    z_sample = np.concatenate((z_sample, np.zeros((1, self.latent_dim - 2))), axis=1)
                    # z_sample = np.concatenate((z_sample, np.ones((1, self.latent_dim - 2))), axis=1)
                print(f'z_sample: {z_sample}')
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
        # set title as "latent space"
        plt.title("latent space")

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
                assert isinstance(image_array_i, np.ndarray), "Supplied array is not of type numpy.ndarray, aborting..."
        else:
            nrows = 1
            image_array = [image_array]

        if not len(image_array_i) == len(image_array[0]):
            print(  """Supplied arrays are of different length. 
                        If no sample list is provided, 
                        indices will be drawn randomly from the first array.""")

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
            if labels is not None and i==0:
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
                    plt.imshow(image2.reshape(image2_shape), vmin=images2_min, vmax=images2_max)
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if labels is not None and j==0:
                        plt.title(labels[i + 1])
                plt.colorbar()

    def _resample_data(self, data, resampled_size):
        """
        Up- or downsample data
        """
        
        assert len(resampled_size) == 2, 'Resampled_size must be a 2-element array, aborting...'

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
            aa = data[this_sample, :, :] # 2-D array of data with shape (x.size,y.size)
            f = RectBivariateSpline(x_orig_grid, y_orig_grid, aa)
            upsampled_data[this_sample, :, :, 0] = f(x_new_grid, y_new_grid)

        return upsampled_data

    def _get_spatial_vae_model(self, val_data):
        """
        Builds a model for the spatial VAE
        """
        # Build model
        # vae = VAE(image_shape=self.image_shape, latent_dim=self.latent_dim, val_data=val_data)
        vae = TwoStageVAE(image_shape=self.image_shape, latent_dim=self.latent_dim, val_data=val_data)
        pdb.set_trace()
        # change beta
        vae.beta = self.beta
        
        # Compile model
        vae.compile(optimizer=self.optimizer)

        return vae

    def _fit_spatial_vae(self, data, val_data):
       
        vae = self._get_spatial_vae_model(val_data)

        # Fit model
        fit_history = vae.fit(data, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=self.verbose)

        return vae, fit_history
    
    def _get_spatial_apricot_data(self):
        """
        Get spatial data from file using the inherited method read_spatial_filter_data()
        """
        gc_spatial_data_np_orig, _, bad_data_indices = self.read_spatial_filter_data()

        # drop bad data
        gc_spatial_data_np = np.delete(gc_spatial_data_np_orig, bad_data_indices, axis=2)

        # reshape to (n_samples, xdim, ydim, 1)
        gc_spatial_data_np = np.moveaxis(gc_spatial_data_np, 2, 0)
        gc_spatial_data_np = np.expand_dims(gc_spatial_data_np, axis=3)
 
        return gc_spatial_data_np

    def _plot_batch_sizes(self, ds):
        
        plt.figure()
        batch_sizes = [batch.shape[0] for batch in ds]
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.xlabel('Batch number')
        plt.ylabel('Batch size')

    def _augment_data(self, np_array, n_repeats):
        """
        Rotate and repeat datasets
        """
        def _random_rotate_image(image, rot_angle):
            
            assert image.ndim == 4, 'Image array must be 4-dimensional for rotation axes to match, aborting...'

            image_rot = rotate(image, rot_angle, axes=(2, 1), reshape=False, mode='reflect')
            return image_rot

        def _random_shift_image(image, shift):
                
            assert image.ndim == 4, 'Image array must be 4-dimensional for shift axes to match, aborting...'
            image_shifted = image.copy()
            for this_image_idx in range(image.shape[0]):
                this_image = image[this_image_idx, :, :, 0]
                input_ = np.fft.fft2(this_image)
                result = fourier_shift(input_, shift=shift) # shift in pixels, tuple of (y, x) shift
                # result = fourier_shift(input_, shift=(-5, 0)) # shift in pixels, tuple of (y, x) shift
                result = np.fft.ifft2(result)
                image_shifted[this_image_idx, :, :, 0] = result.real

            return image_shifted   

        def _rotate_and_shift_image_QA(image, image_transf, this_ima=0, this_transf=0):

            n_ima = image.shape[0]

            plt.figure()
            plt.imshow(image[this_ima,:,:,0])
            plt.title('Original image')
            plt.figure()
            plt.imshow(image_transf[this_ima + n_ima * this_transf,:,:,0])
            plt.title('Transformed image')
            plt.show()


        n_orig_images = np_array.shape[0]
        
        # Augment with rotations
        # n_repeats which will be rotated
        np_array_rep = np.repeat(np_array, n_repeats, axis=0) 
        rot = np.random.uniform(self.angle_min, self.angle_max, size=n_repeats)

        for rot_idx, this_rot in enumerate(rot):
            np_array_rep[rot_idx * n_orig_images : (rot_idx + 1) * n_orig_images, :, :] = _random_rotate_image(np_array, this_rot)
            # rot_all = rot_all + [this_rot] * n_orig_images # append this_rot to list rot_all n_orig_images times

        np_array_augmented = np.vstack((np_array, np_array_rep))

        # Augment with shifts
        # n_repeats which will be shifted
        np_array_rep = np.repeat(np_array, n_repeats, axis=0)
        shift = np.random.uniform(self.shift_min, self.shift_max, size=(n_repeats, 2))

        for shift_idx, this_shift in enumerate(shift):
            np_array_rep[shift_idx * n_orig_images : (shift_idx + 1) * n_orig_images, :, :] = _random_shift_image(np_array, this_shift)

        # if 0:
        #     _rotate_and_shift_image_QA(np_array, np_array_rep, this_ima=0, this_transf=2)
        
        np_array_augmented = np.vstack((np_array_augmented, np_array_rep))

        return np_array_augmented

    def _to_numpy_array(self, ds):
        """
        Convert tf.data.Dataset to numpy array e.g. for easier visualization and analysis
        This function is on hold, because I am working directly with numpy arrays at the moment (small datasets)
        """

        if isinstance(ds, tf.data.Dataset):
            ds_np =  np.stack(list(ds))
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

    def _normalize_data(self, data, scale_type='standard', scale_min=None, scale_max=None):
        """
        Normalize data to either mean 0, std 1 or [scale_min, scale_max] 

        :param data: numpy array
        :param type: 'standard' or 'minmax'
        :param scale_min: minimum value of scaled data
        :param scale_max: maximum value of scaled data
        """
        
        if scale_type == 'standard':
            data_mean = data.mean()
            data_std = data.std()
            data = (data - data_mean) / data_std
            
            # Get data transfer parameters 
            self.data_mean = data_mean
            self.data_std = data_std
        elif scale_type == 'minmax':
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
            print('No filtering with Gaussian filter')
            return data

        dataf = np.zeros(data.shape)
        for idx in range(data.shape[0]):

            # Apply gaussian filter
            dataf[idx, :, :, 0] = gaussian(data[idx, :, :, 0], sigma=[filter_size_pixels, filter_size_pixels], channel_axis=-1, mode='reflect', truncate=2.0)
 
        return dataf.astype("float32")    
        
    def _filter_data_PCA(self, data):
        """
        Filter data with PCA
        :param data: numpy array
        """
                
        n_pca_components = self.n_pca_components

        if n_pca_components is None:
            print('No filtering with PCA')
            return data, None

        # PCA without Gaussian filter
        # Flatten images to provide 2D input for PCA. Each row is an image, each column a pixel
        data_2D_np = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        
        # Apply PCA to data
        pca = PCA(n_components=n_pca_components)
        pca.fit(data_2D_np)

        # Get inverse PCA transformation
        pca_inv = pca.inverse_transform(pca.transform(data_2D_np))

        # Resahpe to 3D
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
            plt.imshow(data[example_image_idx,:,:])
            plt.colorbar()
            plt.title('Original image')

            plt.subplot(1, 2, 2)
            plt.imshow(dataf[example_image_idx,:,:])
            plt.colorbar()
            plt.title('Gaussian filtered image')
            
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
            plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
            plt.title('PCA explained variance ratio')

            # Plot first 2 PCA components as scatter plot
            plt.subplot(2, 2, 2)
            plt.scatter(pca.components_[0,:], pca.components_[1,:])
            plt.title('PCA components 1 and 2')

            # From the previous subplot, color the example_image red
            plt.scatter(pca.components_[0,example_image_idx], pca.components_[1,example_image_idx], c='r')

            # Show exaple original and filtered image
            plt.subplot(2, 2, 3)
            plt.imshow(data[example_image_idx,:,:])
            plt.colorbar()
            plt.title('Original image')

            plt.subplot(2, 2, 4)
            plt.imshow(data_pca[example_image_idx,:,:])
            plt.colorbar()
            plt.title('PCA reconstructed image')
            
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
        data_usn = self._normalize_data(data_us, scale_type='minmax', scale_min=None, scale_max=None)
                
        # split data into train, validation and test sets using proportion of 60%, 20%, 20%
        skip_size = int(data_us.shape[0] * self.test_split)
        data_train_np = data_usn[skip_size * 2:]
        data_val_np = data_usn[:skip_size]
        data_test_np = data_usn[skip_size:skip_size * 2]
        
        # Report split sizes
        print(f'The train sample size before augmentation is {data_train_np.shape[0]}')
        print(f'The validation sample size is {data_val_np.shape[0]}')
        print(f'The test sample size is {data_test_np.shape[0]}')
        
        # Augment data with random rotation and shift
        data_train_np_reps = self._augment_data(data_train_np, self.n_repeats) 

        # Report split sizes
        print(f'The train sample size after augmentation is {data_train_np_reps.shape[0]}')

        # Set batch size as the number of samples in the smallest dataset
        if self.batch_size == None:
            self.batch_size = min(data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0])
            print (f'Batch size is {self.batch_size}')
        elif self.batch_size > min(data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0]):
            self.batch_size = min(data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0])
            print(f'Batch size is set to {self.batch_size} to match the smallest dataset')
        elif self.batch_size < min(data_train_np_reps.shape[0], data_val_np.shape[0], data_test_np.shape[0]):
            print(f'Validation and test data size is set to {self.batch_size} to match the batch size')
            data_val_np = data_val_np[:self.batch_size]
            data_test_np = data_test_np[:self.batch_size]
        print('\n') 

        return data_train_np_reps, data_val_np, data_test_np
    
    def _plot_fit_history(self, fit_history):
        """
        Plot the fit history
        """

        plt.figure()
        keys = [k for k in fit_history.history.keys()]
        for key in keys:
            plt.plot(fit_history.history[key])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(keys, loc='upper left')

        val_min = min(fit_history.history['val_reconstruction_loss'])
        # plot horizontal dashed red line at the minimum of val_reconstruction_loss
        plt.axhline(val_min, color='r', linestyle='dashed', linewidth=1)
        
        # get the max value of x-axis
        x_max = plt.xlim()[1]
        # add the minimum value as text
        plt.text(x_max, val_min, round(val_min, 1), ha="right", va="bottom", color="r")

    def _plot_z_mean_in_2D(self, z_mean):
        plt.figure()
        plt.scatter(z_mean[:,0], z_mean[:,1])
        plt.title('Latent space')

    def _get_mnist_data(self):
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        return mnist_digits

    def _k_fold_cross_validation(self, data_np, n_folds=3):
        """
        Perform k-fold cross validation on the data
        """
        # loss for validation reconstruction
        mse=keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        num_validation_samples = len(data_np) // n_folds
        np.random.shuffle(data_np)
        
        validation_scores = []
        reconstruction_losses = []
        
        for fold in range(n_folds):

            validation_data = data_np[num_validation_samples * fold:num_validation_samples * (fold + 1)]
            training_data = np.vstack((data_np[:num_validation_samples * fold], data_np[num_validation_samples * (fold + 1):]))

            model = self._get_spatial_vae_model()
            model.fit(training_data, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=self.verbose)
            
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
        fit_history = model.fit(data_np, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=self.verbose)

        return model, (validation_score, validation_score_std), fit_history

    def fid_score(codes_g, codes_r, eps=1e-6):
        d = codes_g.shape[1]
        assert codes_r.shape[1] == d
        
        mn_g = codes_g.mean(axis=0)
        mn_r = codes_r.mean(axis=0)

        cov_g = np.cov(codes_g, rowvar=False)
        cov_r = np.cov(codes_r, rowvar=False)

        covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
        if not np.isfinite(covmean).all():
            cov_g[range(d), range(d)] += eps
            cov_r[range(d), range(d)] += eps 
            covmean = linalg.sqrtm(cov_g.dot(cov_r))

        score = np.sum((mn_g - mn_r) ** 2) + (np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
        return score 
    
    def _fit_all(self):

        # Get numpy array of data in correct dimensions
        gc_spatial_data_np = self._get_spatial_apricot_data()

        # # Process input data for training and validation dataset objects
        rf_training, rf_val, rf_test = self._input_processing_pipe(gc_spatial_data_np)

        # rf_training = self._get_mnist_data()

        spatial_vae, fit_history = self._fit_spatial_vae(rf_training, rf_val)
        # spatial_vae, validation_data, fit_history = self._k_fold_cross_validation(rf_training, n_folds=5)

        # print(f'Validation score: {validation_data[0]} +/- {validation_data[1]}')

        # validation_score = spatial_vae.evaluate(rf_test, rf_test)

        # Plot history of training and validation loss
        self._plot_fit_history(fit_history)

        # Quality of fit
        self.plot_latent_space(spatial_vae, n=5)
        predictions, z_mean, z_log_var = spatial_vae.predict(rf_test)
        # print(spatial_vae.evaluate(rf_test, rf_test))

        # Plot latent space
        self._plot_z_mean_in_2D(z_mean)

        # Random sample from latent space
        # SEURAAVA ON ILMEISESTI VÄÄRÄ TAPA SAADA RANDOM SAMPPELI LATENTISTA AVARUUDESTA.
        # HOMMAA VARTEN LIENEE KUVAUS SEURAAVASSA: https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
        z_random = spatial_vae.sampler(z_mean, z_log_var)
        reconstruction = spatial_vae.decoder(z_random)

        # rf_validation_ds_np = self._to_numpy_array(rf_validation_ds)
        n_samples = 10
        self.plot_sample_images([rf_test, predictions, reconstruction.numpy()], labels=['test', 'pred', 'randreco'], n=n_samples)
        # self.plot_sample_images(reconstruction.numpy())

        plt.show()
        sys.exit()
