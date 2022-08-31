# Numerical
import numpy as np
# import scipy.optimize as opt
# import scipy.io as sio
# from scipy.optimize import curve_fit
from scipy.ndimage import rotate, fourier_shift 
# import scipy.ndimage as ndimage
from scipy.interpolate import RectBivariateSpline
# import pandas as pd

# Machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras import backend as keras_backend
from keras.preprocessing.image import ImageDataGenerator


# Viz
# from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
# from retina.retina_math_module import RetinaMath
from retina.apricot_fitter_module import ApricotData

# Builtin
import sys
import pdb
import os


# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        z_dist = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_dist


class VAE(keras.Model):
    def __init__(self, image_shape=None, latent_dim=None, **kwargs):
        # super(VAE, self).__init__(**kwargs)
        super().__init__(**kwargs)

        assert image_shape is not None, 'Argument image_shape  must be specified, aborting...'
        assert latent_dim is not None, 'Argument latent_dim must be specified, aborting...'

        self.beta = 1.0

        """
        Build encoder
        """
        # encoder_inputs = keras.Input(shape=image_shape)
        # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(16, activation="relu")(x)
        # z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        # z = Sampling()([z_mean, z_log_var])
        # self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # self.encoder.summary()

        encoder_inputs = keras.Input(shape=image_shape)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        self.encoder.summary()

        '''
        Build decoder
        '''

        # latent_inputs = keras.Input(shape=(latent_dim,))
        # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        # self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        # self.decoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

        self.sampler = Sampler()

        # self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # self.reconstruction_loss_tracker = keras.metrics.Mean(
        #     name="reconstruction_loss"
        # )
        # self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
        name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z_random = Sampler()(z_mean, z_log_var)
        reconstruction = self.decoder(z_random)
        return reconstruction, z_mean, z_log_var 

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
            ]

    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         z_mean, z_log_var, z = self.encoder(data)
    #         reconstruction = self.decoder(z)
    #         reconstruction_loss = tf.reduce_mean(
    #             tf.reduce_sum(
    #                 keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
    #             )
    #         )
    #         kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #         kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #         total_loss = reconstruction_loss + kl_loss
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     self.kl_loss_tracker.update_state(kl_loss)
    #     return {
    #         "loss": self.total_loss_tracker.result(),
    #         "reconstruction_loss": self.reconstruction_loss_tracker.result(),
    #         "kl_loss": self.kl_loss_tracker.result(),
    #     }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + self.beta * tf.reduce_mean(kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class ApricotVAE(ApricotData, VAE):
# class ApricotVAE(VAE, ApricotData):
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
        self.latent_dim = 2
        self.image_shape = (28, 28, 1) # Images will be smapled to this space. If you change this you need to change layers, too, for consistent output shape

        self.batch_size = None # None will take the batch size from test_split size
        self.epochs = 60
        self.test_split = 0.2   # Split data for testing
        self.verbose = 2 #  'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
        
        # Augment data. Final n samples =  n * (1 + n_repeats * 2): n is the original number of samples, 2 is the rot & shift 
        self.n_repeats = 5 # Each repeated array of images will have only one transformation applied to it (same rot or same shift).
        self.angle_min = -30 # rotation int in degrees
        self.angle_max = 30
        self.shift_min = -5 # shift int in pixels (upsampled space, rand int for both x and y)
        self.shift_max = 5

        self.random_seed = 42
        tf.keras.utils.set_random_seed(self.random_seed)

        self.beta = 1 # Beta parameter for KL loss

        self.optimizer = keras.optimizers.Adam(lr = 0.001)  # default lr = 0.001

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
        scale = 1.0
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

    def _fit_spatial_vae(self, data):
       
        # Build model
        vae = VAE(image_shape=self.image_shape, latent_dim=self.latent_dim)

        # change beta
        vae.beta = self.beta
        
        # Compile model
        vae.compile(optimizer=self.optimizer)

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

    def _rotate_and_repeat_datasets(self, np_array, n_repeats):
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

    def _input_processing_pipe(self, data_np):
        """
        Process input data for training and validation
        """
        # Up or downsample 2D image data
        data_us = self._resample_data(data_np, self.image_shape[:2])

        # Normalize data
        data_usn = self._normalize_data(data_us, scale_type='minmax', scale_min=None, scale_max=None)
        
        
        # split data into test and validation sets using proportion of 20%
        skip_size = int(data_us.shape[0] * self.test_split)
        data_train_np = data_usn[skip_size:]
        data_test_np = data_usn[:skip_size]
        
        # Report split sizes
        print(f'The train sample size before augmentation is {data_train_np.shape[0]}')
        print(f'The test sample size before augmentation is {data_test_np.shape[0]}')
        
        # Augment data with random rotation and repeat
        # Rotate between repeats
        data_train_np_reps = self._rotate_and_repeat_datasets(data_train_np, self.n_repeats) 
        data_test_np_reps = self._rotate_and_repeat_datasets(data_test_np, self.n_repeats)

        # Report split sizes
        print(f'The train sample size after augmentation is {data_train_np_reps.shape[0]}')
        print(f'The test sample size after augmentation is {data_test_np_reps.shape[0]}')

        # Set batch size as the number of samples in the smallest dataset
        if self.batch_size == None:
            self.batch_size = min(data_train_np_reps.shape[0], data_test_np_reps.shape[0])

        return data_train_np_reps, data_test_np_reps
    
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

    def _get_mnist_data(self):
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        return mnist_digits

    def _fit_all(self):

        # Get numpy array of data in correct dimensions
        gc_spatial_data_np = self._get_spatial_apricot_data()

        
        # # Process input data for training and validation dataset objects
        rf_training, rf_test = self._input_processing_pipe(gc_spatial_data_np)

        # rf_training = self._get_mnist_data()

        spatial_vae, fit_history = self._fit_spatial_vae(rf_training)

        # Plot history of training and validation loss
        self._plot_fit_history(fit_history)

        # Quality of fit
        n_samples = 5
        self.plot_latent_space(spatial_vae, n=n_samples)
        predictions, z_mean, z_log_var = spatial_vae.predict(rf_test)
        print(spatial_vae.evaluate(rf_test, rf_test))

        # Random sample from latent space
        # SEURAAVA ON ILMEISESTI VÄÄRÄ TAPA SAADA RANDOM SAMPPELI LATENTISTA AVARUUDESTA.
        # HOMMAA VARTEN LIENEE KUVAUS SEURAAVASSA: https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
        z_random = Sampler()(z_mean, z_log_var).numpy()
        reconstruction = spatial_vae.decoder(z_random)

        # rf_validation_ds_np = self._to_numpy_array(rf_validation_ds)
        self.plot_sample_images([rf_test, predictions, reconstruction.numpy()], labels=['test', 'pred', 'randreco'], n=n_samples)
        # self.plot_sample_images(reconstruction.numpy())

        plt.show()
        sys.exit()
