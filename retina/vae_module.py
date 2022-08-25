# Numerical
import numpy as np
# import scipy.optimize as opt
# import scipy.io as sio
# from scipy.optimize import curve_fit
from scipy.ndimage import rotate 
# import scipy.ndimage as ndimage
from scipy.interpolate import RectBivariateSpline
# import pandas as pd

# Machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras import backend as keras_backend



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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, image_shape=None, latent_dim=None, **kwargs):
        super(VAE, self).__init__(**kwargs)

        assert image_shape is not None, 'Argument image_shape  must be specified, aborting...'
        assert latent_dim is not None, 'Argument latent_dim must be specified, aborting...'

        """
        Build encoder
        """
        encoder_inputs = keras.Input(shape=image_shape)
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
        # self.spatial_filter_data = self.read_spatial_filter_data()
        # self.spatial_filter_sums = self.compute_spatial_filter_sums()
        # self.temporal_filter_sums = self.compute_temporal_filter_sums()
        # self.bad_data_indices = self.spatial_filter_data[2]

        # Set common VAE model parameters
        self.latent_dim = 2
        self.image_shape = (28, 28, 1) # If you change this you need to change layers, too, for consistent output shape
        # self.image_shape = (13, 13, 1)
        # self.resample_size = (28, 28) # x, y

        # TÄHÄN JÄIT. NÄILLÄ PARAMETREILLA MENEE LAHJAKKAASTI PERSIILLEEN
        self.batch_size = 32
        self.epochs = 100
        self.n_repeats = 10
        self.angle_min = -45 # int in degrees
        self.angle_max = 45
        self.buffer_size = 1000
        self.test_validation_split = 0.2
        self.rotation_seed = 1
        self.shuffle_seed = 42

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)  # default lr = 0.001

        n_threads = 30
        self._set_n_cpus(n_threads)
        
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

    def plot_sample_images(self, image_array, n=10, sample=None):
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

    def _fit_spatial_vae(self, data, validation_data=None):

        # Build model
        vae = VAE(image_shape=self.image_shape, latent_dim=self.latent_dim)
        
        # Compile model
        vae.compile(optimizer=self.optimizer)

        # Fit model
        fit_history = vae.fit(data, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data)

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

    def _rotate_and_repeat_datasets(self, ds, n_repeats):
        """
        Rotate and repeat datasets
        """
        def _random_rotate_image(image):
            
            image = rotate(image, np.random.uniform(self.angle_min, self.angle_max), reshape=False, mode='reflect')
            return image

        # @tf.autograph.experimental.do_not_convert
        def _tf_random_rotate_image(image):

            im_shape = image.shape
            [image,] = tf.py_function(_random_rotate_image, [image], [tf.float64])
            image.set_shape(im_shape)
            return image

        np.random.seed(self.rotation_seed)

        ds_rep = ds
        for _ in range(n_repeats):
            rot_ds = ds.map(_tf_random_rotate_image)
            ds_rep = ds_rep.concatenate(rot_ds)

        return ds_rep

    def _to_numpy_array(self, ds):
        """
        Convert tf.data.Dataset to numpy array e.g. for easier visualization and analysis
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

    def _normalize_data(self, data):
        """
        Normalize data to [0, 1]
        """
        data_min = data.min()
        data_scale = data.max() - data_min
        data = (data - data_min) / data_scale
        
        # Get data transfer parameters 
        self.data_min = data_min
        self.data_scale = data_scale
        
        return data
    
    def _input_processing_pipe(self, data_np):
        """
        Process input data for training and validation
        """
        # Up or downsample 2D image data
        data_us = self._resample_data(data_np, self.image_shape[:2])

        # Normalize data
        data_usn = self._normalize_data(data_us)

        # Turn data from numpy array to tensorflow dataset
        data_tf = tf.data.Dataset.from_tensor_slices(data_usn)

        # Shuffle data
        data_tf = data_tf.shuffle(buffer_size=self.buffer_size, seed=self.shuffle_seed)

        # split data into test and validation sets using proportion of 20%
        skip_size = int(data_us.shape[0] * self.test_validation_split)
        data_tf_train = data_tf.skip(skip_size, name='train')
        data_tf_test = data_tf.take(skip_size, name='test')
        
        # Report split sizes
        print(f'The test sample size is {tf.data.experimental.cardinality(data_tf_train).numpy()}')
        print(f'The validation sample size is {tf.data.experimental.cardinality(data_tf_test).numpy()}')
        

        # Augment data with random rotation and repeat
        # Rotate between repeats
        data_tf_train_reps = self._rotate_and_repeat_datasets(data_tf_train, self.n_repeats) 
        data_tf_test_reps = self._rotate_and_repeat_datasets(data_tf_test, self.n_repeats)
        
        # Batch data for more efficient processing
        data_tf_train_batches = data_tf_train_reps.batch(self.batch_size, drop_remainder=True)
        data_tf_test_batches = data_tf_test_reps.batch(self.batch_size, drop_remainder=True)
            
        # Quality check on data
        if 0:
            self._plot_batch_sizes(data_tf_train_batches)
            data_tf_np = self._to_numpy_array(data_tf_train_reps)
            this_sample = 12
            sample_images = [this_sample, len(data_tf_train) + this_sample, 2 * len(data_tf_train) + this_sample]  
            self.plot_sample_images([data_tf_np], sample=sample_images)
            plt.show()

        return data_tf_train_batches, data_tf_test_batches
    
    def _fit_all(self):

        # Get numpy array of data in correct dimensions
        gc_spatial_data_np = self._get_spatial_apricot_data()

        # Process input data for training and validation dataset objects
        rf_training_ds, rf_validation_ds = self._input_processing_pipe(gc_spatial_data_np)
        
        spatial_vae, fit_history = self._fit_spatial_vae(rf_training_ds, validation_data=rf_validation_ds)
        n_samples = 5

        # Quality of fit
        self.plot_latent_space(spatial_vae, n=n_samples)
        predictions, z_mean, z_log_var, z_input = spatial_vae.predict(rf_validation_ds)

        # Random sample from latent space
        # SEURAAVA ON ILMEISESTI VÄÄRÄ TAPA SAADA RANDOM SAMPPELI LATENTISTA AVARUUDESTA.
        # HOMMAA VARTEN LIENEE KUVAUS SEURAAVASSA: https://blog.tensorflow.org/2019/03/variational-autoencoders-with.html
        z_random = Sampling()([z_mean, z_log_var]).numpy()
        reconstruction = spatial_vae.decoder(z_random)

        rf_validation_ds_np = self._to_numpy_array(rf_validation_ds)
        self.plot_sample_images([rf_validation_ds_np, predictions, reconstruction.numpy()], n=n_samples)
        # self.plot_sample_images(reconstruction.numpy())

        plt.show()
        sys.exit()
