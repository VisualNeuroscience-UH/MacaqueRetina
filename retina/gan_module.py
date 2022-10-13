import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pdb
import matplotlib.pyplot as plt


"""
Copied from https://keras.io/examples/generative/wgan_gp/
"""


class WGAN(keras.Model):
    def __init__(
        self,
        img_shape,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        discriminator = self.get_discriminator_model(img_shape)
        generator = self.get_generator_model()

        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                # REAL IMG GETS MULTIPLE VALUES -- MISTÄ TULEE. VERTAILIT ORIGINAL JA TÄTÄ, JOTKA NÄYTTÄVÄT SAMANKOKOISILTA
                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def conv_block(
        self,
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
    ):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_discriminator_model(self, img_shape=(32, 32, 3)):
        img_input = layers.Input(shape=img_shape)
        # Zero pad the input to make the input images size to (32, 32, 1).
        x = layers.ZeroPadding2D((2, 2))(img_input)
        x = self.conv_block(
            x,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            512,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

        d_model = keras.models.Model(img_input, x, name="discriminator")
        return d_model

    def upsample_block(
        self,
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    ):
        x = layers.UpSampling2D(up_size)(x)
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = layers.BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_generator_model(self):
        noise = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Reshape((4, 4, 256))(x)
        x = self.upsample_block(
            x,
            128,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x,
            64,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x, 1, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
        )
        # At this point, we have an output which has the same shape as the input, (32, 32, 1).
        # We will use a Cropping2D layer to make it (28, 28, 1).
        x = layers.Cropping2D((2, 2))(x)

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


class ApricotGAN:
    def __init__(self, train_images):
        # self.data_np = data_np
        self.epochs = 20
        self.batch_size = 512
        self.noise_dim = 128

        self.img_shape = (28, 28, 1)

        # Set the number of epochs for trainining.
        self.epochs = 20

        # Eager mode works like normal python code. You can access variables better.
        # Graph mode postpones computations, but is more efficient.
        tf.config.run_functions_eagerly(True)

        print(f"Number of examples: {len(train_images)}")
        print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

        # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
        train_images = train_images.reshape(
            train_images.shape[0], *self.img_shape
        ).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        # Instantiate the optimizer for both networks
        # (learning_rate=0.0002, beta_1=0.5 are recommended)
        generator_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )

        # Instantiate the customer `GANMonitor` Keras callback.
        cbk = GANMonitor(num_img=3, latent_dim=self.noise_dim)

        # Get the wgan model
        wgan = WGAN(
            img_shape=self.img_shape,
            latent_dim=self.noise_dim,
            discriminator_extra_steps=3,
        )

        wgan.discriminator.summary()
        wgan.generator.summary()

        # Compile the wgan model
        wgan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            g_loss_fn=wgan.generator_loss,
            d_loss_fn=wgan.discriminator_loss,
        )

        # Start training
        wgan.fit(
            train_images,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[cbk],
        )


# from IPython.display import Image, display

# display(Image("generated_img_0_19.png"))
# display(Image("generated_img_1_19.png"))
# display(Image("generated_img_2_19.png"))

if __name__ == "__main__":
    # Load the dataset
    (train_images, train_labels), (_, _) = keras.datasets.fashion_mnist.load_data()

    # Create an instance of the ApricotGAN class
    apricot_gan = ApricotGAN(train_images)

    # # Train the GAN
    # apricot_gan.train()

    # # Display the generated images
    # from IPython.display import Image, display

    # display(Image("generated_img_0_19.png"))
    # display(Image("generated_img_1_19.png"))
    # display(Image("generated_img_2_19.png"))

    # fashion_mnist = keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
