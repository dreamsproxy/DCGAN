
import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
w_init = tf.keras.initializers.GlorotNormal()

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    #print(images_path)
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
    )(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    return x

def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
    return x


def build_generator(latent_dim, debug=False):
    f = [2**i for i in range(5)][::-1]  # Increase the range to add more layers\
    #print(f)
    filters = 32  # Increase the number of filters
    output_strides = 16  # Decrease the output stride for higher resolution
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides
    if debug:
        print(f[0] * filters * h_output * w_output)
        raise
        
    noise = layers.Input(shape=(latent_dim,), name="generator_noise_input")
    x = layers.Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((h_output, w_output, f[0] * filters))(x)

    for i in range(1, 5):  # Increase the range to add more layers
        x = deconv_block(
            x,
            num_filters=f[i] * filters,
            kernel_size=3,
            strides=2,
            bn=True
        )

    x = deconv_block(
        x,
        num_filters=32,
        kernel_size=3,
        strides=1,
        bn=True
    )

    x = deconv_block(
        x,
        num_filters=16,
        kernel_size=3,
        strides=1,
        bn=True
    )

    x = conv_block(
        x,
        num_filters=IMG_C,
        kernel_size=3,  # Use smaller kernel size for finer details
        strides=1
    )

    fake_output = layers.Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")


def build_discriminator():
    f = [2**i for i in range(5)]  # Increase the range to add more layers
    image_input = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64  # Increase the number of filters
    output_strides = 16  # Decrease the output stride for higher resolution
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 5):  # Increase the range to add more layers
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return Model(image_input, x, name="discriminator")

@tf.function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(samples, output):
    gradients = tf.gradients(output, samples)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1))
    return gradient_penalty

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim, n_critic=1):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        #self.epoch_loss = {"d_loss": [], "g_loss": []}
        
    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        for i in range(self.n_critic):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)

            with tf.GradientTape() as ftape, tf.GradientTape() as rtape:
                real_predictions = self.discriminator(real_images)
                fake_predictions = self.discriminator(generated_images)
                d1_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolated_images = real_images * alpha + generated_images * (1 - alpha)
                interpolated_predictions = self.discriminator(interpolated_images)
                gradients = tf.gradients(interpolated_predictions, interpolated_images)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1))
                d_loss = d1_loss + 10 * gradient_penalty

            d_grads = ftape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gtape:
            generated_images = self.generator(random_latent_vectors)
            fake_predictions = self.discriminator(generated_images)
            g_loss = -tf.reduce_mean(fake_predictions)

        g_grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))


        return {"d_loss": d_loss, "g_loss": g_loss}

def save_plot(test_id, examples, epoch, n, debug = False):
    examples = (examples + 1) / 2.0
    if debug:
        print(examples[0].shape)
        print(examples[0])
    for i in range(n * n):
        examples[i] = examples[i] * 127.5 + 127.5
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i], cmap="gray")
    filename = f"./{test_id}/samples/generated_plot_epoch-{epoch+1}.png"
    plt.tight_layout()
    plt.savefig(filename, )
    plt.close()

def dump_hyperparams(test_id: str, hyperparam_str: str, global_hyperparams: bool = False):
    # Global_hyperparams
    #   It is a boolean statement that
    #   checks wether to use the supplied hyperparams for both
    #   the Discriminator (Critic) model and the Generator (Artist) model
    if global_hyperparams == True:
        hyperparam_str = str("Global params" + hyperparam_str)
    
    # Dump hyperparams to file for tracking
    with open(f"./{test_id}/hyperparams.txt", "w") as outfile:
        outfile.write(hyperparam_str)
    
def prepare_dirs(test_id, delete_dir = False):
    if not os.path.exists(f"./{test_id}/"):
        os.makedirs(f"./{test_id}/checkpoints")
        os.makedirs(f"./{test_id}/samples/")
        os.makedirs(f"./{test_id}/saved_model/")
    
    else:
        print("Directory exists!")
        raise Exception(
            """
            Halting training sequence
            It is not recommended to overwrite an existing session!
            Exiting training sequence!
            """
        )
    
def print_summary(d_model, g_model, DEBUG = False):
    d_model.summary()
    g_model.summary()
    if DEBUG == True:
        raise

if __name__ == "__main__":
    ## Hyperparameters
    img_size = 256
    IMG_H = img_size
    IMG_W = img_size
    IMG_C = 1
    
    latent_dim = 256
    
    batch_size = 10
    num_epochs = 5000
    
    #lr_tests = [0.0001, 0.00009]
    #layer_diff_tests = ["coarse-level", "fine-level"]
    
    
    #for diff in layer_diff_tests:
    test_id = "photography/0"
    LOAD = False
    g_model_learning_rate = 0.0001
    d_model_learning_rate = 0.0003
    model_beta_1 = 0.5
    model_amsgrad = True
    test_id = test_id
    global_hyperparams = False
    
    hyperparams = str(
    f"""
    img_size = {img_size}
    latent_dim = {latent_dim}
    batch_size = {batch_size}
    num_epochs = {num_epochs}
    G learning_rate = {g_model_learning_rate}
    D learning_rate = {d_model_learning_rate}
    beta_1 = {model_beta_1}
    amsgrad = {model_amsgrad}
    """
    )
    prepare_dirs(test_id)
    dump_hyperparams(test_id, hyperparams, global_hyperparams)
    
    g_model = build_generator(latent_dim, debug=False)
    d_model = build_discriminator()
    #print(g_model.summary())
    #raise
    
    # determine wether to load model or not
    #load_id = test_id
    #if LOAD == True:
    #    d_model.load_weights(f"./{load_id}/saved_model/d_model.h5")
    #    g_model.load_weights(f"./{load_id}/saved_model/g_model.h5")

    # Build and initialize the GAN model
    gan = GAN(d_model, g_model, latent_dim, n_critic = 2)
    
    # Prepare Optimizers
    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate = d_model_learning_rate,
        beta_1 = model_beta_1,
        amsgrad = model_amsgrad
    )
    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=g_model_learning_rate,
        beta_1 = model_beta_1,
        amsgrad = model_amsgrad
    )

    gan.compile(d_optimizer, g_optimizer)

    # Prepare Callbacks
    csv_logger = tf.keras.callbacks.CSVLogger(
        f"./{test_id}/training.log",
        append = True
    )
    
    #checkpoint_path = f"./{test_id}"+"/checkpoints/cp-{epoch:04d}.ckpt"
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #    filepath = checkpoint_path,
    #    verbose=0,
    #    save_weights_only=True,
    #    save_freq='epoch'
    #)
    
    images_path = glob("./photography/grayscale/**")
    images_dataset = tf_dataset(images_path, batch_size)
    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch}/{num_epochs}")
            gan.fit(
                images_dataset,
                epochs=1,
                callbacks=[csv_logger]
            )

            n_samples = 16
            noise = np.random.normal(size=(n_samples, latent_dim))
            examples = g_model.predict(noise)

            save_plot(test_id, examples, epoch, int(np.sqrt(n_samples)), debug = False)
        except KeyboardInterrupt:
            g_model.save(f"./{test_id}/saved_model/g_model.h5", save_format="h5")
            d_model.save(f"./{test_id}/saved_model/d_model.h5", save_format="h5")