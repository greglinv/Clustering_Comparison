from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
import numpy as np


class VAE:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.vae = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.input_dim,))
        h = Dense(256, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=0.1)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        encoder = Model(inputs, z_mean)

        latent_inputs = Input(shape=(self.latent_dim,))
        h_decoded = Dense(256, activation='relu')(latent_inputs)
        outputs = Dense(self.input_dim, activation='sigmoid')(h_decoded)

        decoder = Model(latent_inputs, outputs)

        vae = Model(inputs, decoder(encoder(inputs)))
        vae.compile(optimizer='rmsprop', loss='binary_crossentropy')

        return encoder, decoder, vae

    def train(self, x_train, epochs=50, batch_size=128):
        self.vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_train, x_train))

if __name__ == "__main__":
    x_train = np.random.rand(1000, 64)  # Example data
    vae = VAE(input_dim=64, latent_dim=10)
    vae.train(x_train)
