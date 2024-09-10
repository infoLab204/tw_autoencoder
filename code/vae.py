import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error
import sys
import numpy as np


data_type=sys.argv[1]
infile=sys.argv[2]
latent_dim = int(sys.argv[3])

# First, here's our encoder network, mapping inputs to our latent distribution parameters:
if data_type=="MNIST" or data_type=="FMNIST":
    original_dim = 28 * 28
    intermediate_dim = 128
    E=200
    BS=100
elif data_type=="CIFAR" or data_type=="SVHN" :
    original_dim = 32 * 32
    intermediate_dim = 256
    E=200
    BS=100
elif data_type="BC":  ## Breast Cancer 
    original_dim = 30
    intermediate_dim = 100
    E=100
    BS=20
elif data_type="WINE":
    original_dim = 13
    intermediate_dim = 100
    E=100
    BS=20
else :
    print("data type error")
 
total_mse=open(f"{data_type}_VAE_{latent_dim}_mse.txt","w")

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)


#We can use these parameters to sample new similar points from the latent space:
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling,output_shape=(latent_dim,))([z_mean, z_log_sigma])

#Finally, we can map these sampled latent points back to reconstructed inputs:
# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

reconstruction_loss =  keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

## data load
x_train = np.loadtxt(f"{infile}", delimiter=",", dtype=None)

#vae.fit(x_train, x_train, epochs=2, batch_size=32, validation_data=(x_test, x_test))
vae.fit(x_train, x_train, epochs=E, batch_size=BS)

encoded_imgs = encoder.predict(x_train)[2]
np.savetxt(f"{data_type}_VAE_code_{latent_dim}.csv",encoded_imgs, fmt='%.10f', delimiter=",")
decoded_imgs = decoder.predict(encoded_imgs)
np.savetxt(f"{data_type}_VAE_out_{latent_dim}.csv",decoded_imgs, fmt='%.10f', delimiter=",")

total_mse.write(f"{latent_dim}\t{mean_squared_error(x_train, decoded_imgs)}")
total_mse.close()
