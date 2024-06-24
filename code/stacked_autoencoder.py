# Load library

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

import datetime
import sys,os

# Load  data
data_type=sys.argv[1]  ## data_type : MNIST, FMNIST, CIFAR, HOUSE

X_data = np.loadtxt(f"{data_type}_X_data.csv", delimiter=",", dtype=None)


# z_list : define experiment code(Z) size
z_list = [4,8,12,16,20]

autoencoder = [[] for i in range(len(z_list))]


# E : epoch, BS = batch size

E = 200
BS = 100


# Train model and save data(code(Z), output and total loss data)

model_index = 0

t01 = datetime.datetime.now()

total_summary_loss_data = ['model_type', 'z_size', 'train_loss']

# Define first pre-training(784 -> 400) model

if data_type=="HOUSE" or data_type="CIFAR" :
    INPUT_SIZE = 1024
else :
    INPUT_SIZE = 784

HIDDEN_SIZE = 400

w_initializer = initializers.glorot_uniform(seed=None)
b_initializer = initializers.glorot_uniform(seed=None)

dense1 = Input(shape=(INPUT_SIZE,))
dense2 = Dense(HIDDEN_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
dense3 = Dense(INPUT_SIZE, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

autoencoder[model_index] = Model(dense1, dense3)
/
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adam)

autoencoder[model_index].fit(X_data, X_data, epochs=E, batch_size=BS, verbose=0)

pre_train_w1 = autoencoder[model_index].layers[1].get_weights()
pre_train_w2 = autoencoder[model_index].layers[2].get_weights()

get_pre_train_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[1].output])
X_data2 = get_pre_train_z([X_data])[0]

for z_size in z_list : 
    
    t11 = datetime.datetime.now()
    
    # Define second pre-training(400 -> z_size) models
    
    INPUT_SIZE = 400
    HIDDEN_SIZE = z_size
    
    dense1 = Input(shape=(INPUT_SIZE,))
    dense2 = Dense(HIDDEN_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
    dense3 = Dense(INPUT_SIZE, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

    autoencoder[model_index] = Model(dense1, dense3)

    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adam)

    autoencoder[model_index].fit(X_data2, X_data2, epochs=E, batch_size=BS, verbose=0)
    
    pre_train_w3 = autoencoder[model_index].layers[1].get_weights()
    pre_train_w4 = autoencoder[model_index].layers[2].get_weights()

    # Define stacked models 

    if data_type=="HOUSE" or data_type="CIFAR" :
        INPUT_SIZE = 1024
    else :
        INPUT_SIZE = 784

    HIDDEN_SIZE1 = 400
    HIDDEN_SIZE2 = z_size
    
    dense1 = Input(shape=(INPUT_SIZE,))
    dense2 = Dense(HIDDEN_SIZE1, activation='relu')(dense1)
    dense3 = Dense(HIDDEN_SIZE2, activation='linear')(dense2)
    dense4 = Dense(HIDDEN_SIZE1, activation='relu')(dense3)
    dense5 = Dense(INPUT_SIZE, activation='linear')(dense4)

    autoencoder[model_index] = Model(dense1, dense5)

    autoencoder[model_index].layers[1].set_weights(pre_train_w1)
    autoencoder[model_index].layers[2].set_weights(pre_train_w3)
    autoencoder[model_index].layers[3].set_weights(pre_train_w4)
    autoencoder[model_index].layers[4].set_weights(pre_train_w2)

    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adam)

    autoencoder[model_index].fit(X_data, X_data, epochs=E, batch_size=BS, verbose=0)
    
    # Get output and calculate loss
    get_output = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[4].output])
    X_output = get_output([X_data])[0]
    
    train_loss = np.sum((X_data - X_output)**2) / (X_data.shape[0] * X_data.shape[1])

    summary_loss_data = ['Stacked', z_size, train_loss]

    total_summary_loss_data = np.vstack((total_summary_loss_data, summary_loss_data))

    np.savetxt(f"{data_type}_Stacked_total_loss_data.csv", total_summary_loss_data, delimiter=',', fmt='%s')
    np.savetxt(f"{data_type}_Stacked_recon_{z_size}.csv", X_output, delimiter=',')    
    
    # Get code(Z)
    
    get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
    train_z = get_z([X_data])[0]
    
    np.savetxt(f"{data_type}_Stacked_z_{z_size}.csv", train_z, delimiter=',')    
    
    # Calculate run time
    
    t12 = datetime.datetime.now()
    
    print(str(z_size) + " finish! run time " + str(t12 - t11))
    
    model_index = model_index + 1

t02 = datetime.datetime.now()

print("total run time " + str(t02 - t01))

# Print total loss
print(total_summary_loss_data)


## End of stacked_autoencoder 
