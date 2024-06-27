import numpy as np


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

from matplotlib import pyplot

import datetime

from numpy.linalg import pinv
import sys,os

## tied weight class
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
    
    def get_config(self) :
        config=super().get_config().copy()
   
        return config


data_type = sys.argv[1]  #MNIST, FMNIST ,CIFAR or SVHN
infile = sys.argv[2] ## input data
z_size = sys.argv[3]  ## code size

# data loading
X_train = np.loadtxt(f"{infile}", delimiter=",", dtype=None)

E = 200   # Epoch
BS = 100  # Batch size

# Pre-train model (before common procedure)

t01 = datetime.datetime.now()


file_index = 2000

if data_type=="CIFAR" or data_type=="SVHN" :
    INPUT_SIZE = 1024
else :
    INPUT_SIZE = 784
HIDDEN_SIZE1 = 2000
HIDDEN_SIZE2 = z_size

w_initializer = initializers.Orthogonal(gain=1.0, seed=None)
b_initializer = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)

# model setting : 784 - 2000 - code_size - 2000 - 784
inputs = Input(shape=(INPUT_SIZE,))
dense_1 = Dense(HIDDEN_SIZE1, activation='relu',kernel_initializer=w_initializer, bias_initializer=b_initializer)
dense_2 = Dense(HIDDEN_SIZE2, activation='linear',kernel_initializer=w_initializer, bias_initializer=b_initializer)

x1 = dense_1(inputs)
x2 = dense_2(x1)
x3 = DenseTranspose(dense_2, activation='relu')(x2)
outputs = DenseTranspose(dense_1, activation='linear')(x3)

autoencoder = Model(inputs, outputs)
autoencoder.summary()

# model compile
sgd = optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(loss='mean_squared_error', optimizer=sgd)

# model trainning
autoencoder_hist=autoencoder.fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)

loss_train=autoencoder.evaluate(X_train, X_train, batch_size=BS)

get_z = K.function([autoencoder.layers[0].input],[autoencoder.layers[2].output])
train_z = get_z([X_train])[0]

# Common 

B_matrix = []  ## ReLU condition fulfill stored
    
for i in range(0, X_train.shape[0], 1) :
    vec = np.dot(X_train[i], autoencoder.layers[1].get_weights()[0]) + autoencoder.layers[1].get_weights()[1]   

    B = []
    
    for j in range(0, len(vec)) :    
        if(vec[j] >= 0.0) :
            B.extend([j])

    B_list = np.array([0] * (HIDDEN_SIZE1))
        
    B_list[B] = 1        

    B_matrix.append(B_list)

B_matrix = np.array(B_matrix)
#np.savetxt(f"{data_type}_{z_size}_B_matrix.csv", B_matrix, fmt='%d', delimiter=",")

B_matrix_col_sum = B_matrix.sum(axis = 0)
np.savetxt(f"{data_type}_{z_size}_unit_node.csv", B_matrix_col_sum,fmt='%d',  delimiter=",")

get_3rd_layer_output = K.function([autoencoder.layers[0].input], [autoencoder.layers[2].output])
layer_z = get_3rd_layer_output([[X_train]])[0]

loss_list = ['step_value', 'loss_value']

step_value_list = list(range(0, X_train.shape[0]+1, 10))

outfile_name=f"{data_type}_{z_size}_theta.csv"
outfile=open(outfile_name,"w")

for step_value in step_value_list :

    C1 = []
    
    for node_value in range(0, len(B_matrix_col_sum)) :
        if(B_matrix_col_sum[node_value] >= step_value) :
            C1.extend([node_value])
    
    new_w12 = autoencoder.layers[1].get_weights()[0]
    new_w12 = new_w12[:, C1] 
    new_w23 = autoencoder.layers[2].get_weights()[0]
    new_w23 = new_w23[C1, :]
    new_W13 = np.dot(new_w12, new_w23)
    
    new_b1 = autoencoder.layers[1].get_weights()[1]
    new_b1 = new_b1[C1]
    new_b2 = autoencoder.layers[2].get_weights()[1]

    new_z = np.dot(X_train, new_W13) + np.dot(new_b1, new_w23) + new_b2
    
    loss_value = np.sum((layer_z - new_z)**2) / (layer_z.shape[0] * layer_z.shape[1])

    outfile.write(str(loss_value))
    outfile.write("\n")

    loss_list = np.vstack((loss_list, [step_value, loss_value]))

outfile.close()

loss_list = np.delete(loss_list, 0, axis=0)
    
selected_value = loss_list[loss_list[:, 1] == min(loss_list[:, 1]), 0]
 

if(len(selected_value) > 1) :
    selected_value = max(selected_value)


C1 = []

for node_value in range(0, len(B_matrix_col_sum)) :
    if(float(B_matrix_col_sum[node_value]) >= float(selected_value)) :
        C1.extend([node_value])


### Set number of node

before_w12 = autoencoder.layers[1].get_weights()
before_w12[0] = before_w12[0][:,C1]
before_w12[1] = before_w12[1][C1]

before_w23 = autoencoder.layers[2].get_weights()
before_w23[0] = before_w23[0][C1,:]

before_w34 = autoencoder.layers[3].get_weights()
before_w34[0] = before_w34[0][C1]
before_w34[1] = before_w34[1][C1,:] 

before_w45 = autoencoder.layers[4].get_weights()
before_w45[2] = before_w45[2][C1]
before_w45[1] = before_w45[1][:,C1]

### Train after common procedure

if data_type=="CIFAR" or data_type=="SVHN" :
    INPUT_SIZE = 1024
else :
    INPUT_SIZE = 784

HIDDEN_SIZE1 = len(C1)
HIDDEN_SIZE2 = z_size

inputs = Input(shape=(INPUT_SIZE,))
dense_1 = Dense(HIDDEN_SIZE1, activation='relu')
dense_2 = Dense(HIDDEN_SIZE2, activation='linear')

x1=dense_1(inputs)
x2=dense_2(x1)
x3 = DenseTranspose(dense_2, activation='relu')(x2)
outputs = DenseTranspose(dense_1, activation='linear')(x3)

autoencoder = Model(inputs, outputs)
autoencoder.summary()

autoencoder.layers[1].set_weights(before_w12)
autoencoder.layers[2].set_weights(before_w23)
autoencoder.layers[3].set_weights(before_w34)
autoencoder.layers[4].set_weights(before_w45)
 

sgd = optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(loss='mean_squared_error', optimizer=sgd)

autoencoder_hist=autoencoder.fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)

common_train=autoencoder.evaluate(X_train, X_train, batch_size=BS)

### save code and output layer

get_layer_output = K.function([autoencoder.layers[0].input],[autoencoder.layers[4].output])
train_output = get_layer_output([X_train])[0]

get_z = K.function([autoencoder.layers[0].input],[autoencoder.layers[2].output])
temp_z = get_z([X_train])[0]

np.savetxt(f"{data_type}_LAE_out_{z_size}.csv", train_output, fmt='%.6f', delimiter=",")
np.savetxt(f"{data_type}_LAE_z_{z_size}.csv", temp_z, fmt='%.6f', delimiter=",")

### LAE recon
w34 = autoencoder.layers[3].get_weights()[1].T
w45 = autoencoder.layers[4].get_weights()[1].T
W35 = np.dot(w34, w45)

b3 = autoencoder.layers[3].get_weights()[0]
b4 = autoencoder.layers[4].get_weights()[0]

temp_recon = np.dot(temp_z, W35) + np.dot(b3, w45) + b4

np.savetxt(f"{data_type}_LAE_reout_{z_size}.csv", temp_recon, fmt='%.6f', delimiter=",")

### End

t02 = datetime.datetime.now()
print("run time :", t02 - t01)
