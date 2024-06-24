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


data_type = sys.argv[1]  #MNIST , FMNIST , CIFAR, HOUSE

# data loading
X_data = np.loadtxt(f"{data_type}_X_data.csv", delimiter=",", dtype=None)

E = 200   # Epoch
BS = 100  # Batch size

z_code = [4,8,12,16,20]  # z_code size

autoencoder=[[] for i in range(len(z_code))]  

total_summary_loss_data = ['model_type', 'z_size', 'loss']
# Pre-train model (before common procedure)

t01 = datetime.datetime.now()


model_index=0

for z_size in z_code :

    total_summary_loss_data = ['file_index', 'code_size','train_loss']

    file_index = 2000

    if data_type="HOUSE" or data_type="CIFAR" :
        INPUT_SIZE=1024
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

    autoencoder[model_index] = Model(inputs, outputs)
    autoencoder[model_index].summary()

    # model compile
    sgd = optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

    # model trainning
    autoencoder_hist=autoencoder[model_index].fit(X_data, X_data, epochs=E, batch_size=BS, verbose=0)

    get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
    train_z = get_z([X_data])[0]


    ### LAE recon with Encoding part
    w12 = autoencoder[model_index].layers[1].get_weights()[0]
    w23 = autoencoder[model_index].layers[2].get_weights()[0]
    W13 = np.dot(w12, w23)

    b1 = autoencoder[model_index].layers[1].get_weights()[1]
    b2 = autoencoder[model_index].layers[2].get_weights()[1]

    train_enc_recon = np.dot(train_z - (np.dot(b1, w23) + b2), pinv(W13))


    ### LAE recon with Decoding part
    w34 = autoencoder[model_index].layers[3].get_weights()[1].T
    w45 = autoencoder[model_index].layers[4].get_weights()[1].T
    W35 = np.dot(w34, w45)

    b3 = autoencoder[model_index].layers[3].get_weights()[0]
    b4 = autoencoder[model_index].layers[4].get_weights()[0]

    train_dec_recon = np.dot(train_z, W35) + np.dot(b3, w45) + b4


    ### End


    file_index = file_index + 1

    # Common  part

    B_matrix = []  ## ReLU condition fulfill stored
    
    for i in range(0, X_data.shape[0], 1) :
        vec = np.dot(X_data[i], autoencoder[model_index].layers[1].get_weights()[0]) + autoencoder[model_index].layers[1].get_weights()[1]   

        B = []
    
        for j in range(0, len(vec)) :    
            if(vec[j] >= 0.0) :
                B.extend([j])

        B_list = np.array([0] * (HIDDEN_SIZE1))
        
        B_list[B] = 1        

        B_matrix.append(B_list)

    B_matrix = np.array(B_matrix)

    B_matrix_col_sum = B_matrix.sum(axis = 0)

    get_3rd_layer_output = K.function([autoencoder[model_index].layers[0].input], [autoencoder[model_index].layers[2].output])
    layer_z = get_3rd_layer_output([[X_data]])[0]

    loss_list = ['step_value', 'loss_value']

    step_value_list = list(range(50, X_data.shape[0]+1, 50))

    for step_value in step_value_list :

        C1 = []
    
        for node_value in range(0, len(B_matrix_col_sum)) :
            if(B_matrix_col_sum[node_value] >= step_value) :
                C1.extend([node_value])
    
        new_w12 = autoencoder[model_index].layers[1].get_weights()[0]
        new_w12 = new_w12[:, C1] 
        new_w23 = autoencoder[model_index].layers[2].get_weights()[0]
        new_w23 = new_w23[C1, :]
        new_W13 = np.dot(new_w12, new_w23)
    
        new_b1 = autoencoder[model_index].layers[1].get_weights()[1]
        new_b1 = new_b1[C1]
        new_b2 = autoencoder[model_index].layers[2].get_weights()[1]

        new_z = np.dot(X_data, new_W13) + np.dot(new_b1, new_w23) + new_b2
    
        loss_value = np.sum((layer_z - new_z)**2) / (layer_z.shape[0] * layer_z.shape[1])

        loss_list = np.vstack((loss_list, [step_value, loss_value]))

    loss_list = np.delete(loss_list, 0, axis=0)
    
    selected_value = loss_list[loss_list[:, 1] == min(loss_list[:, 1]), 0]
 

    if(len(selected_value) > 1) :
        selected_value = max(selected_value)


    C1 = []

    for node_value in range(0, len(B_matrix_col_sum)) :
        if(float(B_matrix_col_sum[node_value]) >= float(selected_value)) :
            C1.extend([node_value])


    ### Set number of node

    before_w12 = autoencoder[model_index].layers[1].get_weights()
    before_w12[0] = before_w12[0][:,C1]
    before_w12[1] = before_w12[1][C1]

    before_w23 = autoencoder[model_index].layers[2].get_weights()
    before_w23[0] = before_w23[0][C1,:]

    before_w34 = autoencoder[model_index].layers[3].get_weights()
    before_w34[0] = before_w34[0][C1]
    before_w34[1] = before_w34[1][C1,:] 

    before_w45 = autoencoder[model_index].layers[4].get_weights()
    before_w45[2] = before_w45[2][C1]
    before_w45[1] = before_w45[1][:,C1]


    ### Train after common procedure

    if data_type="HOUSE" or data_type="CIFAR" :
        INPUT_SIZE=1024
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

    autoencoder[model_index] = Model(inputs, outputs)
    autoencoder[model_index].summary()

    autoencoder[model_index].layers[1].set_weights(before_w12)
    autoencoder[model_index].layers[2].set_weights(before_w23)
    autoencoder[model_index].layers[3].set_weights(before_w34)
    autoencoder[model_index].layers[4].set_weights(before_w45)
 

    sgd = optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

    ### Train after common procedure
    autoencoder_hist=autoencoder[model_index].fit(X_data, X_data, epochs=E, batch_size=BS, verbose=0)


    common_train=autoencoder[model_index].evaluate(X_data, X_data, batch_size=BS)

    print("common loss",common_train)
    
    ### Calculate loss and save output
    get_layer_output = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[4].output])
    train_output = get_layer_output([X_data])[0]

    get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
    train_z = get_z([X_data])[0]

    train_loss = np.sum((X_data - train_output)**2) / (X_data.shape[0] * X_data.shape[1])

    np.savetxt(f"{data_type}_proposed_recon_{z_size}.csv", train_output, fmt='%.6f', delimiter=",")
    np.savetxt(f"{data_type}_proposed_z_{z_size}.csv", train_z, fmt='%.6f', delimiter=",")

    summary_loss_data = [file_index, z_size, train_loss]
    total_summary_loss_data = np.vstack((total_summary_loss_data, summary_loss_data))
    print(total_summary_loss_data)
    np.savetxt(f"{data_type}_proposed_total_summary_loss_data.csv", total_summary_loss_data, fmt='%s', delimiter=",")

    ### LAE recon with Encoding part

    w12 = autoencoder[model_index].layers[1].get_weights()[0]
    w23 = autoencoder[model_index].layers[2].get_weights()[0]
    W13 = np.dot(w12, w23)

    b1 = autoencoder[model_index].layers[1].get_weights()[1]
    b2 = autoencoder[model_index].layers[2].get_weights()[1]

    train_enc_recon = np.dot(train_z - (np.dot(b1, w23) + b2), pinv(W13))

    np.savetxt(f"{data_type}_proposed_enc_recon_{z_size}.csv", train_enc_recon, fmt='%.6f', delimiter=",")

    ### LAE recon with Decoding part

    w34 = autoencoder[model_index].layers[3].get_weights()[1].T
    w45 = autoencoder[model_index].layers[4].get_weights()[1].T
    W35 = np.dot(w34, w45)

    b3 = autoencoder[model_index].layers[3].get_weights()[0]
    b4 = autoencoder[model_index].layers[4].get_weights()[0]

    train_dec_recon = np.dot(train_z, W35) + np.dot(b3, w45) + b4

    np.savetxt(f"{data_type}_proposed_dec_recon_{z_size}.csv", train_dec_recon, fmt='%.6f', delimiter=",")

    ### End

t02 = datetime.datetime.now()
print("run time :", t02 - t01)

model_index=model_index+1

## End of tied_weight_proposed.py
