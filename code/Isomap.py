from tensorflow.keras.datasets import mnist
from sklearn.manifold import Isomap
import pandas as pd
import numpy as np
import sys

data_type=sys.argv[1]  ## data type
infile=sys.argv[2]   ## input file
z_size=int(sys.argv[3]  ## components size
n_neigh=int(sys.argv[4])  ## neighbors size


train_x=np.loadtxt(f"{infile}", delimiter=",")

isomap=Isomap(n_components=z_size,n_neighbors=n_neigh)

X_reduced=isomap.fit_transform(train_x)
np.savetxt(f"{data_type}_isomap_code{z_size}_neighbor{n_neigh}.csv",X_reduced, delimiter=",", fmt='%f')



