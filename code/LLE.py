from sklearn.manifold import LocallyLinearEmbedding

import pandas as pd
import numpy as np
import sys

data_type=sys.argv[1] ## data type
infile=sys.argv[2]   ## input file
z_size=sys.argv[3]
n_neigh=sys.argv[4]

train_x=np.loadtxt(f"{infile}", delimiter=",")

lle=LocallyLinearEmbedding(n_components=z_size, n_neighbors=n_neigh)
X_reduced=lle.fit_transform(train_x)
np.savetxt(f"{data_type}_LLE_code{z_size}_neighbor{n_neigh}.csv",X_reduced, delimiter=",", fmt='%.8f')
