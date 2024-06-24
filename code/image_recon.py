import sys
import numpy as np

from matplotlib import pyplot

data_type =sys.argv[1]  ## data_type : MNIST, FMNIST, HOUSE, CIFAR
fidx=sys.argv[2]  ## CV number

# Load data
X_test = np.loadtxt(f"{data_type}_X_data.csv", delimiter=',', dtype=None)
Y_test = np.loadtxt(f"{data_type}_Y_label.csv", delimiter=',', dtype=None)

#z_list = [4, 8, 12, 16,20]
z_list = [4]

LAE_test_recon = [[]] * len(z_list)
ICA_test_output = [[]] * len(z_list)
SBAE_test_output = [[]] * len(z_list)
PCA_test_recon = [[]] * len(z_list)

for z_index in range(len(z_list)) :
    
    z_size = z_list[z_index]

    LAE_test_recon[z_index] = np.loadtxt(f"{data_type}_proposed_recon_{z_size}.csv", delimiter=',', dtype=None)
    ICA_test_output[z_index] = np.loadtxt(f"{data_type}_ICA_recon_{z_size}.csv", delimiter=',', dtype=None)
    SBAE_test_output[z_index] = np.loadtxt(f"{data_type}_Stacked_recon_{z_size}.csv", delimiter=',', dtype=None)
    PCA_test_recon[z_index] = np.loadtxt(f"{data_type}_PCA_recon_{z_size}.csv", delimiter=',', dtype=None)


# Split each class

class_num = 10

LT = [[]] * class_num

for k in range(10) :
    LT[k] = []
            
for i in range(len(Y_test)) :
    for j in range(len(LT)) :
        if Y_test[i] == j :
            LT[j] = np.append(LT[j], i)
            LT[j] = [int(LT[j]) for LT[j] in LT[j]]

# Image select in each class

idx=10  ## select image number
selected = [LT[0][idx], LT[1][idx], LT[2][idx], LT[3][idx], LT[4][idx], LT[5][idx], LT[6][idx], LT[7][idx], LT[8][idx], LT[9][idx]]

X_test_selected = list(X_test[selected])

LAE_test_recon_selected = [[]] * len(z_list)
SBAE_test_output_selected = [[]] * len(z_list)
ICA_test_output_selected = [[]] * len(z_list)
PCA_test_recon_selected = [[]] * len(z_list)

for z_index in range(len(z_list)) :
  
    LAE_test_recon_selected[z_index] = list(LAE_test_recon[z_index][selected])
    SBAE_test_output_selected[z_index] = list(SBAE_test_output[z_index][selected])
    ICA_test_output_selected[z_index] = list(ICA_test_output[z_index][selected])
    PCA_test_recon_selected[z_index] = list(PCA_test_recon[z_index][selected])

if data_type="HOUSE" or daty_type="CIFAR" :
    fsize=32
else :
    fsize=28

for z_size in z_list :

    print("z size : " + str(z_size))

    n = 10
    pyplot.figure(figsize=(10, 5))
    pyplot.subplots_adjust(top=1.15, bottom=0, right=1, left=0, hspace=0, wspace=0)

    for i in range(n) :
    
        ax = pyplot.subplot(5, n, i + 1)

        pyplot.imshow(X_test_selected[i].reshape(fsize, fsize),cmap=pyplot.get_cmap('gray'))  ## original image
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = pyplot.subplot(5, n, i + 1 + n)
        pyplot.imshow(LAE_test_recon_selected[z_index][i].reshape(fsize, fsize),cmap=pyplot.get_cmap('gray')) ## LAB autoencoder image reconstruction
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = pyplot.subplot(5, n, i + 1 + 2 * n)
        pyplot.imshow(SBAE_test_output_selected[z_index][i].reshape(fsize, fsize),cmap=pyplot.get_cmap('gray')) ## stacked autoencoder image reconstruction
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = pyplot.subplot(5, n, i + 1 + 3 * n)
        pyplot.imshow(ICA_test_output_selected[z_index][i].reshape(fsize, fsize),cmap=pyplot.get_cmap('gray'))  ## ICA image reconstruction
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = pyplot.subplot(5, n, i + 1 + 4 * n)
        pyplot.imshow(PCA_test_recon_selected[z_index][i].reshape(fsize, fsize),cmap=pyplot.get_cmap('gray')) ## PCA image reconstruction
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    

    pyplot.savefig(data_type + "_result_" + str(z_size) + "_image_CV"+fidx+".png", bbox_inches='tight', dpi=500)

## End of image_recon.py
