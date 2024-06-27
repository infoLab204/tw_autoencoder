import sys
import numpy as np

from matplotlib import pyplot
# Choice MNIST or Fashion-MNIST

data_type =sys.argv[1]
infileX=sys.argv[2]
infileY=sys.argv[3]
code=sys.argv[4]
img_idx=int(sys.argv[5])

# Load data
X_data = np.loadtxt(f"{infileX}", delimiter=',', dtype=None)
Y_label = np.loadtxt(f"{infileY}", delimiter=',', dtype=None)

LAE_recon = np.loadtxt(f"{data_type}_LAE_out_{code}.csv", delimiter=',', dtype=None)
SAE_recon = np.loadtxt(f"{data_type}_SAE_out_{code}.csv", delimiter=',', dtype=None)
PCA_recon = np.loadtxt(f"{data_type}_PCA_out_{code}.csv", delimiter=',', dtype=None)
ICA_recon = np.loadtxt(f"{data_type}_ICA_out_{code}.csv", delimiter=',', dtype=None)


# Split each class

class_num = 11

LT = [[]] * class_num

for k in range(11) :
    LT[k] = []
            
for i in range(len(Y_label)) :
    for j in range(len(LT)) :
        if Y_label[i] == j :
            LT[j] = np.append(LT[j], i)
            LT[j] = [int(LT[j]) for LT[j] in LT[j]]

# Image select in each class

if data_type=="SVHN" :
    selected = [LT[1][img_idx], LT[2][img_idx], LT[3][img_idx], LT[4][img_idx], LT[5][img_idx], LT[6][img_idx], LT[7][img_idx], LT[8][img_idx], LT[9][img_idx], LT[10][img_idx]]
else :
    selected = [LT[0][img_idx], LT[1][img_idx], LT[2][img_idx], LT[3][img_idx], LT[4][img_idx], LT[5][img_idx], LT[6][img_idx], LT[7][img_idx], LT[8][img_idx], LT[9][img_idx]]

X_data_selected = list(X_data[selected])

  
LAE_recon_selected = list(LAE_recon[selected])
SAE_recon_selected = list(SAE_recon[selected])
PCA_recon_selected = list(PCA_recon[selected])
ICA_recon_selected = list(ICA_recon[selected])


n = 10
pyplot.figure(figsize=(10, 5))
pyplot.subplots_adjust(top=1.15, bottom=0, right=1, left=0, hspace=0, wspace=0)

for i in range(n) :
    
    ax = pyplot.subplot(5, n, i + 1)

    pyplot.imshow(X_data_selected[i].reshape(28, 28),cmap=pyplot.get_cmap('gray'))
    #pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = pyplot.subplot(5, n, i + 1 + n)
    pyplot.imshow(LAE_recon_selected[i].reshape(28, 28),cmap=pyplot.get_cmap('gray'))
    #pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    ax = pyplot.subplot(5, n, i + 1 + 2 * n)
    pyplot.imshow(SAE_recon_selected[i].reshape(28, 28),cmap=pyplot.get_cmap('gray'))
    #pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    ax = pyplot.subplot(5, n, i + 1 + 3 * n)
    pyplot.imshow(PCA_recon_selected[i].reshape(28, 28),cmap=pyplot.get_cmap('gray'))
    #pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = pyplot.subplot(5, n, i + 1 + 4 * n)
    pyplot.imshow(ICA_recon_selected[i].reshape(28, 28),cmap=pyplot.get_cmap('gray'))
    #pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

# pyplot.savefig(data_type + "_result_" + str(z_size) + "_image.eps", bbox_inches='tight', dpi=100)
pyplot.savefig(data_type + "_image_"+code+"_img"+str(img_idx)+".png", bbox_inches='tight', dpi=500)
