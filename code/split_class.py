# Load library
import sys
import numpy as np

data_type = sys.argv[1] ## data_type : MNIST, FMNIST, HOUSE, CIFAR

# Load data

X_data = np.loadtxt(data_type + "_X_data.csv", delimiter=',', dtype=None)
Y_data = np.loadtxt(data_type + "_Y_label.csv", delimiter=',', dtype=None)

z_list = [4, 8, 12, 16, 20]


LAE_recon = [[]] * len(z_list)
ICA_recon = [[]] * len(z_list)
SBAE_recon = [[]] * len(z_list)
PCA_recon = [[]] * len(z_list)

for z_index in range(len(z_list)) :
    
    z_size = z_list[z_index]

    LAE_recon[z_index] = np.loadtxt(data_type +  "_proposed_recon_" + str(z_size) + ".csv", delimiter=',', dtype=None)
    ICA_recon[z_index] = np.loadtxt(data_type + "_ICA_recon_" + str(z_size) + ".csv", delimiter=',', dtype=None)
    SBAE_recon[z_index] = np.loadtxt(data_type + "_Stacked_recon_" + str(z_size) + ".csv", delimiter=',', dtype=None)
    PCA_recon[z_index] = np.loadtxt(data_type + "_PCA_recon_" + str(z_size) + ".csv", delimiter=',', dtype=None)

# Split each class

class_num = 10

LT = [[]] * class_num

for k in range(10) :
    LT[k] = []
            
for i in range(len(Y_data)) :
    for j in range(len(LT)) :
        if Y_data[i] == j :
            LT[j] = np.append(LT[j], i)
            LT[j] = [int(LT[j]) for LT[j] in LT[j]]

X_data_class = [[]] * class_num

LAE_recon_class = [[]] * len(z_list)
ICA_recon_class = [[]] * len(z_list)
SBAE_recon_class = [[]] * len(z_list)
PCA_recon_class = [[]] * len(z_list)

for z_index in range(len(z_list)) :
    
    LAE_recon_class[z_index] = [[]] * class_num
    ICA_recon_class[z_index] = [[]] * class_num
    SBAE_recon_class[z_index] = [[]] * class_num
    PCA_recon_class[z_index] = [[]] * class_num

for class_index in range(class_num) :
    
    X_data_class[class_index] = X_data[LT[class_index]]
    
    for z_index in range(len(z_list)) :
    
        LAE_recon_class[z_index][class_index] = LAE_recon[z_index][LT[class_index]]
        ICA_recon_class[z_index][class_index] = ICA_recon[z_index][LT[class_index]]
        SBAE_recon_class[z_index][class_index] = SBAE_recon[z_index][LT[class_index]]
        PCA_recon_class[z_index][class_index] = PCA_recon[z_index][LT[class_index]]

# Save data

for class_index in range(10) :
    
    np.savetxt(data_type + "_X_data_"+str(z_size)+"_class" +str(class_index) + ".csv", X_data_class[class_index], delimiter=',')
    
    for z_index in range(len(z_list)) :
        
        z_size = z_list[z_index]
    
        np.savetxt(data_type + "_proposed_recon_" + str(z_size) + "_class" + str(class_index) + ".csv", LAE_recon_class[z_index][class_index], delimiter=',')
        np.savetxt(data_type + "_ICA_recon_" + str(z_size) + "_class" + str(class_index) + ".csv", ICA_recon_class[z_index][class_index], delimiter=',')
        np.savetxt(data_type + "_Stacked_recon_" + str(z_size) + "_class" + str(class_index) + ".csv", SBAE_recon_class[z_index][class_index], delimiter=',') 
        np.savetxt(data_type + "_PCA_recon_" + str(z_size) + "_class" + str(class_index) + ".csv", PCA_recon_class[z_index][class_index], delimiter=',')    

    print(str(class_index) + " finish!")

## End of split_class.py
