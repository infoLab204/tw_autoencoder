# Load library
import sys
import numpy as np

data_type = sys.argv[1]
infileX = sys.argv[2]
infileY = sys.argv[3]
code = sys.argv[4]

# Load data
X_data = np.loadtxt(infileX, delimiter=',', dtype=None)
Y_label = np.loadtxt(infileY, delimiter=',', dtype=None)

LAE_recon = np.loadtxt(f"{data_type}_LAE_out_{code}.csv", delimiter=',', dtype=None)
SAE_recon = np.loadtxt(f"{data_type}_SAE_out_{code}.csv", delimiter=',', dtype=None)
PCA_recon = np.loadtxt(f"{data_type}_PCA_out_{code}.csv", delimiter=',', dtype=None)
ICA_recon = np.loadtxt(f"{data_type}_ICA_out_{code}.csv", delimiter=',', dtype=None)

# Split each class
if data_type=="SVHN" :
    class_num = 11
else : 
    class_num = 10
    
LT = [[]] * class_num

for k in range(class_num) :
    LT[k] = []
            
for i in range(len(Y_label)) :
    for j in range(len(LT)) :
        if Y_label[i] == j :
            LT[j] = np.append(LT[j], i)
            LT[j] = [int(LT[j]) for LT[j] in LT[j]]

X_data_class = [[]] * class_num
LAE_recon_class = [[]] * class_num
SAE_recon_class = [[]] * class_num
PCA_recon_class = [[]] * class_num
ICA_recon_class = [[]] * class_num

for class_index in range(class_num) :
    X_data_class[class_index] = X_data[LT[class_index]]
    LAE_recon_class[class_index] = LAE_recon[LT[class_index]]
    SAE_recon_class[class_index] = SAE_recon[LT[class_index]]
    PCA_recon_class[class_index] = PCA_recon[LT[class_index]]
    ICA_recon_class[class_index] = ICA_recon[LT[class_index]]

# Save data
for class_index in range(class_num) :
    np.savetxt(data_type + "_X_data_class_" + str(class_index) + ".csv", X_data_class[class_index], delimiter=',')
    
    np.savetxt(data_type + "_LAE_recon_" + code + "_class" + str(class_index) + ".csv", LAE_recon_class[class_index], delimiter=',')
    np.savetxt(data_type + "_SAE_recon_" + code + "_class" + str(class_index) + ".csv", SAE_recon_class[class_index], delimiter=',') 
    np.savetxt(data_type + "_ICA_recon_" + code + "_class" + str(class_index) + ".csv", ICA_recon_class[class_index], delimiter=',')
    np.savetxt(data_type + "_PCA_recon_" + code + "_class" + str(class_index) + ".csv", PCA_recon_class[class_index], delimiter=',')    

