argv <-commandArgs(trailingOnly=TRUE)
# Load MNIST test data and run PCA

# setwd("") # Set directory

# Load MNIST test data

library(fastICA)
data_type <- argv[1]
infile <- argv[2]
code <- as.integer(argv[3])

X_test <- read.csv(infile, header = F)
x_test <- as.matrix(X_test)

mu <- (x_test - scale(x_test, scale = F))

ica_df <- fastICA(x_test, code)
  
ica_test_z <- ica_df$S[,1:code]
  
ica_test_recon <- ica_df$S[,1:code] %*% (ica_df$A[1:code,]) + mu
  
loss_data <- mean((ica_test_recon - x_test)^2)
  
ica_test_z <- data.frame(ica_test_z)
colnames(ica_test_z) <- NULL
  
ica_test_recon <- data.frame(ica_test_recon)
colnames(ica_test_recon) <- NULL
  
write.csv(ica_test_z, file = paste0(data_type,"_ICA_z_", code,".csv"), row.names = FALSE)
write.csv(ica_test_recon, file = paste0(data_type,"_ICA_out_", code,".csv"), row.names = FALSE)
write.csv(loss_data, file=paste0(data_type,"_ICA_loss_",code,".csv"), row.names=FALSE)
