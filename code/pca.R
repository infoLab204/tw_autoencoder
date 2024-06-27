argv <-commandArgs(trailingOnly=TRUE)

# Load MNIST test data and run PCA

# setwd("") # Set directory

# Load MNIST test data
data_type <- argv[1]
infile <- argv[2]
code <- as.integer(argv[3])

X_test <- read.csv(infile, header = F)
x_test <- as.matrix(X_test)

# Run PCA
pca <- prcomp(x_test, center = T)

mu <- (x_test - scale(x_test, scale = F))
pca_test_z <- scale(x_test, scale = F) %*% pca$rotation[, 1:code] ## pca$x[, 1:code]
pca_test_recon <- pca_test_z %*% t(pca$rotation[, 1:code]) + mu
  
loss_data <- mean((pca_test_recon - x_test)^2)

pca_test_z <- data.frame(pca_test_z)
colnames(pca_test_z) <- NULL

pca_test_recon <- data.frame(pca_test_recon)
colnames(pca_test_recon) <- NULL
  
write.csv(pca_test_z, file = paste0(data_type,"_PCA_z_", code, ".csv"), row.names = FALSE)
write.csv(pca_test_recon, file = paste0(data_type,"_PCA_out_", code,".csv"), row.names = FALSE)
write.csv(loss_data, file = paste0(data_type,"_PCA_loss_",code,".csv"), row.names = FALSE)
