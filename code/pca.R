argv <-commandArgs(trailingOnly=TRUE)

data_type=argv[1] ## data_type : MNIST, FMNIST, HOUSE, CIFAR

# Load  data
X_data <- read.csv(paste0(data_type,"_X_data.csv"), header = F)
x_data <- as.matrix(X_data)

# Run PCA and save result(reconstruct and code(Z, principal component))

# Define code size list
z_list <- c(4,8,12,16,20) 

pca <- prcomp(x_data, center = T)

loss_data <- data.frame(z_size = z_list, loss = rep(NA, length(z_list)))

mu <- (x_data - scale(x_data, scale = F))

loop_index <- 1

for(comp_value in z_list){ 
  
  pca_data_z <- scale(x_data, scale = F) %*% pca$rotation[, 1:comp_value] ## pca$x[, 1:comp_value]
  
  pca_data_recon <- pca_data_z %*% t(pca$rotation[, 1:comp_value]) + mu
  
  loss_data$loss[loop_index] <- mean((pca_data_recon - x_data)^2)
  print(loss_data)

  pca_data_z <- data.frame(pca_data_z)
  colnames(pca_data_z) <- NULL
  
  pca_data_recon <- data.frame(pca_data_recon)
  colnames(pca_data_recon) <- NULL
  
  write.csv(pca_data_z, file = paste0(data_type,"_PCA_z_", comp_value, ".csv"), row.names = FALSE)
  write.csv(pca_data_recon, file = paste0(data_type,"_PCA_recon_", comp_value,".csv"), row.names = FALSE)
  
  loop_index = loop_index + 1
  
  print(paste(comp_value, "finish!"))
  
}
write.csv(loss_data, file = paste0(data_type,"_PCA_loss.csv"), row.names = FALSE)
print(loss_data)

# End of pca.R
