argv <-commandArgs(trailingOnly=TRUE)

library(ica)

data_type <- argv[1]  ## data_type : MNIST, FMNIST, HOUSE, CIFAR

# Load MNIST test data
X_data <- read.csv(paste0(data_type,"_X_data.csv"), header = F)
x_data <- as.matrix(X_data)

# Run PCA and save result(reconstruct and code(Z, principal component))

# Define code size list
z_list <- c(4,8,12,16,20) 


loss_data <- data.frame(z_size = z_list, loss = rep(NA, length(z_list)))

mu <- (x_data - scale(x_data, scale = F))

loop_index <- 1

for(comp_value in z_list){ 
  print(comp_value)
  ica_df <- ica(x_data, nc=comp_value)
  
  ica_data_z <- ica_df$S[,1:comp_value]
  
  ica_data_recon <- ica_df$S[,1:comp_value] %*% t(ica_df$M[, 1:comp_value]) + mu
  
  loss_data$loss[loop_index] <- mean((ica_data_recon - x_data)^2)
  print(loss_data)

  ica_data_z <- data.frame(ica_data_z)
  colnames(ica_data_z) <- NULL
  
  ica_data_recon <- data.frame(ica_data_recon)
  colnames(ica_data_recon) <- NULL
  
  write.csv(ica_data_z, file = paste0(data_type,"_ICA_z_", comp_value, ".csv"), row.names = FALSE)
  write.csv(ica_data_recon, file = paste0(data_type,"_ICA_recon_", comp_value, ".csv"), row.names = FALSE)
  
  loop_index = loop_index + 1
  
  print(paste(comp_value, "finish!"))
  
}
write.csv(loss_data, file=paste0(data_type,"_ICA_loss.csv"), row.names=FALSE)
print(loss_data)

## End of ica.R
