#Calculate and visualize loss
argv <-commandArgs(trailingOnly=TRUE)

data_type <-argv[1]  ## data_type : MNIST, FMNIST, HOUSE, CIFAR
infile <- argv[2] ## CV Number
code <- argv[3]

library(ggplot2)
library(dplyr)

# 1. Calculate loss per code size

X_data <- read.csv(infile, header = F)

X_data <- as.matrix(X_data)

total_data <- data.frame(model_type = c(), mean = c(), se = c())

LAE_recon <- read.csv(paste0(data_type, "_LAE_out_", code, ".csv"), header = F)
SAE_recon <- read.csv(paste0(data_type, "_SAE_out_", code, ".csv"), header = F)
PCA_recon <- read.csv(paste0(data_type, "_PCA_out_", code, ".csv"), header = F)
ICA_recon <- read.csv(paste0(data_type, "_ICA_out_", code, ".csv"), header = F)
    
LAE_recon <- as.matrix(LAE_recon)
ICA_recon <- as.matrix(ICA_recon)
SAE_recon <- as.matrix(SAE_recon)
PCA_recon <- as.matrix(PCA_recon)
  
LAE_recon_loss <- c()
ICA_recon_loss <- c()
SAE_recon_loss <- c()
PCA_recon_loss <- c()
  
for(i in 1:nrow(PCA_recon)){
    LAE_recon_loss[i] <- mean((X_data[i, ] - LAE_recon[i, ])^2)
    ICA_recon_loss[i] <- mean((X_data[i, ] - ICA_recon[i, ])^2)
    SAE_recon_loss[i] <- mean((X_data[i, ] - SAE_recon[i, ])^2)
    PCA_recon_loss[i] <- mean((X_data[i, ] - PCA_recon[i, ])^2)
}
  
total_rec_loss <- data.frame(model_type = rep(c("LAE", "SAE", "PCA", "ICA"), each = length(PCA_recon_loss)), loss = c(LAE_recon_loss, SAE_recon_loss, PCA_recon_loss, ICA_recon_loss))
temp_data <- total_rec_loss %>% group_by(model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
temp_data <- data.frame(temp_data)
total_data <- rbind(total_data, temp_data)
write.csv(total_data, file = paste0(data_type, "_total_model_loss_",code,".csv"), row.names = F)


# 2. Calculate loss per code size and each class

total_data <- data.frame(model_type = c(), class_index = c(), mean = c(), se = c())

for(class_index in 0:9) {  

  X_data <- read.csv(paste0(data_type, "_X_data_class_", class_index, ".csv"), header = F)
  X_data <- as.matrix(X_data)
  
  LAE_recon_class <- read.csv(paste0(data_type, "_LAE_recon_", code, "_class", class_index, ".csv"), header = F)
  SAE_recon_class <- read.csv(paste0(data_type, "_SAE_recon_", code, "_class", class_index, ".csv"), header = F)
  PCA_recon_class <- read.csv(paste0(data_type, "_PCA_recon_", code, "_class", class_index, ".csv"), header = F)
  ICA_recon_class <- read.csv(paste0(data_type, "_ICA_recon_", code, "_class", class_index, ".csv"), header = F)
    
  LAE_recon_class <- as.matrix(LAE_recon_class)
  SAE_recon_class <- as.matrix(SAE_recon_class)
  PCA_recon_class <- as.matrix(PCA_recon_class)
  ICA_recon_class <- as.matrix(ICA_recon_class)
    
  LAE_recon_class_loss <- c()
  SAE_recon_class_loss <- c()
  PCA_recon_class_loss <- c()
  ICA_recon_class_loss <- c()
    
  for(i in 1:nrow(PCA_recon_class)){
      LAE_recon_class_loss[i] <- mean((X_data[i, ] - LAE_recon_class[i, ])^2)
      SAE_recon_class_loss[i] <- mean((X_data[i, ] - SAE_recon_class[i, ])^2)
      PCA_recon_class_loss[i] <- mean((X_data[i, ] - PCA_recon_class[i, ])^2)
      ICA_recon_class_loss[i] <- mean((X_data[i, ] - ICA_recon_class[i, ])^2)
  }
    
  total_recon_loss <- data.frame(model_type = rep(c("LAE", "SAE", "PCA", "ICA"), each = length(PCA_recon_class_loss)), class_index = class_index, loss = c(LAE_recon_class_loss,SAE_recon_class_loss, PCA_recon_class_loss, ICA_recon_class_loss))
    
  temp_data <- total_recon_loss %>% group_by(class_index, model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
    
  temp_data <- data.frame(temp_data)
    
  total_data <- rbind(total_data, temp_data)
    
  print(paste(class_index, code, "finish!"))
    
  }
  

write.csv(total_data, file = paste0(data_type, "_total_class_loss_",code,".csv"), row.names = F)

