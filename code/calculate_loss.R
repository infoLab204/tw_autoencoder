argv <-commandArgs(trailingOnly=TRUE)
data_type <- argv[1]  ## data_type : MNIST, FMNIST, HOUSE, CIFAR

library(ggplot2)
library(dplyr)

# 1. Calculate loss per code size

X_data <- read.csv(paste0(data_type, "_X_data.csv"), header = F)

X_data <- as.matrix(X_data)

total_data <- data.frame(model_type = c(), z_size = c(), mean = c(), se = c())

z_list <- c(4, 8, 12, 16, 20)


for(loop_index in 1:length(z_list)) {
  
  z_size <- z_list[loop_index]

  LAE_rec_data <- read.csv(paste0(data_type, "_proposed_recon_", z_size, ".csv"), header = F)
  ICA_rec_data <- read.csv(paste0(data_type, "_ICA_recon_", z_size, ".csv"), header = F)
  SBAE_rec_data <- read.csv(paste0(data_type, "_Stacked_recon_", z_size, ".csv"), header = F)
  PCA_rec_data <- read.csv(paste0(data_type, "_PCA_recon_", z_size, ".csv"), header = F)
    
  LAE_rec_data <- as.matrix(LAE_rec_data)
  ICA_rec_data <- as.matrix(ICA_rec_data)
  SBAE_rec_data <- as.matrix(SBAE_rec_data)
  PCA_rec_data <- as.matrix(PCA_rec_data)
  
  LAE_rec_loss <- c()
  ICA_rec_loss <- c()
  SBAE_rec_loss <- c()
  PCA_rec_loss <- c()
  
  for(i in 1:nrow(PCA_rec_data)){
    LAE_rec_loss[i] <- mean((X_data[i, ] - LAE_rec_data[i, ])^2)
    ICA_rec_loss[i] <- mean((X_data[i, ] - ICA_rec_data[i, ])^2)
    SBAE_rec_loss[i] <- mean((X_data[i, ] - SBAE_rec_data[i, ])^2)
    PCA_rec_loss[i] <- mean((X_data[i, ] - PCA_rec_data[i, ])^2)
  }
  
  total_rec_loss <- data.frame(model_type = rep(c("PCA", "ICA", "SBAE", "LAE"), each = length(PCA_rec_loss)), z_size = z_size, loss = c(PCA_rec_loss,ICA_rec_loss, SBAE_rec_loss, LAE_rec_loss))
  
  temp_data <- total_rec_loss %>% group_by(z_size, model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
  
  temp_data <- data.frame(temp_data)
  
  total_data <- rbind(total_data, temp_data)
  
  print(paste(z_size, "finish!"))
  
}

ggplot(total_data, aes(x = z_size, y = mean, group = model_type, color = model_type)) +
  geom_line() + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 1, size = 0.5) +
  scale_x_continuous(breaks = seq(4, 20, 4)) + ggtitle(paste0(data_type, ", model recon loss")) +
  theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))

write.csv(total_data, file = paste0(data_type, "_total_code_loss_data.csv"), row.names = F)



# 2. Calculate loss per code size and each class

total_data <- data.frame(model_type = c(), z_size = c(), class_index = c(), mean = c(), se = c())

for(class_index in 0:9) {  

  X_test <- read.csv(paste0(data_type, "_X_data_",z_size,"_class", class_index, ".csv"), header = F)
  
  X_test <- as.matrix(X_test)
  
  for(z_size in z_list) {
    
    LAE_rec_data <- read.csv(paste0(data_type, "_proposed_recon_", z_size, "_class", class_index, ".csv"), header = F)
    ICA_rec_data <- read.csv(paste0(data_type, "_ICA_recon_", z_size, "_class", class_index, ".csv"), header = F)
    SBAE_rec_data <- read.csv(paste0(data_type, "_Stacked_recon_", z_size, "_class", class_index, ".csv"), header = F)
    PCA_rec_data <- read.csv(paste0(data_type, "_PCA_recon_", z_size, "_class", class_index, ".csv"), header = F)
    
    LAE_rec_data <- as.matrix(LAE_rec_data)
    ICA_rec_data <- as.matrix(ICA_rec_data)
    SBAE_rec_data <- as.matrix(SBAE_rec_data)
    PCA_rec_data <- as.matrix(PCA_rec_data)
    
    LAE_rec_loss <- c()
    ICA_rec_loss <- c()
    SBAE_rec_loss <- c()
    PCA_rec_loss <- c()
    
    for(i in 1:nrow(PCA_rec_data)){
      LAE_rec_loss[i] <- mean((X_data[i, ] - LAE_rec_data[i, ])^2)
      ICA_rec_loss[i] <- mean((X_data[i, ] - ICA_rec_data[i, ])^2)
      SBAE_rec_loss[i] <- mean((X_data[i, ] - SBAE_rec_data[i, ])^2)
      PCA_rec_loss[i] <- mean((X_data[i, ] - PCA_rec_data[i, ])^2)
    }
    
    total_rec_loss <- data.frame(model_type = rep(c("PCA", "ICA", "SBAE", "LAE"), each = length(PCA_rec_loss)), z_size = z_size, class_index = class_index, loss = c(PCA_rec_loss, ICA_rec_loss,SBAE_rec_loss, LAE_rec_loss))
    
    temp_data <- total_rec_loss %>% group_by(z_size, class_index, model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
    
    temp_data <- data.frame(temp_data)
    
    total_data <- rbind(total_data, temp_data)
    
    print(paste(class_index, z_size, "finish!"))
    
  }
  
}

z4_total_data <- total_data[total_data$z_size == 4, ]
z8_total_data <- total_data[total_data$z_size == 8, ]
z12_total_data <- total_data[total_data$z_size == 12, ]
z16_total_data <- total_data[total_data$z_size == 16, ]
z20_total_data <- total_data[total_data$z_size == 20, ]

ggplot(z4_total_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) +
  geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) +
  scale_x_continuous(breaks = c(0:9)) +
  ggtitle(paste0(data_type, ", Z = ", 4 ,", model recon loss")) +
  theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))

ggplot(z8_total_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", 8 ,", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))

ggplot(z12_total_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", 12 ,", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))

ggplot(z16_total_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", 16 ,", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))
ggplot(z20_total_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", 20 ,", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))

write.csv(z4_total_data, file = paste0(data_type, "_z4_class_loss_data.csv"), row.names = F)
write.csv(z8_total_data, file = paste0(data_type, "_z8_class_loss_data.csv"), row.names = F)
write.csv(z12_total_data, file = paste0(data_type, "_z12_class_loss_data.csv"), row.names = F)
write.csv(z16_total_data, file = paste0(data_type, "_z16_class_loss_data.csv"), row.names = F)
write.csv(z20_total_data, file = paste0(data_type, "_z20_class_loss_data.csv"), row.names = F)

# write.csv(total_data, file = paste0(data_type, "_total_class_loss_data.csv"), row.names = F)

