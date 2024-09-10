argv <-commandArgs(trailingOnly=TRUE)

data_type <- argv[1]  ## data_type : MNIST, FMNIST, HOUSE, CIFAR, BC, or Wine
infileX <- argv[2]   ## X data
infileY <- argv[3]   ## Y data
code <- argv[4]  ## code size

library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(nnet)

X_data <- read.csv(infileX, header = F)
x_data <- as.matrix(X_data)

Y_label <- read.csv(infileY, header = F)
y_label <- Y_label[, 1]

model_list <- c("LAE", "SAE", "PCA", "ICA","LLE","Isomap","VAE")

kf <- 5
idx <- createFolds(factor(y_label), k = kf)

total_classifi_data <- data.frame(classifi_method = c(), model_type = c(), fold_num = c(), class_num = c(),bacc = c(),F1=c())

# 1. Run SVM (with parameter tuning)

classifi_method <- "SVM"

LAE_data <- read.csv(paste0(data_type,"_LAE_z_", code, ".csv"), header = F)
SAE_data <- read.csv(paste0(data_type, "_SAE_z_", code, ".csv"), header = F)
PCA_data <- read.csv(paste0(data_type, "_PCA_z_", code, ".csv"), header = F)
ICA_data <- read.csv(paste0(data_type, "_ICA_z_", code, ".csv"), header = F)
VAE_data <- read.csv(paste0(data_type, "_VAE_z_", code, ".csv"), header = F)
LLE_data <- read.csv(paste0(data_type, "_LLE_z_", code, ".csv"), header = F)
Isomap_data <- read.csv(paste0(data_type, "_Isomap_z_", code, ".csv"), header = F)
  
data_list <- list(LAE_data, SAE_data, PCA_data, ICA_data, VAE_data, LLE_data, Isomap_data)
  
comp_value <- ncol(LAE_data)
  
# SVM Tuning (with 10% data)
  
Start_Time <- Sys.time()
  
tune_kf <- 10
tune_idx <- createFolds(factor(y_label), k = tune_kf)
  
tuning_data <- data.frame(row.names = c(paste0("F", seq(1, comp_value)), "label"))
  
for(data_index in 1:length(data_list)){
    temp_data <- data_list[[data_index]]
    temp_data <- cbind(temp_data, y_label)
    colnames(temp_data) <- c(paste0("F", seq(1, comp_value)), "label")
    
    temp_data <- temp_data[tune_idx[[1]], ]
    
    tuning_data <- rbind(tuning_data, temp_data)
}
  
rows <- sample(nrow(tuning_data))
tuning_data <- tuning_data[rows, ]
  
tune <- tune.svm(as.factor(label) ~ ., data = tuning_data, gamma = 10^c(-3:3), cost = 10^c(-3:3))
  
End_Time <- Sys.time()
  
Run_Time <- End_Time - Start_Time
  
print(paste0("SVM Tuning, z size : ", code, ", finish!"))
print(Run_Time)
  
# Run SVM
  
for(data_index in 1:length(data_list)){
    
    Start_Time <- Sys.time()
    
    model_type <- model_list[data_index]
    
    train_data <- data_list[[data_index]]
    
    train_data <- cbind(train_data, y_label)
    colnames(train_data) <- c(paste0("F", seq(1, comp_value)), "label")
    
    train_data$label <- as.factor(train_data$label)
    
    test_list <- rep(list(), kf)
    test_ylist <- rep(list(), kf)
    
    for(i in 1:kf){
      kf_train_data <- train_data[-idx[[i]], ]
      kf_test_data <- train_data[idx[[i]], ]
      
      m <- svm(label ~ ., data = kf_train_data, cost = tune$best.parameters[1, "cost"],
               gamma = tune$best.parameters[1, "gamma"], kernel = "radial")
      
      test_list[[i]] <- predict(m, newdata = kf_test_data[, -(comp_value+1)])
      test_ylist[[i]] <- y_label[idx[[i]]]
    }
    
    test_actual <- c()
    test_predict <- c()
    
    for(i in 1:kf){
      test_actual <- test_ylist[[i]]
      test_predict <- test_list[[i]]
      
      test_actual <- as.factor(test_actual)
      test_predict <- as.factor(test_predict)

      if data_type=="BC" :
          levels(test_actual) <- 0:1
          levels(test_predict) <- 0:1
      elif data_type=="Wine" :
          levels(test_actual) <- 0:2
          levels(test_predict) <- 0:2
      else :
          levels(test_actual) <- 0:9
          levels(test_predict) <- 0:9
     
      test_confu_mat <- confusionMatrix(test_predict, test_actual)$byClass

      if data_type=="BC":
          for(ind in 0:1){
             bacc_value <- test_confu_mat[ind+1, "Balanced Accuracy"]
             F1_value <- test_confu_mat[ind+1, "F1"]
        
             classifi_data <- data.frame(classifi_method = classifi_method,
                                    model_type = model_type, fold_num = i, class_num = ind,
                                    bacc = bacc_value, F1=F1_value)
        
             total_classifi_data <- rbind(total_classifi_data, classifi_data)
          }
      elif data_type=="Wine":
          for(ind in 0:2){
              bacc_value <- test_confu_mat[ind+1, "Balanced Accuracy"]
              F1_value <- test_confu_mat[ind+1, "F1"]
        
              classifi_data <- data.frame(classifi_method = classifi_method,
                                    model_type = model_type, fold_num = i, class_num = ind,
                                    bacc = bacc_value, F1=F1_value)
        
              total_classifi_data <- rbind(total_classifi_data, classifi_data)
           }
      else :
           for(ind in 0:9){
              bacc_value <- test_confu_mat[ind+1, "Balanced Accuracy"]
              F1_value <- test_confu_mat[ind+1, "F1"]
        
              classifi_data <- data.frame(classifi_method = classifi_method,
                                    model_type = model_type, fold_num = i, class_num = ind,
                                    bacc = bacc_value, F1=F1_value)
        
              total_classifi_data <- rbind(total_classifi_data, classifi_data)
           }     
     
    }
    
    End_Time <- Sys.time()

    
    Run_Time <- End_Time - Start_Time
    
    print(paste0("classifi method : ", classifi_method, ", z size : ", code, ",
                 model type : ", model_type, ", finish!"))
    print(Run_Time)
    
  
}


# 3. Visualize results

summary_classifi_data <- total_classifi_data %>% 
  group_by(classifi_method, model_type, class_num) %>% 
  summarise(bacc_mean = mean(bacc), bacc_se = sd(bacc)/sqrt(n()), F1_mean = mean(F1), F1_se = sd(F1)/sqrt(n()) )

write.csv(summary_classifi_data, file = paste0(data_type, "_summary_classification_",code,".csv"), row.names = F)

## Enf of classcification.R
