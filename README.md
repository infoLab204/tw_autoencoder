# A tied-weight autoencoder for the linear dimensionality reduction of sample data 

Sunhee Kim, Chang-Yong Lee


Dimensionality reduction is a method used in machine learning and data science to reduce the dimensions in a dataset. While linear methods are generally less effective at dimensionality reduction than nonlinear methods, they can provide a linear relationship between the original data and the dimensionality-reduced representation, leading to better interpretability. We proposed a tied-weight autoencoder model as a dimensionality reduction model with the advantages of both linear and nonlinear methods. Although the tied-weight autoencoder is a nonlinear dimensionality reduction model, we approximate it to function as a linear model.        

The tw_autoencoder provides Python and R scripts to evaluate the proposed model and compare its performance with other models such as Stacked Autoencoder (SAE), Principal Component Analysis (PCA), and Independent Component Analysis (ICA) using the MNIST, FMNIST, SVHN, and CIFAR10 datasets. The provided Python and R scripts produce the loss function, image reconstruction, and classification results. We provide the Python and R scripts together so that readers can reproduce the results discussed in the manuscript.   

## Install prerequisites
* __Python__ : version 3.6 or later
* __R__ : version 4 or later
* __R-packages__: ggplot2, caret, e1071, nnet, dplyr, fastICA
* __Python__: tensorflow (version 2.2 or later), keras, numpy, matplotlib, scikit-learn, scipy, opencv-python

## Data sets
The dataset used are MNIST, Fashion-MNIST, SVHN and CIFAR10. They are available on Keras and the following links.
* MNIST :  http://yann.lecun.com/exdb/mnist/
* FMNIST : https://github.com/zalandoresearch/fashion-mnist
* CIRAR10 :  https://www.cs.toronto.edu/~kriz/cifar.html
* SVHN : http://ufldl.stanford.edu/housenumbers/
 

## Loading the scripts : ###Copy the following Python and R scripts from its GitHub repository
* Python scripts     
    * __load_data.py__  : loading data set, such as MNIST, FMNIST, SVHN and CIFAR10        
    * __stacked.py__ : learning stacked autoencoder model       
    * __tw_proposed.py__ : learning proposed autoencoder model    
    * __split.py__ : store function according to the class label
    * __recon.py__ : image reconstruction of each model       
* R scripts   
    * __pca.R__ : dimensionality reduction with pricipal component analysis          
    * __ica.R__ : dimensionality reduction with independent componet analysis   
    * __loss.R__ : evaluating mean squared error according to the class label    
    * __classification.R__ : performing classification analysis in terms of support vector machine     

## tutorial
### 1. Loading data set    
To loading data set, run load_data.py with following parameters.
  ```
    Usage : python load_data.py data_type
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR10
  * output  : result of storing MNIST, FMNIST, SHVN, or CIFAR-10 datasets and their labels in 10,000 parts
    
  ```
   (eg) python load_data.py MNIST
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_X_data_1.csv : data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_Y_label_1.csv : label of data set of MNIST

     
### 2. Learning proposed autoencoder models    
To learn the proposed autoencoder model, run tw_proposed.py with MNIST, FMNIST, SVHN and CIFAR10 as input datasets. The scripts will evaluate the units in the code and output layer.    
    
  ```
    Usage : python tw_proposed.py data_type input_data code_size
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR10
  * input_data : input data set
  * code_size : number of nodes in the code layer
  * output  : values of units in the code and output layers
     
  ```
    (eg) python tw_proposed.py MNIST MNIST_X_data_1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_LAE_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_LAE_out_4.csv    

### 3. Learning stacked autoencoder models    
To learn the stacked autoencoder models, run stacked.py with MNIST, FMNIST, SVHN and CIFAR10 as input data sets. The scripts will evaluate the units in the code and output layer.        

  ```
    Usage : python stacked.py data_type input_data code_size
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR10
  * input_data : input data set
  * code_size : number of nodes in the code layer
  * output  : values of units in the code and output layers.
     
  ```
    (eg) python stacked.py MNIST  MNIST_X_data_1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_SAE_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_SAE_out_4.csv    

### 4. Performing PCA for hte dimensionality reduction    
To reduce the dimensionality with PCA, simply run PCA.R with MNIST, FMNIST, SVHN and CIFAR10 as input data sets.   
    
  ```
    Usage : Rscipt pca.R data_type input_data code_size
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * input_data_set : input data set
  * code_size : number of nodes in the code layer
  * output  : values of units in the  dimensionality-reduced codes and output layers
 
    
  ```
   (eg) Rscipt pca.R MNIST MNIST_X_data_1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_PCA_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_PCA_out_4.csv  

### 5. Performing ICA for hte dimensionality reduction     
To reduce the dimensionality with ICA, simply run PCA.R with MNIST, FMNIST, SVHN and CIFAR10 as input data sets.   

  ```
    Usage : Rscipt ica.R data_type input_data code_size
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR10
  * input_data : input data set
  * code_size : number of nodes in the code layer
  * output  : values of units in the  dimensionality-reduced codes and output layers
     
  ```
   (eg) Rscipt ica.R MNIST MNIST_X_data_1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_ICA_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_ICA_out_4.csv  

### 6. Reconstructing input images    
To reconstruct input images, run recon.py with MNIST, FMNIST, SVHN and CIFAR10 as input data sets.       

  ```
    Usage : python recon.py data_type X_data Y_label code_size img_idx
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : data set
  * Y_label : label data set
  * code_size : number of nodes in the code layer
  * img_idx : select image index
  * output  : reconstructed images
     
  ```
   (eg) python recon.py MNIST MNIST_X_data_1.csv MNIST_Y_label_1.csv 4 20
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_image_4_img20.png  
     
### 7. Store data set according to the class labels    
To divide for each class, run split.py with data set and their class labels.         

  ```
    Usage : python split.py data_type X_data Y_label code
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : data set
  * Y_label : label data set
  * code : number of nodes in the code layer
  * output  :  data divided by each class   
    
  ```
   (eg) python  split.py  MNIST MNIST_X_data_1.csv, MNIST_Y_label_1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_LAE_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_SAE_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_PCA_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_ICA_4_class0.csv for each class label   

### 8. Evaluating the loss function for the proposed model, SAE, PCA and ICA    
To evaluate the loss function for all models and each class, simply run loss.R with the values of units in the output layer of each model.         

  ```
    Usage : Rscipt loss.R type X_data code_size
  ```
  * data type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR10
  * X_data : input data set
  * code_size : number of nodes in the code layer
  * output  : mean squred error of all models and each class label
     
  ```
   (eg) Rscipt loss.R MNIST MNIST_X_data_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_total_loss_4.csv : mean squred error of all models      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_total_class_loss_4.csv : mean squred error of each class label        

### 9. Performing classification analysis using support vector machine    
To perform classification analysis, run classification.R with code information as the input data.     

  ```
    Usage : Rscipt classification.R data_type X_data, Y_label code_size
  ```
  * data_type : data type, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : data set
  * Y_data : label data set
  * code : number of nodes in the code layer
  * output  : values of evalutation metrics of each class label
 
    
  ```
   (eg) Rscipt  classification.R MNIST MNIST_X_data_CV1.csv MNIST_Y_label_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(output) MNIST_classification_4.csv 
 
