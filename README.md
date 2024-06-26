# A tied-weight autoencoder for the linear dimensionality reduction of sample data 

Sunhee Kim, Chang-Yong Lee

tw_autoencoder represents Python and R scripts to assess the proposed auto-encoder model and to compare its performance with other models, such as stacked autoencoder (SAE),  principal component analysis (PCA), and independent componet analysis(ICA) by using MNIST, Fashion-MNIST, SVHN and CIFAR-10 data sets.


We proposed a tied-weight autoencoder model as a dimensionality reduction model with the merit of both linear and nonlinear methods. Although the tied-weight autoencoder is a nonlinear dimensionality reduction model, we approximate it to function as a linear model. The Python and R scripts provide the assessment of the proposed model and compare with other models in terms of the loss function, image reconstruction, and classification results. We provide Python and R scripts together in order for the readers to reproduce the results discussed in the manuscript.

## Install prerequisites
* __Python__ : version 3.6 or later
* __R__ : version 4 or later
* __R-packages__: ggplot2, caret, e1071, nnet, dplyr, fastICA
* __Python__: tensorflow (version 2.2 or later), keras, numpy, matplotlib, scikit-learn, scipy, opencv-python

## Data sets
The dataset used are MNIST, Fashion-MNIST, SVHN and CIFAR-10. They are available on Keras and the following links.
* MNIST :  http://yann.lecun.com/exdb/mnist/
* FMNIST : https://github.com/zalandoresearch/fashion-mnist
* CIRAR-10 :  https://www.cs.toronto.edu/~kriz/cifar.html
* SVHN : http://ufldl.stanford.edu/housenumbers/
 

## Loading the scripts 
Copy the following Python and R scripts from its GitHub repository
* Python scripts     
    * __load_data.py__  : loading data set, such as MNIST, Fashion-MNIST, SVHN and CIFAR-10    
    * __stacked.py__ : learning stacked autoencoder model    
    * __tw_proposed.py__ : learning proposed autoencoder model    
    * __split.py__ : store function according to the class label    
    * __recon.py__ : image reconstruction of each model    
* R scripts   
    * __pca.R__ : dimensionality reduction with pricipal component analysis    
    * __ica.R__ : dimensionality reduction with independent componet analysis    
    * __loss.R__ : evaluating mean squared error according to the class label    
    * __classification.R__ : performing classification analysis in terms of support vector machine    

## Scipts tutorial
* Loading data set    
    To loading data set, run load_data.py with following parameters.
  ```
    Using : python load_data.py type
  ```
  * type : data set, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * output  : result of storing MNIST, FMNIST, SHVN, or CIFAR-10 datasets and their labels in 10,000 parts
    
  ```
   (eg) python load_data.py MNIST
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_X_data_CV1.csv : data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_Y_label_CV1.csv : label of data set of MNIST

     
* Learning autoencoder models    
    To learn the autoencoder models, run stacked.py and proposed.py . The scripts will evaluate the units in the code and output layer.     
  ```
    Format : python tw_proposed.py type input code
  ```
  * type : data set, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * input : data set
  * code : number of nodes in the code layer
  * output  : values of units in the code and output layers.
 
    
  ```
    (eg) python tw_proposed.py MNIST  MNIST_X_data_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_LAE_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_LAE_out_4.csv    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(note) In a similar manner, run stacked.py    
<br>
* Performing PCA for hte dimensionality reduction    
    To reduce the dimensionality with PCA, simply run PCA.R with MNIST, FMNIST, SVHN and CIFAR-10 as input data sets.   
    Output will be the dimensionality-reduced codes and output layer.
  ```
    Format : Rscipt pca.R type input_data_set code_size
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * input_data_set : input data set
  * code : number of nodes in the code layer
  * output  : values of units in the code and output layers
 
    
  ```
   (eg)   Rscipt pca.R MNIST MNIST_X_data_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_PCA_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_PCA_out_4.csv  

* Performing ICA for hte dimensionality reduction     
    To reduce the dimensionality with ICA, simply run PCA.R with MNIST, FMNIST, SVHN and CIFAR-10 as input data sets.    
    Output will be the dimensionality-reduced codes and output layer.
  ```
    Format : Rscipt ica.R type input_data_set code_size
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * input_data_set : input data set
  * code : number of nodes in the code layer
  * output  : values of units in the code and output layers
 
    
  ```
   (eg)   Rscipt ica.R MNIST MNIST_X_data_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_ICA_code_4.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_ICA_out_4.csv  

* Reconstructing input images    
    To reconstruct input images, simply run recon.py with MNIST, FMNIST, SVHN and CIFAR-10 as input data sets.    
  ```
       python recon.py type code img_idx
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * code : number of nodes in the code layer
  * img_idx : select image index
  * output  : reconstructed images
 
    
  ```
   (eg)  python recon.py MNIST 20
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_image_recon_4_20.png  
     
* Store loss function according to the class labels    
    To get loss function for each class, run split.py with data set and their class labels. Ouput will be loss functions of data set for each class label.
  ```
    Format : python split.py type X_data Y_label code
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : data set
  * Y_data : label data set
  * code : number of nodes in the code layer
  * output  : values of units in the output layer
 
    
  ```
   (eg)  python  split.py  MNIST 
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_LAE_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_SAE_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_PCA_4_class0.csv for each class label    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_ICA_4_class0.csv for each class label   

* Evaluating the loss function for the proposed model, SAE, PCA and ICA    
    To evaluate the loss function for all models, simply run loss.R with the values of units in the output lapyer of each model.    
    Ouput will be the loss function of all models.
  ```
    Format : Rscipt loss.R type X_data code_size
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : input data set
  * code : number of nodes in the code layer
  * output  : values of units in the output layer of each class label
 
    
  ```
   (eg)   Rscipt loss.R MNIST MNIST_X_data_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_total_loss_4.csv     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MNIST_total_class_loss_4.csv    

* Performing classification analysis using support vector machine    
    To classify data set, run classification.R with the codes of all models as the input data.
    Ouput will be the classification results
  ```
    Format : Rscipt classification.R type X_data, Y_label code_size
  ```
  * type : datatype, select data set from MNIST, FMNIST, SVHN or CIFAR-10
  * X_data : data set
  * Y_data : label data set
  * code : number of nodes in the code layer
  * output  : values of units in the output layer of each class label
 
    
  ```
   (eg)   Rscipt  classification.R MNIST MNIST_X_data_CV1.csv MNIST_Y_label_CV1.csv 4
  ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(eg) MNIST_classification_4.csv 
 
