# tw_autoencoder: A tied-weight autoencoder for the linear dimensionality reduction of sample data with Python and R

Sunhee Kim, Chang-Yong Lee

tw_autoencoder represents Python and R scripts to assess the proposed auto-encoder model and to compare its performance with other models, such as stacked autoencoder (SAE),  principal component analysis (PCA), and independent componet analysis(ICA) by using MNIST, Fashion-MNIST, SVHN and CIFAR10 data sets.


We proposed a tied-weight autoencoder model as a dimensionality reduction model with the merit of both linear and nonlinear methods. Although the tied-weight autoencoder is a nonlinear dimensionality reduction model, we approximate it to function as a linear model. The Python and R scripts provide the assessment of the proposed model and compare with other models in terms of the loss function, image reconstruction, and classification results. We provide Python and R scripts together in order for the readers to reproduce the results discussed in the manuscript.

### Install prerequisites:
* __Python__ : version 3.6 or later
* __R__ : version 4 or later
* __R-packages__: ggplot2, caret, e1071, nnet, dplyr, fastICA
* __Python__: tensorflow (version 2.2 or later), keras, numpy, matplotlib, scikit-learn

## Data sets
1. Data acquisition
    * MNIST :  in the installed tensorflow
    * FMNIST : in the installed tensorflow
    * CIRAR10: in the installed tensorflow
    * SVHN : http://ufldl.stanford.edu/housenumbers/
        * Format 2: Cropped Digits: train_32x32.mat, test_32x32.mat
 

### Loading the scripts: 
   copy the following Python and R scripts from its GitHub repository
       * Python scripts 
           * load_data.py  : loading data set, such as MNIST, Fashion-MNIST, SVHN and CIRAR10
           * stacked.py : learning stacked autoencoder model
           * proposed.py : learning proposed autoencoder model
           * split.py : store loss function according to the class label
           * recon.py : image reconstruction of each model
       * R scripts
           * pca.R : dimensionality redction with pricipal component analysis
           * ica.R : dimensionality redction with independent componet analysis
           * calculate.R : evaluating mean squared error according to the class label
           * classification.R : performing classification analysis in terms of support vector machine


