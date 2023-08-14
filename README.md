# File types classification 
## General Overview
This code includes two novel architectures
<br>(1) Transformers NLP for file classification
<br>(2) Handcrafted statistical with FiFTy ensemble
## Directory structure
<br>1. The structure of the /src directory is as follows
  
![image](https://github.com/simonalis/ThesisPaper/assets/104734787/2b1b23d4-b6ee-47ac-83d5-6b398e01fe39)
<br>2. train.npz, test.npz, val.npz files from FiFTy dataset should be located under /src/512_1/ directory
<br>3. 2 models created are  under /src/512_1/ directory

![image](https://github.com/simonalis/ThesisPaper/assets/104734787/2bb71063-e507-4813-8e2e-c6e2bf617bf9)
<br>4. All outputs will be created under /src/512_1/ directory
## Files description
<br>1. LoadData - loads the data from .npz files and is used for both models
<br>2. TorchDataLoaderWithNLP - creates NLP based architecture, trains and predicts accuracy using FiFTy dataset (*.npz files)
<br>This file is for related to the implementation of architecture (1).
<br>3. The architecture (2) is created in 2 steps - creates statistical features, creates and trains statistical model (a) and then creates ensemble using statistical model and Fifty model, then we are doing train with Transfer Learning and predict (b).
<br>(a)StatisticalFeatures
<br>(b)StatDNNwithFiFTy
## How to execute
<br>In order explore architecture
<br>1. (1), please use the TorchDataLoaderWithNLP file as main
<br>2. (2), please use StatDNNwithFiFTy file as main

<br>It is possible to run only predict function for each model as models are loaded to this repository.
<br>(1) saved_weights_512_1_full_chunked_roberta.pt
<br>(2) tf_stat_fifty_weights.h5
<br>Models above can be downloaded from my drive:
<br>https://drive.google.com/drive/folders/1nc1l70pEMZRA31_u1sADFovxRZQND60A?usp=drive_link

<br>You will need to change the code appropriately, comment train function and uncomment lines which load weights near train function.

It is also possible to train models from scratch and then run predict function.

## File Types
<br>Classifier to 75 file types
<br>![image](https://github.com/simonalis/ThesisPaper/assets/104734787/5133a2c2-3460-4640-ada6-7ee841c145db)
<br>*Taken from Fifty referenced below.
## References
FiFTy model used as part of architecture (2) as well as same dataset.
<br>https://github.com/mittalgovind/fifty/tree/master
