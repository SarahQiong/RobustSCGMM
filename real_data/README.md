# Data preparation

1. Download NIST by_class.zip
2. Get the images for 0-9 (corresponds to folder 30-39) and A-J (41-49, 4a)
3. Keep the original folder for training for each digit, and keep hsf_4 as test as their original documentation
4. Put all the training dataset for digits into dataset/train/digits, test dataset for digits into dataset/test, the training set for letters into dataset/train/letters


# Model training

`python nn_feature_extractor.py --train --ckpf '' --log-interval 1000`

# Evaluation and feature extraction

`python nn_feature_extractor.py --evaluate`

The accuracies of the trained model on different dataset
    
    - Digits train: 341095/344307 = 0.9721208
    - Digits test:  56544/58646 = 0.9380691
    - Letters train: 5869/65971 = 0.07130406


##############################################
# NIST dataset download
##############################################
Directory for dataset download: https://www.nist.gov/srd/nist-special-database-19
download the 2nd edition release on September 2016
download the zip file by_class
read the user guide for the description of the subfoloders
1. the directory 30-39 contains the digits 0-9
2. within each directory, there is a suggested subdirectory for training
3. the hsf_4 is the suggested test directory


#############################################
# Feature extraction
#############################################
1. Train the model on NIST training dataset
	python nn_feature_extractor.py --train
this model gets a classification accuracy of about 98.34% on the training and 97.67% on the test

2. Use the pretrained model to do feature extraction
    
    python nn_feature_extractor.py --evaluate

3. Use extracted 50D features from NIST dataset and perform clustering

