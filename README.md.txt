Overview: 

# Folder 'DeMeo_csv_files' - The folder containing the Bus-DeMeo asteroid spectra. These data are loaded into a dataframe in the working.py file and classified using the balanced training dataset included in this repo.

specmap.py - The Spectral Meteorite Mapping Tool are several functions that create balanced training datasets and use a supervised machine learning approach (logistic regression) to classify data. The logistic regression is trained using the provided balanced dataset and indicated hyperparameter. These functions are used to classify asteroid or meteorite spectra into one of 10 groups based on their spectral similarity.

Working.py - This file is an example of how the software is called and used in python. This file can be localized by other scientists to use and understand our software.

# Pickle files:

balanced_training_dataset2026_01_20_1620 - This file is the balanced training dataset that was selected based on its performance classifying the test dataset. We generated 30 balanced datasets then assessed their performance on the test data. This training dataset was selected based on how well it classified hydrated carbonaceous chondrites and iron meteorites. The C parameter for this training dataset is 64. This was obtained by a cross-validation analysis for this balanced dataset where we determined the best overall accuracy for this C parameter.

Bus-DeMeo-results - This pickle includes the assigned classification and group probability for each asteroid in the dataset. Can be loaded independently of the working.py and used as needed.

BD_asteroid_characteristics.xlsx - This file contains selected orbital and physical parameters of the 371 asteroids that we classify in the paper. These are all publicly available data that were gathered from the JPL Solar System Dynamics Small Bodies search query (ssd.jpl.nasa.gov). 

Dyar_etal(2023)_dataset.xlsx - This file is the original 1422 meteorite spectra. It is a duplicate of the supplemental data file from Dyar et al., 2023. Citation: Dyar, M.D., Wallace, S.M., Burbine, T.H. and Sheldon, D.R., 2023. A machine learning classification of meteorite spectra applied to understanding asteroids. Icarus, 406, p.115718.

new_RELAB_spectra_resampled.xlsx - This file includes the ~600 new spectra that were put on the RELAB archive after Dyar et al., (2023) created their dataset. These spectra augment the original dataset.

trainTestSplitTesting.xlsx - This file contains 20% of the spectral dataset (including the original 1422 meteorite spectra from Dyar et al., 2023 as well as the new data). We stratified the dataset (ordering by unnormalized reflectance value at 1.25-µm) and selected every 5th spectrum. These spectra we reserved as a test cohort to validate the accuracy of our classifier.

trainTestSplitTraining.xlsx - This file contains 80% of the dataset. These data are used to generate a balanced training dataset then to train the logistic regression classifier.

Bus-DeMeo-results_vfinal.xlsx - This file contains the physical characteristics as well as the predicted class and probability of each class for all 371 asteroids in the study.
