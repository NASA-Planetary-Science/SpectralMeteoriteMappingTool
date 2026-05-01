# Overview: 

This repository contains the functions, data and scripts that produced the results in two publications: 

McAdam, M. M., Dotson, J., Dyar, M. D., Hidden in plain sight: a new Main Belt population of aqueously altered and thermally metamorphosed asteroids. _Submitted to Earth and Space Sciences Journal May 2026_

McAdam, M. M., Dotson, J., Dyar, M. D., Spectral Meteorite Mapping Tool for Asteroid Composition Identification: uses in Planetary Defense and General Asteroid Science. (in preparation)

Please contact Dr. McAdam with any questions regarding the code and data: margaret.m.mcadam at nasa dot gov

### Dependencies: 
The code and scripts were created using Python version 3.11.11. 
I used conda: spyder-env integrated development environment to write, test and run the code and scripts presented here. 

To run the functions and scripts, you'll need the following packages: 
- Numpy
- Pandas
- SciPy 
- SciKitLearn
- MatplotLib 
- PySankey

  
### Functions and scripts

**specmap.py** - The Spectral Meteorite Mapping Tool are several functions that create balanced training datasets and use a supervised machine learning approach (logistic regression) to classify data. The logistic regression is trained using the provided balanced dataset and indicated hyperparameter. These functions are used to classify asteroid or meteorite spectra into one of 10 groups based on their spectral similarity.

 _classify_ function is the function that classifies near-infrared spectroscopy into one of 10 groups of meteorites using a logistic regression machine learning algorithm trained on near-infrared archival meteorite spectra. This function can be used to classify any near-infrared dataset that has the same sampling and wavelength range of the Bus-DeMeo asteroid spectra saved in the _DeMeo_csv_files_ folder.

_balanced_generator_ function. The training data has uneven classes and so we elected to balance our training dataset, over- and under-sampling the meteorite classification groups to create a balanced training dataset. We generated multiple balanced training datasets and selected the one that performed best for the iron and hydrated carbonaceous chondrite classification groups. The function that generates the balanced training dataset is called 'balanced_generator' in the specmap.py code. The balanced_generator function uses the data saved in trainTestSplitTraining.xlsx to created balanced training datasets.

There are two other functions in this file, _normalize_ and _renormalize_, which are called in the function 'classify'. Data normalization is a rescaling of data. In the spectroscopy context, we divide the whole spectrum by one particular datapoint in the spectrum; that point becomes unity and the rest of the data have the same attributes (absorption bands, slope etc.) as the raw data but are now scaled. Spectral normalization allows us to directly compare spectral features of different spectra. The _renormalize_ function takes any data, including previously normalized data, and renormalizes it to the user-defined normalization point. The _normalize_ function takes unnormalized data and normalizes it at the user-defined wavelength.  

**BusDeMeoReanalysis.py** - This file is the script used to ingest, classify and generate the results presented in the publications listed above. We have not included all the plots for the figures in the papers but two of the Sankey diagrams are demonstrated. 

### Pickle files:

**balanced_training_dataset2026_01_20_1620** - This file is the balanced training dataset that was selected based on its performance classifying the test dataset. We generated 30 balanced datasets then assessed their performance on the test data. This training dataset was selected based on how well it classified hydrated carbonaceous chondrites and iron meteorites. The C parameter for this training dataset is 64. This was obtained by a cross-validation analysis for this balanced dataset where we determined the best overall accuracy for this C parameter.

**Bus-DeMeo-results** - This pickle includes the assigned classification and group probability for each asteroid in the dataset. Can be loaded independently of the BusDeMeoReanalysis.py and used as needed.

### Data files
Folder **DeMeo_csv_files** 
The folder containing the Bus-DeMeo asteroid spectra. These data are loaded into a dataframe in the working.py file and classified using the balanced training dataset included in this repo.

**trainTestSplitTesting.xlsx** - This file contains 20% of the spectral dataset (including the original 1422 meteorite spectra from Dyar et al., 2023 as well as the new data). We stratified the dataset (ordering by unnormalized reflectance value at 1.25-µm) and selected every 5th spectrum. These spectra we reserved as a test cohort to validate the accuracy of our classifier.

**trainTestSplitTraining.xlsx** - This file contains 80% of the dataset. These data are used to generate a balanced training dataset then to train the logistic regression classifier.

**Dyar_etal(2023)_dataset.xlsx** - This file is the original 1422 meteorite spectra. It is a duplicate of the supplemental data file from Dyar et al., 2023. Citation: Dyar, M.D., Wallace, S.M., Burbine, T.H. and Sheldon, D.R., 2023. A machine learning classification of meteorite spectra applied to understanding asteroids. Icarus, 406, p.115718.

**new_RELAB_spectra_resampled.xlsx** - This file includes the ~600 new spectra that were put on the RELAB archive after Dyar et al., (2023) created their dataset. These spectra augment the original dataset.

**BD_asteroid_characteristics.xlsx** - This file contains selected orbital and physical parameters of the 371 asteroids that we reanalyze in the paper. These are all publicly available data that were gathered from the JPL Solar System Dynamics Small Bodies search query (ssd.jpl.nasa.gov). 

**Bus-DeMeo-results_vfinal.xlsx** - This file contains the physical characteristics as well as the predicted class and probability of each class for all 371 asteroids in the study.
