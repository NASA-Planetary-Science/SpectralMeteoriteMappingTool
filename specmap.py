# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:59:01 2025

@author: maggi
"""
# Import
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter 
import random
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
import time

import os 
from os.path import isfile, join
import pathlib
from scipy.interpolate import interp1d
from scipy import interpolate

def balanced_generator(checkAccuracy = False,writefile = True):
    
    ## This file generates a balanced training dataset for the classification of meteorite or asteroid spectra. 
    # 
    # Inputs: 
    #   checkAccuracy = True or False. This will run a 
    #   section of code that will classify all of the meteorite dataset using the 
    #   newly generated balanced dataset as the training data. This section of the code
    #   takes a few minutes to run.
    #
    #   writefile = true or false; The code saves a dataframe to the disk to esnure
    #   reproducibility. This dataframe contains the generated balanced dataset
    #   for future reference. The code checks if a dataframe file (with a given prefix)
    #   is already on the disk; if set to false, the function will NOT write a file
    #   if set to true, the code will write or overwrite an existing file. The new file
    #   will have a date-time stamp in the filename to avoid an accidental overwriting
    # Returns: 
    #   Balanced dataset - data frame of ~1000 spectra, each classification with ~100 representatives
    #   C - the optimal regularization parameter
    #   Saves a dataframe to the disk named "balanced_training_dataset + current time (in %Y_%m_%d_%H%M format)"

    if  type(checkAccuracy) != bool:
        print('checkAccuracy must be bool')
    if type(writefile) != bool:
        print('writefile must be bool')
    #%% Importing the datasets
    
    # First dataset is from Dyar et al., 2023. A machine learning classification of
    # meteorite spectra applied to understanding asteroids. Icarus, vol. 406, 115718
    # This dataset includes all RELAB meteorite data from c. 2020 and before
    
    # Update 1/20/2026:
    # Changing input data. The original dataset of 2044 spectra from RELAB
    # (1422 from Dyar et al., 2023 and 624 that were archived after 2020)
    # The original dataset prepared for this balanced generator called: all-RELAB_data.xlsx"
    
    # After consulting with Darby on 1/20/2026, we changed the protocol to set aside
    # test data to evaluate the accuracy of the machine. I took each training group
    # and sorted from smallest to largest at 1.25 microns. Then I selected every 5th 
    # spectra to be included in the test set. The remaining data are the training data. 
    # The only exception are the primitive CCs. I hand selected these to ensure that
    # we selected representative meteorites.

    file = "C:/Users/maggi/Dropbox/Work/Research/specmap/Dyar_data/trainTestSplitTraining.xlsx"
    data = pd.read_excel(file)
    
    file2 = "C:/Users/maggi/Dropbox/Work/Research/specmap/Dyar_data/trainTestSplitTesting.xlsx"
    testData = pd.read_excel(file2)
    
    i_s = 2
    spectra = data.iloc[i_s:,:]
    # 
    normalization_index = lambda normalization_wavelength: spectra.columns.get_loc(normalization_wavelength)
    def normalize_spectra(wavelength,spectra) : return spectra.apply(lambda row: row/row[normalization_index(wavelength)], axis=1)
    # Classifications of the data are based on the groupings described in Dyar et al., 2023
    # Group 1: Hydrated carbonaceous chondrites (CM, C2, CR, CI);
    # characterized by phyllosilicates; spectra tend to be relatively featureless
    # with the exception of a 0.7-micron charge transfer band which most CMs exhibit
    # Group 2: Anhydrous carbonaceous chondrites (CO, CV); 
    # relatively weak olivine and pyroxene features
    # Group 3: Olivine dominated  (CK, R and Brachinites); olivine dominated spectra
    # Group 4: Ordinary chondrites and Urelities (H/L/LL/Urelities); 
    # absorptions due to olivine and pyroxene on a continuum depending on the 
    # relative abundances of minerals in a given sample
    # Group 5: Enstatite dominated (EH/EL/Aubrites); featureless spectra due to enstatite
    # and metal.
    # Group 6: Primitive achondrites (ACA/LOD); orthopyroxene-dominated spectra
    # Group 7: Irons (IAB, IIAB); featureless, slightly red spectra 
    # Group 8: HEDs - continuous range of absorption features due to varying amounts of pyroxene
    
    # We have now added two additional classifications to this scheme: CYs (heated CMs)
    # and CPs, primitive carbonaceous chondrites
    # Group 9: CYs - heated CMs; characterized by amorphous materials and secondary 
    # crystalline silicates; typically lack a 0.7 micron feature
    # Group 10: CPs (primitive carbonaceous chondrites); dominated by primary 
    # amorphous silicates (solids that condensed from the proto-solar nebula and remain unchanged)
    
    
    #%% Balancing the dataset
    
    # The goal of this part of the code is to generate a balanced dataset such that
    # to ensure the most compositionally accurated predictions
    # Classification groups with large numbers are randomly undersampled with the 
    # NearMiss (version 2) approach. Small classification groups are oversampled 
    # using the SMOTE approach.
    # Additionally, we balance the meteorite groups within the larger classification group. 
    # Many of the classification groups have multiple meteorite sub-types included. 
    # We find this approch improves the overally accuracy of the ML classification
    n_bal = 106 #number of spectra from each classification group in the balanced training dataset; 
    prov_n = Counter(data['Dyar_group']).keys()
       
    i_s = 2# start of spectra column in 'data'; this overwrites the one above
    c = 1 #class column
    tn = 0 #type number column (needed to balance the datasets)
    bal_y = pd.DataFrame([])
    bal_test = pd.DataFrame([]) 
    cur_count = 0
    for i in range(1,len(prov_n)+1):
        prov = data[data['Dyar_group'].isin([i])] 
        l = len(prov)
        if l <= n_bal:
           oversample = SMOTE(sampling_strategy={i: n_bal})
           X_smote, y_smote = oversample.fit_resample(data.iloc[:,i_s:],data.iloc[:,c])
             
           prov_num_cur_ = np.ones((n_bal))*i
           prov_num_cur = pd.Series(prov_num_cur_)
           df_prov = pd.DataFrame([])
           df_prov.insert(0,"class",prov_num_cur)
           bal_y = pd.concat([bal_y,df_prov],axis=0)
                
           df1 = pd.DataFrame(X_smote)
           df2 = pd.DataFrame(y_smote)
           newdf = pd.concat([df2,df1],axis = 1)
                
               
           bal_test = pd.concat([bal_test,newdf[newdf['Dyar_group'].isin([i])].iloc[:,1:]],axis=0)
           cur_count += len(newdf[newdf['Dyar_group'].isin([i])].iloc[:,0])
           #print(f'i = {i}, count: {cur_count})')
        if l > n_bal:
           type_keys = Counter(prov['type_num']).keys() 
           # this is a dictionary view object; need to use list() then call specific values with [] after
           types_in_prov = len(type_keys)
           cur_num = round(n_bal/types_in_prov)
           num_type_in_prov = np.empty([types_in_prov,1])
           for v in range(0,types_in_prov):
               num_type_in_prov[v] = len(prov[prov['type_num'].isin([list(type_keys)[v]])])
               if num_type_in_prov[v] <= cur_num:
                  oversample = SMOTE(sampling_strategy={list(type_keys)[v]: cur_num})
                  X_smote, y_smote = oversample.fit_resample(prov.iloc[:,i_s:],prov.iloc[:,tn])
                     
                  prov_num_cur_ = np.ones((cur_num))*i
                  prov_num_cur = pd.Series(prov_num_cur_)
                  df_prov = pd.DataFrame([])
                  df_prov.insert(0,"class",prov_num_cur)
                  bal_y = pd.concat([bal_y,df_prov],axis=0)
                       
                  df1 = pd.DataFrame(X_smote)
                  df2 = pd.DataFrame(y_smote)
                  newdf = pd.concat([df2,df1],axis = 1)
                        
                        
                  bal_test = pd.concat([bal_test,newdf[newdf['type_num'].isin([list(type_keys)[v]])].iloc[:,1:]],axis=0)
                        
                  cur_count += len(newdf[newdf['type_num'].isin([list(type_keys)[v]])].iloc[:,0])
                  #print(f'i = {i}, v= {v}, cur_num = {cur_num}, count = {cur_count})')
                     
               if num_type_in_prov[v] > cur_num:
                   undersample = NearMiss(sampling_strategy={list(type_keys)[v]: cur_num},version = 2)
                   X_under, y_under = undersample.fit_resample(prov.iloc[:,i_s:],prov.iloc[:,tn])
                       
                       
                   prov_num_cur_ = np.ones((cur_num))*i
                   prov_num_cur = pd.Series(prov_num_cur_)
                   df_prov = pd.DataFrame([])
                   df_prov.insert(0,"class",prov_num_cur)
                       
                   df1 = pd.DataFrame(X_under)
                   df2 = pd.DataFrame(y_under)
                   newdf = pd.concat([df2,df1],axis = 1)
                       
                   bal_y = pd.concat([bal_y,df_prov],axis=0)
                   bal_test = pd.concat([bal_test,newdf[newdf['type_num'].isin([list(type_keys)[v]])].iloc[:,1:]],axis=0)
                       
                   cur_count += len(newdf[newdf['type_num'].isin([list(type_keys)[v]])].iloc[:,0])
                   #print(f'i = {i}, v= {v}, cur_num = {cur_num}, count = {cur_count})')
                 #   print(f'number of types in prov {types_in_prov}')
             
    
    #%% Cross validation of the balanced dataset to estimate accuracy
    # Assigning the spectra in the balanced dataset into quintiles
    idx_set = {""}
    
    bal_test = pd.DataFrame(bal_test)
    bal_y = pd.DataFrame(bal_y)
    bal_test = bal_test.reset_index()
    bal_y = bal_y.reset_index()
    
    bal = pd.concat([bal_y.iloc[:,1],bal_test.iloc[:,1:]],axis = 1)
    
    bal.insert(1,"quint",np.zeros(len(bal_y)))
    size = len(bal_y)-1
    for i in range(0,5):
        for j in range(0,round(size/5)):
            n = random.randint(0,size)
            while n in idx_set:
                n = random.randint(0,size)
                idx_set.add(n)
                #print(f'{j},{n}')
                bal.iloc[n,1] = i
    
    for g in range(0,len(y_under)):
        if g not in idx_set:
           #print(f'{g}')
           bal.iloc[g,1] = random.randint(1,5)
    
    # Determining the best regularization parameter, C. 
    
    max_iter = 1500 #until 4/2/2025 1:05 pm: used 350
    
    acc_result = [[],[],[],[],[]]
    outer_count = 0
    inner_count = 0
    for i in range(0,5):
        #print(f'{inner_count},{outer_count}')
        test_fold_data = bal[bal["quint"].isin([i])].iloc[:,2:] #defining the test fold spectral data
        test_fold_data_norm_1 = normalize_spectra(1,test_fold_data)
        test_fold_class = bal[bal["quint"].isin([i])].iloc[:,0] # defining the test fold classification, 'y'
        
        all_but_i = [0,1,2,3,4]
        all_but_i.remove(i)
        
        train_folds_data = bal[bal["quint"].isin(all_but_i)].iloc[:,2:] #defining the training data
        train_fold_data_norm_1 = normalize_spectra(1,train_folds_data)
        train_folds_class = bal[bal["quint"].isin(all_but_i)].iloc[:,0] #defining the classification of the training data
        
        for t in range(5,17):
            reg_param = 2**t
            #print(reg_param)
            logreg = LogisticRegression(solver = 'lbfgs',
                                        multi_class = 'multinomial',
                                        C = reg_param,
                                        fit_intercept=False, max_iter = max_iter)
            logreg.fit(train_fold_data_norm_1, train_folds_class)

            y_pred = logreg.predict(test_fold_data_norm_1)
            score = logreg.score(test_fold_data_norm_1, test_fold_class)
            #print(f'outer score is {outer_count} and inner score is {inner_count}')
            acc_result[outer_count].append(score)
            
            inner_count += 1 
            
        outer_count += 1 

    a = np.array(acc_result)
    #np.savetxt("cross_val_accuracies_normdata.csv", a, delimiter = ",")
    a_means = a.mean(axis = 0) # averaging over columns 
    a_max = a_means.max()
    best_index = np.where(a_means == a_max)[0][0] + 5
    C_best = 2**best_index
    #print(f'best average accuracy of across regularization parameters was {a_max.round(3)}')
    #print(f'best C is 2^{best_index} ({2**best_index})')
    
    C = C_best  
    #%% Classifying Test Data
    # Taking the training data to be the curated dataset, classifying the test dataset
    #  defined in the train-test-split(testData)
    # This function then records of the classification of test data
    # Accuracy is assessed by the number of correct predictions divided by the total predictions
    print('Checking train-test-split data')
    testData_n = normalize_spectra(1,testData.iloc[:,i_s:])
    
    n_groups = 10 #the number of classifications
    spectra = testData.iloc[:,i_s:]
    
    probs_output = np.empty([len(testData),n_groups])
    cur_pred = pd.DataFrame([],columns = ["pred"])
    count = 0
    for i in range(len(testData)):
        test_data = np.array(testData_n.iloc[i,:]).reshape(1,-1)
        train_folds_data = bal.iloc[:,2:] #defining the training data
        train_fold_data_norm_1 = normalize_spectra(1,train_folds_data)
        train_folds_class = bal.iloc[:,0] #defining the classification of the training data
        logreg = LogisticRegression(solver = 'lbfgs',
                                multi_class = 'multinomial',
                                C = C_best,
                                fit_intercept=False, max_iter = max_iter) # change random state to get to 1500 if desired
        # fit the model with data
        logreg.fit(train_fold_data_norm_1, train_folds_class)
        y_pred = logreg.predict(test_data)
        probs_output[i,:] = logreg.predict_proba(test_data)
        cur_pred.at[i,"pred"] = y_pred
        #print(f'iteration: {i}, pred. class: {y_pred}, sofmax probs: {prob}')
        count += 1
        #print(f'iteration: {i},{y_pred}')
    
    # determining accuracy        
    testData['pred_result'] = testData['Dyar_group'].reset_index(drop=True) == cur_pred['pred'].reset_index(drop = True)
    num_correct = len(testData[testData["pred_result"] == True])
    percent_correct = num_correct/len(testData)*100
    
    #change the num_corr, P_total and prov_accuract to have the same length as the  number of classification groups
    num_corr = [[],[],[],[],[],[],[],[],[],[],[]]
    P_total = [[],[],[],[],[],[],[],[],[],[],[]]
    test_accuracy = [[],[],[],[],[],[],[],[],[],[],[]]
    count = 0
    
    for i in range(1,11):
        P = testData[testData["Dyar_group"].isin([i])]
        num_corr[count].append(len(P[P["pred_result"] == True]))
        P_total[count].append(len(P))
        test_accuracy[count].append(len(P[P["pred_result"] == True])/len(P)*100)
        #print(f'{i}')
        j = i-1
        #print(f'{prov_accuracy[j]}% province {i} predicted correctly ({num_corr[j]}/{P_total[j]})')
        
        #print(f'{i}{num_corr[j]}{P_total[j]}{test_accuracy[j]}')
        count += 1
   # print(f'Overall accuracy: {percent_correct:.2f}% ({num_correct}/{len(data)})')
    prov_accuracy = []
    #%% All but i analysis using the curated dataset as the training data 
# =============================================================================
#   If selected (checkAccuracy = true), this section of code will perform an
#   and analysis using an all-but-i apporach. 
#   All-but-i: here, one meteorite spectrum (the i-th) is selected as the test data
#   while the balanced dataset is the training data. All of the original RELAB 
#   spectra from which the balanced dataset is selected are classified using the 
#   balanced dataset. The accuracy of the balanced dataset is assessed and pritned
#   in the console
# =============================================================================
    
    data_n = normalize_spectra(1,data.iloc[:,i_s:])
    
    if checkAccuracy:
        # Taking the training data to be the curated dataset, classifying every spectrum
        #  and recording the predicted class. Accuracy is assessed by the number of correct predictions/total predictions
        n_groups = 10 #the number of classifications
        spectra = data.iloc[:,i_s:] 
        #all_data_norm_0_7 = normalize_spectra(0.7,spectra)#These are the 'feature_cols' 
        probs_output = np.empty([len(data),n_groups])
        cur_pred = pd.DataFrame([],columns = ["pred"])
        count = 0
        for i in range(len(data)):
            test_data = np.array(data_n.iloc[i,:]).reshape(1,-1)
            train_folds_data = bal.iloc[:,2:] #defining the training data
            train_fold_data_norm_1 = normalize_spectra(1,train_folds_data)
            train_folds_class = bal.iloc[:,0] #defining the classification of the training data
            logreg = LogisticRegression(solver = 'lbfgs',
                                    multi_class = 'multinomial',
                                    C = C_best,
                                    fit_intercept=False, max_iter = max_iter) # change random state to get to 1500 if desired
            # fit the model with data
            logreg.fit(train_fold_data_norm_1, train_folds_class)
            y_pred = logreg.predict(test_data)
            probs_output[i,:] = logreg.predict_proba(test_data)
            cur_pred.at[i,"pred"] = y_pred
            #print(f'iteration: {i}, pred. class: {y_pred}, sofmax probs: {prob}')
            count += 1
            #print(f'iteration: {i},{y_pred}')
    
    
        # determining accuracy        
        data['pred_result'] = data['Dyar_group'].reset_index(drop=True) == cur_pred['pred'].reset_index(drop = True)
        num_correct = len(data[data["pred_result"] == True])
        percent_correct = num_correct/len(data)*100
        
    
        #change the num_corr, P_total and prov_accuract to have the same length as the  number of classification groups
        num_corr = [[],[],[],[],[],[],[],[],[],[],[]]
        P_total = [[],[],[],[],[],[],[],[],[],[],[]]
        prov_accuracy = [[],[],[],[],[],[],[],[],[],[],[]]
        count = 0
        
        for i in range(1,11):
            P = data[data["Dyar_group"].isin([i])]
            num_corr[count].append(len(P[P["pred_result"] == True]))
            P_total[count].append(len(P))
            prov_accuracy[count].append(len(P[P["pred_result"] == True])/len(P)*100)
            #print(f'{i}')
            #j = i-1
            #print(f'{prov_accuracy[j]}% province {i} predicted correctly ({num_corr[j]}/{P_total[j]})')
            
         #   print(f'{i}{num_corr[j]}{P_total[j]}{prov_accuracy[j]}')
            count += 1
        #print(f'Overall accuracy: {percent_correct:.2f}% ({num_correct}/{len(data)})')
        
    if writefile:
        curr_time = time.strftime("%Y_%m_%d_%H%M", time.localtime())
        f1 = 'balanced_training_dataset'
        filename = f1 + curr_time
        bal.to_pickle(filename)
    return C, bal, test_accuracy, prov_accuracy, filename


def normalize(norm_wave,wave,spec):
    norm_idx = np.where(np.array(wave) >= norm_wave)[0][0]
    normed = np.empty([len(spec.iloc[:,0]),len(spec.iloc[0,:])])
    for i in range(0,len(spec.iloc[0,:])):
        normed[:,i] = spec.iloc[:,i]/spec.iloc[norm_idx,i]
    return normed

def renormalize(norm_wave,wave,spec):
    spec = pd.DataFrame(spec)
    un = np.where(np.array(wave) >= 2.3)[0][0] #point normalize before renormalizing to the desired location
    norm_point = np.where(np.array(wave) == norm_wave)[0][0]

    renormed = np.empty([len(spec.iloc[:,0]),len(spec.iloc[0,:])])
    for i in range(0,len(spec.iloc[0,:])):
        unnorm = spec.iloc[:,i]/spec.iloc[un,i]
        renormed[:,i] = unnorm/unnorm.iloc[norm_point]
    return renormed




def classify(training, train_spec_start_idx, desig, test_wave, test_data, norm_wave, C):
# =============================================================================
#     This function classifies the data in a dataframe (test) using a logistic
#       regression approach, a defined training dataset (training) and regularization parameter (C)
#       The test data can be a single spectrum or multiple spectra of a meteorite or asteroid.
#       The data provided by the user to be classified should be a dataframe in which
#       each row is a spectrum. 
#
#       Inputs:
#           training - training data (balanced dataset), should have the classification
#           as the first column and the spectra in columns spec_start_idx:
#           norm_wave - normalization wavelength for the training and test data    
#           C - regularization parameter
#           desig - the name of the the spectrum (e.g., filename, asteroid designation,
#           meteorite name etc.)
#           test - test data (single spectrum or multiple spectra); 
#           data should be unnormalized; data cannot have NANs 
#           (use [df].bfill(inplace = True) and [df].ffill(inplace = True) to remove NANs
#           and fill with the nearest real data)
#           start_wave - the starting wavelength of the test data
#       Returns: 
#           result - dataframe with 11 columns, first: the predicted class (based on
#           the ML classification), and 10x1 array of the probability of each class
#           based on the LR training data
# =============================================================================
    
        
    # preprocessing: determining the wavelength range of the test data
    wave_ = training.columns.tolist()
    wave = wave_[train_spec_start_idx:]
    start_idx_ = np.where(np.array(wave) >= test_wave[0])
    start_idx = train_spec_start_idx + start_idx_[0][0]
    training_trunc = training.iloc[:,start_idx:]
    training_trunc_wave = np.array(training_trunc.columns.tolist())
    #training_trun_n = normalize(norm_wave,training_trunc_wave,training_trunc)
    
    norm_wave_idx = np.where(training_trunc_wave >= norm_wave)
    training_trunc_n = np.empty([training_trunc.shape[0],training_trunc.shape[1]])
    for i in range(0,training_trunc.shape[0]):
        training_trunc_n[i,:] = training_trunc.iloc[i,:]/training_trunc.iloc[i,norm_wave_idx[0][0]]
        
    test_renorm = renormalize(norm_wave,test_wave,test_data)
    n_groups = 10 
    probs_output = np.empty([test_renorm.shape[1],n_groups])
    pred = pd.DataFrame([],columns = ["pred"])
    count = 0
    for i in range(test_renorm.shape[1]):
        test_data_ = pd.DataFrame(test_renorm[:,i])
        test_data = np.array(test_data_.dropna()).reshape(1,-1)
        #test_class = np.array(all_y.iloc[i])
        train_folds_data = training_trunc_n #defining the training data
        train_folds_class = training.iloc[:,0] #defining the classification of the training data
        logreg = LogisticRegression(solver = 'lbfgs',
                                multi_class = 'multinomial',
                                C = C,
                                fit_intercept=False, max_iter = 350,
                                random_state = 2827) # change random state to get to 1500 if desired
        # fit the model with data
        logreg.fit(train_folds_data, train_folds_class)
        y_pred = logreg.predict(test_data)
        probs_output[i,:] = logreg.predict_proba(test_data)
        pred.at[i,"pred"] = y_pred
        #print(f'iteration: {i}, pred. class: {y_pred}, sofmax probs: {prob}')
        count += 1

    probs_output_ = pd.DataFrame(probs_output)
    probs_cols = ['Group_1_prob','Group_2_prob','Group_3_prob','Group_4_prob','Group_5_prob',
                  'Group_6_prob','Group_7_prob','Group_8_prob','Group_9_prob','Group_10_prob']
    probs_output_.columns = probs_cols
    pred = pd.concat([pred,probs_output_],axis = 1)
    test_desig = pd.DataFrame(desig)
    result = pd.concat([test_desig,pred],axis = 1)
    return result