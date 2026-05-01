"""
Working Python file to import Bus-DeMeo spectral data, Bus-DeMeo asteroid 
sekected orbital and physical properties, import balanced training dataset, 
and classify the Bus-DeMeo asteroids using the methodology described in the paper.
 

"""
import os 
import pathlib
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn import preprocessing
from matplotlib import pyplot as plt
from pySankey.sankey import sankey
from specmap import *


#%% load balanced training dataset
training = pd.read_pickle('balanced_training_dataset2026_01_20_1620')
C = 64

#%% Load asteroids data
ast_loc = 'DeMeo_csv_files' #provide the location of local folder where the asteroid data is saved.
#%% Load Bus-DeMeo asteroid spectra into a dataframe, remove NANs
infolder = list(pathlib.Path(ast_loc).glob('*.csv'))
count = len(infolder)
wave_ = training.columns.tolist()#bal.columns.tolist()#
nir_start = np.where(np.array(wave_[2:]) >= 0.44)
wave =wave_[nir_start[0][0]+2:]
BDdata = np.empty([len(wave),count])
BDdata[:] = np.nan
BDdata_sm = pd.DataFrame()
BDdata_err = np.empty([len(wave),count])
ast_desig = []
count = 0
wave_col = 1
spec_col = 2
err_col = 3

for i in range(0,len(infolder)):
    path,file_name = os.path.split(infolder[i])
    temp = pd.read_csv(infolder[i],skiprows = 60,header=None)
    ast_desig.append(file_name[0:-9])
    if temp.iloc[0,wave_col] > wave[0] and  temp.iloc[-1,wave_col] <= wave[-1]:
        start_wave = np.where(wave >= temp.iloc[0,wave_col])
        end_wave = np.where(wave >= temp.iloc[-1,wave_col])
        
        sp_interp = interpolate.interp1d(temp.iloc[:,wave_col], temp.iloc[:,spec_col])
        err_interp = interpolate.interp1d(temp.iloc[:,0], temp.iloc[:,err_col])
        wave_range = wave[start_wave[0][0]:end_wave[0][0]]
        b_ = sp_interp(wave_range)
        c_ = err_interp(wave_range)
        BDdata[start_wave[0][0]:end_wave[0][0],count] = b_
        BDdata_err[start_wave[0][0]:end_wave[0][0],count] = c_
        
    if temp.iloc[0,wave_col] <= wave[0] and temp.iloc[-1,wave_col] <= wave[-1]:
        start_idx = np.where(temp.iloc[:,wave_col].round(2) >= wave[0])
        end_wave = np.where(wave >= temp.iloc[-1,wave_col])
        if start_idx[0][0] == 0:
            sp_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]:,wave_col], temp.iloc[start_idx[0][0]:,spec_col])
            err_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]:,wave_col], temp.iloc[start_idx[0][0]:,err_col])
        else:
            sp_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]-1:,wave_col], temp.iloc[start_idx[0][0]-1:,spec_col])
            err_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]-1:,wave_col], temp.iloc[start_idx[0][0]-1:,err_col])
                
        wave_range = wave[0:end_wave[0][0]-1]
        b_ = sp_interp(wave_range)
        c_ = err_interp(wave_range)
        BDdata[0:end_wave[0][0]-1,count] = b_
        BDdata_err[0:end_wave[0][0]-1,count] = c_
    
    if temp.iloc[0,wave_col] <= wave[0] and temp.iloc[-1,wave_col] >= wave[-1]: 
        start_idx = np.where(temp.iloc[:,wave_col].round(2) == wave[0])
        end_idx = np.where(temp.iloc[:,wave_col] >= wave[-1])
        if temp.iloc[start_idx[0][0],0] > wave[0]:
            sp_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]-1:end_idx[0][0]+1,wave_col], temp.iloc[start_idx[0][0]-1:end_idx[0][0]+1,spec_col])
            err_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]-1:end_idx[0][0]+1,wave_col], temp.iloc[start_idx[0][0]-1:end_idx[0][0]+1,err_col])
        else: 
            sp_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]:end_idx[0][0]+1,wave_col], temp.iloc[start_idx[0][0]:end_idx[0][0]+1,spec_col])
            err_interp = interpolate.interp1d(temp.iloc[start_idx[0][0]:end_idx[0][0]+1,wave_col], temp.iloc[start_idx[0][0]:end_idx[0][0]+1,err_col])
        
        b_ = sp_interp(wave)
        c_ = err_interp(wave)
        BDdata[:,count] = b_
        BDdata_err[:,count] = c_
        print(f'for {count}, condition 3')
    count += 1 
    print(f'count = {count}')

BDdata = pd.DataFrame(BDdata)
BDdata.columns = ast_desig

BDdata.bfill(inplace = True)
BDdata.ffill(inplace = True)
#%% Classify Bus-DeMeo asteroids using the balanced training dataset and a logistic regression classifier.

wave_ = training.columns.tolist()
nir_start = np.where(np.array(wave_[2:]) >= 0.44)
wave =wave_[nir_start[0][0]+2:]

train_spec_start_idx = 2
desig = ast_desig
test_wave = wave
test_data = BDdata
norm_wave = 1

BDpredictions = classify(training, train_spec_start_idx, desig, test_wave, test_data, norm_wave, C)
#%% Import selected characteristics of asteroids and save the classification results
bd_char_file = "C:\\Users\\maggi\\Dropbox\\Work\\Research\\specmap\\BD_asteroid_characteristics.xlsx"  
bd_char =       pd.read_excel(bd_char_file)
bd__ = pd.concat([bd_char,BDpredictions],axis = 1)

#%% Defining confidence of predictions
# this piece of code takes the pred_proba values which are generated in the classifier above
# and puts them into a new list; these values are the logistic regression's predicted probability
# that the spectrum being classified is in each group. To define a confidence of the prediction
# we can assess how much each prediction differes from the others. If it's a strong
# prediction, the standard deviation will be high: one very large value and 9 values close to zero
# If the prediction is 'weak' or less confidence, the probabilities of the groups will be similar
# and the standard deviation will be small.
bd_std = []
bd_pred_val = []
for i in range(0,bd__.shape[0]):
    bd_pred_val.append(np.max(bd__.iloc[i,19:]))
    bd_std.append(np.std(bd__.iloc[i,19:]))
    
bd_std = pd.DataFrame(bd_std)
bd_pred_val = pd.DataFrame(bd_pred_val)
bd_ = pd.concat([bd__,bd_pred_val],axis=1)
bd_ = pd.concat([bd_,bd_std],axis = 1)

label_encoder = preprocessing.LabelEncoder()

bd = bd_.sort_values(by = 'BD-complex',ascending = False)

#%% Rename groups from numerical identifiers into words and sort dataframe by spectral complex



label_encoder = preprocessing.LabelEncoder()

bd = bd_.sort_values(by = 'BD-complex',ascending = False)

bd.loc[bd['pred'] == 1,'pred'] = 'Hydrated CCs'
bd.loc[bd['pred'] == 2,'pred'] = 'CO/CV'
bd.loc[bd['pred'] == 3,'pred'] = 'CK/R/Brach'
bd.loc[bd['pred'] == 4,'pred'] = 'H/L/LL/Ure'
bd.loc[bd['pred'] == 5,'pred'] = 'EH/EL/Aub'
bd.loc[bd['pred'] == 6,'pred'] = 'ACA/LOD'
bd.loc[bd['pred'] == 7,'pred'] = 'Irons'
bd.loc[bd['pred'] == 8,'pred'] = 'HEDs'
bd.loc[bd['pred'] == 9,'pred'] ='CY'
bd.loc[bd['pred'] == 10,'pred'] = 'Primitive CCs'

#%%Create a Sankey diagram for all the endmember spectral groups: 
others = ['T', 'R', 'Q', 'L', 'K', 'D', 'A']
bd_t = bd[~bd['BD-complex'].isin(['C'])]
bd_te = bd_t[~bd_t['BD-complex'].isin(['S'])]
bd_tes = bd_te[~bd_te['BD-complex'].isin(['X'])]
bd_test = bd_tes[~bd_tes['BD-complex'].isin(['V'])]

bd_2 = pd.concat([bd_test['spec_B'],bd_test['pred']],axis = 1)

sankey(bd_2['spec_B'],bd_2['pred'],fontsize = 7)
plt.show()

#%% Create a Sankey diagram for the C-complex asteroids

bd_test = bd[bd['BD-complex'].isin(['C'])]

sankey(bd_2['spec_B'],bd_2['pred'],fontsize = 7)
plt.show()