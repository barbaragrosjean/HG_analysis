import os
import json
import warnings
import argparse
import numpy as np

from src.config import  OUT_PATH, FREQ_BAND
from src.decoding import decodingTS, LR, TemporalLR, TemporalGeneralization, TemporalLRRaw, TemporalGeneralizationRaw, CompareClassifier
from src.setting import ExcludSubj

  
def mainTS(band, pc_use, model, method_pca, data_aug_method, bs_decoding=False):
    tfr_path = OUT_PATH + '/Data_longWOBS_mf70-160'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)
    subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'SLCH018', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}
    #subj_included = subj_included_restricted[16]

    # get time 
    if bs_decoding :
        path = tfr_path + f'/{subj_included[0]}_info.json'
        with open(path) as json_data:
            d = json.load(json_data)
            if band == 'broadband':
                time = d['time_epoch']
            else : 
                time = d['time_tfr']
        
        #crop_arg = {'crop' : True, 't_id_min':time.index(np.array(time)[np.array(time) > -0.5][0]), 't_id_max' : len(time)}
        crop_arg = {'crop' : True, 't_id_min':0, 't_id_max' :time.index(np.array(time)[np.array(time) > -0.5][0]) }

    else : 
        crop_arg = {'crop' : False, 't_id_min':None, 't_id_max' : None}

    iteration = 100
    iter_perm = 1000
    nb_trials = 24
    decodingTS(band, 
            method_pca, 
            data_aug_method,
            subj_included, 
            iteration=iteration, 
            PC_use=pc_use, 
            save=True, 
            out_path=f'{OUT_PATH}/Decoding_shuffled_trials_mf70-160_perm10000', 
            iter_perm=iter_perm, 
            data_path=tfr_path, 
            model_name = model, 
            crop_arg=crop_arg, 
            n_tr=nb_trials)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specific frequency band")
    parser.add_argument("--band", 
                        type=str, 
                        choices=FREQ_BAND + ['broadband'], 
                        required=False,
                        help="Frequency band to process.")
    
    parser.add_argument("--pc_use", 
                        type=int, 
                        choices=[0, 1, 2, 3, 4, 5], 
                        required=False,
                        help="PC to use to process.")
    
    parser.add_argument("--model", 
                        type=str, 
                        choices=["LR", "SVC_linear", "SVC_rbf", "RandomForest"], 
                        required=False,
                        help="Model to run.")

    parser.add_argument("--method_pca", 
                        type=str, 
                        choices=["concat", "mean"], 
                        required=False,
                        help="PCA method to select.")
    
    parser.add_argument("--method_data_aug", 
                        type=str, 
                        choices=["mean", "duplicat"], 
                        required=False,
                        help="Method for data augmentation.")
    
    args = parser.parse_args()

    mainTS('high_gamma', 2, 'RandomForest', 'concat', 'mean', bs_decoding=False)
