import os 
import numpy as np

from src.decomposition import ConcatPCA
from src.preproc import TFRmEvents
from src.setting import GetInfo, ExcludSubj, get_data_grp

from src.config import OUT_PATH , DATA_PATH

dataset_label = 'mf70-160'
tfr_path = DATA_PATH + '/Data_longWOBS_' + dataset_label
subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
subj_included = ExcludSubj(subj_included, data_path=tfr_path)
interesting_ev = [0, 1]

subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'SLCH018', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}

print('Subject number : ', len(subj_included))

coord, areas, elect_list, subj_list, region = GetInfo(subj_included, data_path=tfr_path)

## Group level : pseudo-subject 
data_grp = get_data_grp(subj_included, return_subj=False, data_path=tfr_path, polarity_cor=(False,0), type_data='tfr')

nb_compo=5

centered_all_concat= np.concatenate([data_grp[i, :,:,:] for i in interesting_ev], axis=2)
centered_all_mean= np.mean(data_grp, axis=0)

df_Componants_mean, df_X_transformed_mean, mean_mean = ConcatPCA({'grp' : centered_all_mean}, ch_id = False, nb_compo=nb_compo, return_mean=True)
df_Componants_concat, df_X_transformed_concat, mean_concat = ConcatPCA({'grp' : centered_all_concat}, ch_id = False, nb_compo=nb_compo, return_mean=True)

if not os.path.isdir(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_mean') :
    os.makedirs(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_mean')

if not os.path.isdir(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_concat') :
    os.makedirs(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_concat')

df_X_transformed_mean.to_csv(f'{OUT_PATH}/grpPCA_{dataset_label}/supsubj_mean/grp_mean_Xtrans_PCA{nb_compo}.csv')
df_X_transformed_concat.to_csv(f'{OUT_PATH}/grpPCA_{dataset_label}/supsubj_concat/grp_concat_Xtrans_PCA{nb_compo}.csv')

df_Componants_mean['grp'].to_csv(OUT_PATH + f'/grpPCA_mf70-160/supsubj_mean/grp_mean_Compo_PCA{nb_compo}.csv')
df_Componants_concat['grp'].to_csv(OUT_PATH + f'/grpPCA_mf70-160/supsubj_concat/grp_concat_Compo_PCA{nb_compo}.csv')

## Subject level 
concat_dict_ = {}

for subj in subj_included : 
    TFRm =TFRmEvents(subj, baseline_corr=False, data_path=tfr_path)
    concat_dict_[subj] = np.concatenate([TFRm[i, :, :,:] for i in [0, 1]], axis = 2)
    concat_dict_[subj] = np.concatenate([TFRm[i, :, :,:] for i in [0, 1]], axis = 2)

df_Componants, df_X_transformed = ConcatPCA(concat_dict_, ch_id = False, nb_compo=nb_compo)

if not os.path.isdir(f'{OUT_PATH}/subjPCA_{dataset_label}/TFR') :
    os.makedirs(f'{OUT_PATH}/subjPCA_{dataset_label}/TFR')

df_X_transformed.to_csv(f'{OUT_PATH}/subjPCA_{dataset_label}/TFR/subj_Xtrans_PCA{nb_compo}_concat.csv')
