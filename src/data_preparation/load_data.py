#################################################################
#
# Functions for loading data into python variables
#
#################################################################

import os
import numpy as np
import pickle as pkl
import netCDF4 as nc4
import xarray as xr
import pandas as pd

import src.feature_extraction.extract_pca as extract_pca
import src.utils.load_config as load_config
from src.data_preparation import mdl_dataset_prep
import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()



def load_TOPAZ(rootdir, pca_dir, file_config, data_kind, target_field, X_file, pca_file='', return_EOF=False):
    '''
    Parameters:
    ------------
    X_file     : full path to .nc file
    pca_file   : full path to .pkl file containing PCA, if given function will return PCs
    return_EOF : bool, if true, will return 2D EOF
    
    '''


    
    # --------- TOPAZ4 C ------------------

    # load X, mu, RMSE
#     filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4c.nc")
    Xf, chrono = load_nc(X_file, target_field, X_only=True)
#     chrono = pd.to_datetime(Xf['time'])
    
#     import pdb; pdb.set_trace()
    
    if len(pca_file)>2:
        pca_f = load_pca(pca_file)
        maskok = (np.isfinite(Xf)).all(dim='time')
        maskok1d = maskok.stack(z=('y','x'))
        PCs_f = extract_pca.pca_to_PC(pca_f, Xf, maskok1d)    

    # compute again because save with different indexes on dimension 'time'
        if return_EOF:
            EOF1d_f, EOF2d_f = extract_pca.compute_eof(n_components, Xf, pca_f, maskok)
            return Xf, chrono, PCs_f, EOF2d_f
        else:
            return Xf, chrono, PCs_f
            
    return Xf, chrono
    
    
def load_TOPAZ_4b_4c(file_config):
    '''Load 4b and 4c
    '''
    
    nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config, verbose=True)
    _, target_field, _, _, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)
    
    
    #X_file = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4c_2011_2019.nc")
    X_file = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b_2011_2019_FreeRun.nc")
    
#     data_kind = "nosit"
#     n_components = load_config.get_n_components(data_kind, file_config)
#     pca_file = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_noSITass_train.pkl")
    X4c, chrono4c = load_TOPAZ(rootdir, pca_dir, file_config, 'nosit', target_field, X_file=X_file)
    
    X_file = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b_2011_2019.nc")
#     data_kind = "nosit"
#     n_components = load_config.get_n_components(data_kind, file_config)
#     pca_file = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_noSITass_train.pkl")
    X4b, chrono4b = load_TOPAZ(rootdir, pca_dir, file_config, 'withsit', target_field, X_file=X_file)
    
    return X4b, chrono4b, X4c, chrono4c
    
def trunc_da_to_chrono(chrono_ref, chrono_tt, data_tt):
    '''to select only the indexes in both time series
    used to only select TOPAZ4b datetime and remove the others
    
    Parameters:
    -----------
    
    data_tt      : list of DataArray to shorten
    
    '''
    indexes = chrono_tt.date.isin(chrono_ref.date)
    indexes = indexes.to_numpy()
    
    new_data = []
        
    for var in data_tt:
        new_data.append(var[indexes])
    
    return chrono_ref, new_data    

def trunc_dict_to_chrono(chrono_ref, chrono_tt, dico_tt):
    '''to select only the indexes in both time series
    used to only select TOPAZ4b datetime and remove the others
    
    Parameters:
    -----------
    
    data_tt      : list of DataArray to shorten
    
    '''
    indexes = chrono_tt.date.isin(chrono_ref.date)
    indexes = indexes.to_numpy()
    
    new_data = {}
         
    for var in dico_tt.keys():
        new_data[var] = dico_tt[var][indexes]
    
    return chrono_ref, new_data    


def REload_dataset_PCA(config, return_raw=False, freerun=False, train=True):
    '''version 2.0: with config class
    
    Load all the data necessary for ML PCA:
    Bias SIT
    non assimilated SIT TOPAZ4c
    Covariable
    Forcings
    
    
    Parameters:
    -----------
    
    file_config     :    path to .yaml file to get config parameters
    return_raw      :    bool, if true, function will return covariables and forcings without PCA
    freerun         :    bool, if true, will return variables for TOPAZ4b Free instead of TOPAZ4c
    train           :    bool, if true, dataset objective is training ML.
                               if false, dataset objective is ML application over period not used for training
    '''
    # will be used from mdl_dataset.py
    print('Loading data...')
    
    # define here, to be return empty is return_raw == True
    dsCo = {}
    dsFo = {}

    
    # --------- TOPAZ4b Freerun (or no SIT assimilation) ------------------
    data_kind = "nosit"
    
    filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
    pca_f = load_pca(filename)

    filename = os.path.join(config.rootdir, config.pca_dir, config.nc_filenames[data_kind])
    Xf, mu_f, RMSE_f, chronof = load_nc(filename, config.target_field)


    # compute again because save with different indexes on dimension 'time'
    # to keep bcs use later in the same function
    maskok = (np.isfinite(Xf)).all(dim='time')
    maskok1d = maskok.stack(z=('y','x'))
    
    if train:
        PCs_f = extract_pca.pca_to_PC(pca_f, Xf, maskok1d)
    else:  # 'apply'
        PCs_f = pca_f
            

    # -------------------------------------
    
    # --------- TOPAZ4 err ------------------
    # to skip if load dataset for application:
    data_kind = "err"
    if not train:
        chronoe = chronof
        chrono = chronoe
        Xe = None
        PCs_e = None
        print('> Bias not loaded because dataset is used for application.')
        
        
    else:
        filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
        pca_e = load_pca(filename)

        filename = os.path.join(config.rootdir, config.pca_dir, config.nc_filenames[data_kind])
        Xe, mu_e, RMSE_e, chronoe = load_nc(filename, config.target_field)

        # retrieve PC and EOF values
        PCs_e = extract_pca.pca_to_PC(pca_e, Xe, maskok1d)

#         print('Ok err')
    

    
        chrono, [Xf, PCs_f] = trunc_da_to_chrono(chronoe, chronof, [Xf, PCs_f])

    
    # ---------- sea ice age ---------------    
    
    if config.verbose == 1:
        print('Loading sia...')
    
    data_kind = 'sia'
    
    # load nc   
    filename = os.path.join(config.rootdir, 'Leo/sia/', config.nc_filenames[data_kind])
    sia, chrono_sa = load_nc(f'{filename}', 'sia', X_only=True)
    
    # load pca
    filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
    pca_sa = load_pca(filename)
    
    # compute PCs
    if train:
#         PCs_sa = extract_pca.pca_to_PC(pca_sa, sia, maskok1d) # not working - mask does not apply well
        PCs_sa = extract_pca.pca_to_PC_isnan(pca_sa, sia)  # works well: just longer
        _, [sia, PCs_sa] = trunc_da_to_chrono(chrono, chrono_sa, [sia, PCs_sa])
        
    else:  # 'apply'
        PCs_sa = pca_sa
       
        # trunc time necessary for apply ? 
        # will be if we specify more specific period that 2000-2011
    # -------------------------------------


    # --------- covar ------------------
    
    data_kind = "co"
    PCs_co, chrono_co = load_covariables(config.covar_fields, maskok1d, config.rootdir+config.pca_dir, config.n_comp[data_kind], freerun=freerun, train=train)
    
    import pdb; pdb.set_trace()
    
    _, PCs_co = trunc_dict_to_chrono(chronoe, chrono_co, PCs_co)

    # -------------------------------------
    
    # --------- forcings ------------------
    data_kind = "fo"
    field_str = '-'.join(sorted([item.split('_')[0] for item in config.forcing_fields]))
    
    filename = os.path.join(config.rootdir, config.pca_dir, f"pca_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_2011_2022.pkl")
    # selectionne good filename
    if not train:  # 'appply'
        filename = os.path.join(config.rootdir, config.pca_dir, f"pca_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_1999_2010.pkl")
    
    

    if train:
        PCs_fo, EOFs, chrono_fo = load_forcing_PC_EOF(filename, config.forcing_bdir, config.forcing_fields)
    if not train:  # 'apply'
    # load just PC that have been extracted from EOF/PCA of 2011-2019
    # using extract_forcing_FR_2000-2010.py
        PCs_fo = pkl.load(open(filename,'rb')) 
        chrono_fo = np.load(f'{config.rootdir}{config.forcing_bdir}/chrono_forcings_1999_2010.npy')

    import pdb; pdb.set_trace()
    
    chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})
    _, PCs_fo = trunc_dict_to_chrono(chronoe, chrono_fo, PCs_fo)
    
    
    # ---------- return raw ---------------
    if return_raw:   # covar without pca
        covar_fields = [item.split('_')[0] for item in config.covar_fields]
        for covar in config.covar_fields:
#            filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4b_train.nc")
#             filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4c_2011_2019.nc")
#             if freerun:
            # for freerun ONLY
            filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4b23_2011_2022_FreeRun.nc")
    
            tmp, _ = load_nc(filename, covar, X_only=True)
            dsCo[f'{covar}'] = np.mean(tmp, axis=(1,2)).data[:, None]

        _, dsCo = trunc_dict_to_chrono(chronoe, chrono_co, dsCo)

        # forcings without pca
        dsFo, Nf, chrono_fo = load_forcing(config.forcing_fields, config.forcing_bdir, return_chrono=True)
        chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})
        #for forcing in forcing_fields:
        #    tmp = pkl.load(open(f'{rootdir}{pca_dir}/{forcing}_2011_2019.pkl','rb'))
        #    dsFo[f'{forcing}'] = np.mean(tmp, axis=(1))[:, None]
        
        _, dsFo = trunc_dict_to_chrono(chronoe, chrono_fo, dsFo)
        
    # -------------------------------------

    
    return Xf, PCs_f, Xe, PCs_e, PCs_co, PCs_fo, dsCo, dsFo, chrono, PCs_sa
    
def REREload_dataset_PCA(config, return_raw=False, freerun=False, objective='global_train', non_assimilated='freerun'):
    '''version 3.0: only load PC
    Previous version: loaded pca and compute PC from scratch >> too heavy/long
    
    Load all the data necessary for ML PCA:
    Bias SIT
    non assimilated SIT TOPAZ4c
    Covariable
    Forcings
    
    
    Parameters:
    -----------
    
    file_config     :    path to .yaml file to get config parameters
    return_raw      :    bool, if true, function will return covariables and forcings without PCA
    freerun         :    bool, if true, will return variables for TOPAZ4b Free instead of TOPAZ4c
    objective       :   string, usage of the dataset 'train'/'apply'
                                      updated to:      'global_train'      'local_train'
                                                       'global_apply'      'local_apply'
                                                       'global_apply91'      'local_apply91'
                                                       
    non_assimilated : string, to load 'adjSIC' variables or '' to load classic freerun variables
                               very similar to argument <freerun>. todo: change freerun to string to only have 1 argument
                               added later in the developement
   
   
    '''
    # will be used from mdl_dataset.py
    print('Loading data...')
    
    # define here, to be return empty is return_raw == True
    dsCo = {}
    dsFo = {}

    
    # --------- TOPAZ4b Freerun (or no SIT assimilation) ------------------
    data_kind = "nosit"
    
    # Get PC
    filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])    
    PCs_f = pkl.load(open(filename,'rb'))     
    
    # get chrono
    filename = os.path.join(config.rootdir, config.pca_dir, config.nc_filenames[data_kind])
    _, chrono = load_nc(filename, config.target_field, X_only=True)
    

    # -------------------------------------
    
    # --------- TOPAZ4 err ------------------
    # to skip if load dataset for application:
    data_kind = "err"

    if 'apply' in objective or 'apply91' in objective:
#         chronoe = chronof
#         chrono = chronoe
        Xe = None
        PCs_e = None
        print('\n> Bias not loaded because dataset is used for application.\n')
        
        
    else:
        filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
        PCs_e = pkl.load(open(filename,'rb')) 

        filename_nc = os.path.join(config.rootdir, config.pca_dir, config.nc_filenames[data_kind])
        _, chrono = load_nc(filename_nc, config.target_field, X_only=True)

        
    # 
    
    # here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # change so PCs_e 'apply' is an empty array of the same size as PCs_e when 'train'
        
        
        
    
    # ---------- sea ice age ---------------    
    
    if config.verbose == 1:
        print('\tLoading sia...')
    
    data_kind = 'sia'
    
    # load nc   
    filename = os.path.join(config.rootdir, 'Leo/sia/', config.nc_filenames[data_kind])
    sia, chrono_sa = load_nc(f'{filename}', 'sia', X_only=True)
    
    # load PC
    filename_PC = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
    PCs_sa = pkl.load(open(filename_PC,'rb')) 

       
        # trunc time necessary for apply ? 
        # will be if we specify more specific period that 2000-2011
    # -------------------------------------


    # --------- covar ------------------
    
    data_kind = "co"

    if non_assimilated == 'adjSIC':
            PCs_co, _ = load_covariables(config.covar_fields, None, config.rootdir+config.pca_dir, config.n_comp[data_kind], freerun=freerun, objective=objective, adjSIC=True)
    else:  # default
        PCs_co, _ = load_covariables(config.covar_fields, None, config.rootdir+config.pca_dir, config.n_comp[data_kind], freerun=freerun, objective=objective, adjSIC=False)  # PCs_co, chrono_co

#     _, PCs_co = trunc_dict_to_chrono(chronoe, chrono_co, PCs_co)

    # -------------------------------------
    
    # --------- forcings ------------------
    data_kind = "fo"
#     field_str = '-'.join(sorted([item.split('_')[0] for item in config.forcing_fields]))

    PCs_fo = {}
    
    # loop on all forcings to get separate files
    for forcing_field in config.forcing_fields:
        
        field_str = forcing_field.split('_')[0]
        
        ifile = f'PC_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_2011_2022.pkl'
        if 'apply91' in objective:
            ifile = f'PC_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_1991_1998.pkl'
        elif 'apply' in objective:
            ifile = f'PC_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_1999_2010.pkl'


        filename = os.path.join(config.rootdir, config.pca_dir, ifile)    
        tmp_fo = pkl.load(open(f'{filename}','rb'))  
        
        for key, value in tmp_fo.items():
            PCs_fo[key] = value
        

#     PCs_fo = pkl.load(open(f'{filename}','rb')) 
    

    
    
    
    chrono_fo = np.load(f'{config.rootdir}{config.forcing_bdir}/chrono_forcings_2011_2022.npy')
    if 'apply91' in objective:
        chrono_fo = np.load(f'{config.rootdir}{config.forcing_bdir}/chrono_forcings_1991_1998.npy')
    elif 'apply' in objective:
        chrono_fo = np.load(f'{config.rootdir}{config.forcing_bdir}/chrono_forcings_1999_2010.npy')  
    
    # ---------- return raw ---------------
    if return_raw:   # covar without pca
        covar_fields = [item.split('_')[0] for item in config.covar_fields]
        for covar in config.covar_fields:
#            filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4b_train.nc")
#             filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4c_2011_2019.nc")
#             if freerun:
            # for freerun ONLY
            filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4b23_2011_2022_FreeRun.nc")
    
            tmp, _ = load_nc(filename, covar, X_only=True)
            dsCo[f'{covar}'] = np.mean(tmp, axis=(1,2)).data[:, None]

#         _, dsCo = trunc_dict_to_chrono(chronoe, chrono_co, dsCo)

        # forcings without pca
        dsFo, Nf, chrono_fo = load_forcing(config.forcing_fields, config.forcing_bdir, return_chrono=True)
        chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})
        #for forcing in forcing_fields:
        #    tmp = pkl.load(open(f'{rootdir}{pca_dir}/{forcing}_2011_2019.pkl','rb'))
        #    dsFo[f'{forcing}'] = np.mean(tmp, axis=(1))[:, None]
        
#         _, dsFo = trunc_dict_to_chrono(chronoe, chrono_fo, dsFo)
        
    # -------------------------------------

    # same chrono for all data, and no need for Xe, Xf (mean SIT bias and mean SIT freerun)
    return None, PCs_f, None, PCs_e, PCs_co, PCs_fo, dsCo, dsFo, chrono, PCs_sa


def load_dataset_PCA(file_config, return_raw=False, freerun=False):
    '''Load all the data necessary for ML PCA:
    Bias SIT
    non assimilated SIT TOPAZ4c
    Covariable
    Forcings
    
    
    Parameters:
    -----------
    
    file_config     :    path to .yaml file to get config parameters
    return_raw      :    bool, if true, function will return covariables and forcings without PCA
    freerun         :    bool, if true, will return variables for TOPAZ4b Free instead of TOPAZ4c
    '''
    
    
    
    
    nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config, verbose=True)
    timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)
    
    
    # chronology 
#    chrono = np.load(os.path.join(rootdir, forcing_bdir, f'chrono_forcings_2011_2019.npy'))
 #   chrono = pd.to_datetime(chrono)
    

    
    # --------- TOPAZ4 C ------------------
    data_kind = "nosit"
    n_components = load_config.get_n_components(data_kind, file_config)
    filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_noSITass_2011_2019.pkl") # HERE
    if freerun:
        filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_noSITass_2011_2019_FreeRun.pkl")
    pca_f = load_pca(filename)
    
    # load X, mu, RMSE
    filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4c_2011_2019.nc") # HERE
    if freerun:
        filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b_2011_2019_FreeRun.nc")
    Xf, mu_f, RMSE_f, chronof = load_nc(filename, target_field)
    # compute again because save with different indexes on dimension 'time'
    maskok = (np.isfinite(Xf)).all(dim='time')
    maskok1d = maskok.stack(z=('y','x'))
    PCs_f = extract_pca.pca_to_PC(pca_f, Xf, maskok1d)

#     EOF1d_f, EOF2d_f = extract_pca.compute_eof(n_components, Xf, pca_f, maskok)
    
    # Reconstruction from PCA
#     Xf_rec = xr.dot(EOF2d_f,PCs_f) + mu_f
    # RMSE for comparison with RMSE predicted by ML
#     RMSE_recf = np.sqrt((np.square(Xf_rec-Xf)).mean(dim='time'))
    # -------------------------------------
    
    
    # --------- TOPAZ4 err ------------------
    data_kind = "withsit"
    n_components = load_config.get_n_components(data_kind, file_config)
    filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_SITerr_2011_2019.pkl")
    if freerun:
        filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_SITerr_2011_2019_FreeRun.pkl")
    pca_e = load_pca(filename)
    
    # load X, mu, RMSE
    filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4err_2011_2019.nc")
    if freerun:
        filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4err_2011_2019_FreeRun.nc")
    Xe, mu_e, RMSE_e, chronoe = load_nc(filename, target_field)
    
    # retrieve PC and EOF values
    PCs_e = extract_pca.pca_to_PC(pca_e, Xe, maskok1d)
    
#     EOF1d_e, EOF2d_e = extract_pca.compute_eof(n_components, Xe, pca_e, maskok)
#     # Reconstruction from PCA
#     Xe_rec = xr.dot(EOF2d_e,PCs_e) + mu_e
#     # RMSE for comparison with RMSE predicted by ML
#     RMSE_rece = np.sqrt((np.square(Xe_rec-Xe)).mean(dim='time'))
    # -------------------------------------
    
    # keep only chronoe :
 
    # get common indexes between chrono err and chrono non-assimilated
#    indexes = chronof.date.isin(chronoe.date)
#    indexes = indexes.to_numpy()
#    chrono = chronof.merge(chronoe)  # only keep common times between 2 versions of TOPAZ
#    
#    Xf = Xf[indexes]
#    PCs_f = PCs_f[indexes]
    
    chrono, [Xf, PCs_f] = trunc_da_to_chrono(chronoe, chronof, [Xf, PCs_f])

    
    # --------- covar ------------------
    
    data_kind = "covariable"
    n_components = load_config.get_n_components(data_kind, file_config)
    PCs_co, chrono_co = load_covariables(covar_fields, maskok1d, pca_dir, n_components, freerun=freerun)
    _, PCs_co = trunc_dict_to_chrono(chronoe, chrono_co, PCs_co)
    
    # -------------------------------------
    
    # --------- forcings ------------------
    data_kind = "forcing"
    filename = load_config.get_pca_filename(data_kind, file_config, pca_dir)
    PCs_fo, EOFs, chrono_fo = load_forcing_PC_EOF(filename, forcing_bdir, forcing_fields)
    chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})
    _, PCs_fo = trunc_dict_to_chrono(chronoe, chrono_fo, PCs_fo)
    
    
#    print('Chrono returned from "chrono_forcings.npy"')
#    print('Need to return chrono for each sub dataset')

    
    
    # ---------- return raw ---------------
    if return_raw:
            # covar without pca
        dsCo = {}
        covar_fields = [item.split('_')[0] for item in covar_fields]
        for covar in covar_fields:
#            filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4b_train.nc")
            filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4c_2011_2019.nc")
            if freerun:
                filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4b_2011_2019_FreeRun.nc")
    
            tmp, _ = load_nc(filename, covar, X_only=True)
            dsCo[f'{covar}'] = np.mean(tmp, axis=(1,2)).data[:, None]

        _, dsCo = trunc_dict_to_chrono(chronoe, chrono_co, dsCo)



        # forcings without pca
        # dsFo = {}
        dsFo, Nf, chrono_fo = load_forcing(forcing_fields, forcing_bdir, return_chrono=True)
        chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})
        #for forcing in forcing_fields:
        #    tmp = pkl.load(open(f'{rootdir}{pca_dir}/{forcing}_2011_2019.pkl','rb'))
        #    dsFo[f'{forcing}'] = np.mean(tmp, axis=(1))[:, None]
        
        _, dsFo = trunc_dict_to_chrono(chronoe, chrono_fo, dsFo)
        
    # -------------------------------------
    
    
        return Xf, PCs_f, Xe, PCs_e, PCs_co, PCs_fo, dsCo, dsFo, chrono
        
    return Xf, PCs_f, Xe, PCs_e, PCs_co, PCs_fo, chrono
    
    
    
def get_fn_covar_pkl(covar, n_components, freerun=True, objective='global_train', adjSIC=False):
    '''return correct filename in order to load .pickle (containing PCA)
    
    adjSIC    : bool, to use SIC and SIT corrected with SIC > 15%
    '''

    file_name = None
    
    if 'train' in objective:
        file_name = f'PC_{covar}_{n_components}N_2011_2022_FreeRun.pkl'
        if adjSIC and (covar in 'sithick' or covar in'siconc'):
            file_name = f'PC_{covar}_{n_components}N_2011_2022_FreeRun_adjSIC.pkl'
        elif freerun == False:  # unused until now
            file_name = f'PC_{covar}_{n_components}N_2011_2022_noSIT.pkl'

    elif 'apply91' in objective:
        file_name = f'PC_{covar}_{n_components}N_1991_1998_FreeRun.pkl'
    elif 'apply' in objective:
        file_name = f'PC_{covar}_{n_components}N_1999_2010_FreeRun.pkl'

    return file_name


def get_fn_covar_nc(covar, freerun=True, objective='', train=None, apply91=False):
    '''return correct filename in order to load .netcdf
    
    Old default:
        objective='global_train'
    
    '''

    file_name = None

    if 'train' in objective or train==True:
        file_name = f'{covar}_TOPAZ4b23_2011_2022_FreeRun.nc'      
       # if freerun == False:  # unused until now
        #    file_name = f'{covar}_TOPAZ4b23_1999_2010_NoSIT.nc'

    elif 'apply91' in objective or (train==False and apply91==True):
        file_name = f'{covar}_TOPAZ4b23_1991_1998_FreeRun.nc'
    elif 'apply' in objective or (train==False and apply91==False):
        file_name = f'{covar}_TOPAZ4b23_1999_2010_FreeRun.nc'
    
    return file_name



def load_covariables(covar_fields, maskok1d, pca_dir, n_components, freerun=False, objective='', adjSIC=False):
    """Load covariables
    In:
        covar_fields : list of covariables to load
        maskok1d     : land ocean mask in 1d array
        pca_dir      : directory to look for .pkl and .nc files
        n_components : number of components in PCA
        freerun      : bool, if true, will load covariables from TOPAZ4b Free Run instead of TOPAZ4c
    
    Out:
        PCS_co       : dictionary with the PC values for each covariable
    """
    
#     pca_co = dict()
    PCs_co = dict()
    
    for covar in covar_fields:
        print(f'\tRetrieve {covar}')
        # determine .pkl file
        filename_pkl = get_fn_covar_pkl(covar, n_components, freerun=freerun, objective=objective, adjSIC=adjSIC)
#         pca_co[covar] = load_pca(f'{pca_dir}/{filename_pkl}')  # get pca with .pkl
        PCs_co[covar] = pkl.load(open(f'{pca_dir}/{filename_pkl}','rb')) 
        
#         filename_nc = get_fn_covar_nc(covar, freerun=freerun, train=train)
#         Xco, chrono_co = load_nc(f'{pca_dir}/{filename_nc}', covar, X_only=True)  # load .nc
    
        # retrieve PC 
#         if train:
#             Xco1d_nonan = Xco.stack(z=('y','x')).where(maskok1d, drop=True)
#             PCs_co[covar] = xr.DataArray(pca_co[covar].transform(Xco1d_nonan), dims=['time','comp'])
#         else:  # save as PC already (because used 2011-2019 to extract PCA)
#             PCs_co[covar] = pca_co[covar]
        
#     Nco = len(covar_fields)
        
    return PCs_co, None  # chrono_co


def load_PC_forcing(filename, forcing_bdir, forcing_fields):
    '''Load PC of forcing_fields (saved as .pkl)
    '''
    
    PCs_fo = dict()
    
    PCs_fo = pkl.load(open(f'{pca_dir}/{filename_pkl}','rb')) 
    
    for forc in forcing_fields:
        print(f'\tRetrieve {forc}')
        PCs_fo[forc] = PC[forc]
    
    

def load_forcing_PC_EOF(filename, forcing_bdir, forcing_fields):
    '''Load forcings, their PCA
    and retrieve PCs and EOFs values
    '''
    print('Loading forcing values...')
    forcings, Nf, chrono_fo = load_forcing(forcing_fields, forcing_bdir, return_chrono=True)
    print('Loading pca...')
    pca = load_pca(filename)  # load pca for forcings
    
    
    # treat forcings: invers + apply mask
    # --------------- DATA TREATMENT FROM run_pca_forcings.py --------------
    # load mask
#     maskok = load_land_mask((100,550), (300,629), rootdir, 'Leo/results/pca_i100-550_j300-629')
    maskok = load_land_mask((100,550), (150,629), rootdir, 'Leo/results/pca_i100-550_j150-629')
    
    maskok1d = maskok.stack(z=('y','x'))
    # maskok3d = maskok.expand_dims({'time':forcings[forcing_fields[0]].shape[0]})

    # for each forcing:
    # inverse latitude
    for forcing in forcing_fields:
         forcings[forcing][:] = forcings[forcing][:][::-1]

    # ---------- selection ntrain ---------- 
#    ntrain, nval, ntest = mdl_dataset_prep.dataset_split(forcings[forcing_fields[0]].shape[0])
#    suffix = '_train'
#    for forcing in forcing_fields:
#        forcings[forcing] = forcings[forcing][ntest+nval:]        

    forcings2d = forcings.copy()
    # stack lat and lon dimensions
    # apply mask to exclude values not over sea-ice
    # ---------- apply mask ---------- 
    print('Apply land/ocean mask...')
    for forcing in forcing_fields:
        tmp2D = xr.DataArray(forcings2d[forcing].reshape(forcings2d[forcing].shape[0], -1), dims=('time', 'z'))
        tmp2D_nonan = tmp2D.where(maskok1d, drop=True)
        forcings2d[forcing] = tmp2D_nonan # .to_numpy()
    
    # -------------------------------------------------------------------------
#     import pdb;pdb.set_trace()
    
#     extract_pca.compute_pca_forcings(Nf, forcing_fields, forcings)
    print('Retrieve PCs and EOFs')
    PCs = dict()
    EOF2d = dict()
    for forcing in forcing_fields:
        X = forcings2d[forcing]
#         PCs[forcing] = extract_pca.pca_to_PC(pca, X, maskok1d)
        PCs[forcing] = xr.DataArray(pca[forcing].transform(X), dims=['time','comp'])
#         _, EOF2d[forcing] = extract_pca.compute_EOF_2d(Nf, X, pca[forcing], maskok1d)
    
    
    # retrieve PC and EOF for values using pca
#     print('Retrieve PCs and EOFs')
#     PCs = dict()
#     EOFs = dict()
#     for field in forcing_fields:
#         X = forcings[field].reshape(forcings[field].shape[0],-1)
#         PCs[field] = pca[field].transform(X)
#         EOFs[field] = pca[field].components_

    return PCs, EOF2d, chrono_fo

def load_forcing(forcing_fields, forcing_bdir, return_chrono=False, train=True, apply91=False):
    """ Load data for several forcing in a variable
    Parameters:
    -----------
        forcing_fields : String list -- Names of forcings
        forcing_bdir   : String      -- Path where forcing data is stored
        return_chrono  : bool, load time serie
        train          : bool, dataset loading for application (2000-2011) 
                                                or training (2011-2019)
        apply91        : bool, added later in the development. if True, load forcing for period 1991-1998
        
    Out:
        forcings : dict of numpy.ndarray -- Store data for each forcing (key = forcing name)
        Nf       : int                   -- Number of forcing fields
    """
    
    # succession of if a bit ugly, bit it ended like this, sorry.
    
    forcings = dict()
    for field in forcing_fields:  # [:1]
        if train:
            forcings[field] = np.load(os.path.join(rootdir, forcing_bdir, f'{field}_2011_2022.npy'))  # 2011_2019
        if not train:
            if not apply91:
                print(f'Load forcings {field} between 1999-2010')
                forcings[field] = np.load(os.path.join(rootdir, forcing_bdir, f'{field}_1999_2010.npy'))
            else:
                print(f'Load forcings {field} between 1991-1998')
                forcings[field] = np.load(os.path.join(rootdir, forcing_bdir, f'{field}_1991_1998.npy'))
            
    Nf = len(forcing_fields)
    
#     chrono = np.load('/nird/projects/nird/NS2993K/Leo/forcings_full/chrono_forcings.npy')
    chrono = np.load(os.path.join(rootdir, forcing_bdir, f'chrono_forcings_2011_2022.npy'))  # 2011_2019
    if not train:
        if not apply91:
            chrono = np.load(os.path.join(rootdir, forcing_bdir, f'chrono_forcings_1999_2010.npy'))
        else:
            chrono = np.load(os.path.join(rootdir, forcing_bdir, f'chrono_forcings_1991_1998.npy'))
            
    
    if return_chrono:
        return forcings, Nf, chrono
    
    return forcings, Nf



def load_pca(filename, verbose:int=0):
    """Load PCA
    """
    if verbose==1: print(f'Load PCA: {filename}\n')
    pca = pkl.load(open(filename,'rb')) 
    return pca



def load_nc(filename, target_field, X_only=False):
    """load .netcdf
    returns X, mu, RMSE
    returns X only if arg 'X_only' is True
    
    
    """
    nc = nc4.Dataset(filename, mode='r')
    X = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))[target_field]
    chrono = pd.DataFrame({'date':pd.to_datetime(X['time'].to_numpy())})
    
    if X_only:
        return X, chrono
    
    mu = X.mean(dim='time').compute()  # mean for EOF reconstruction
    RMSE = np.sqrt((np.square(X)).mean(dim='time')) # RMSE_a for comparison avec RMSEa_est(imated) (predicted by ML)
     

       
    return X, mu, RMSE, chrono


def load_land_mask(lim_idm, lim_jdm, rootdir, pca_dir):
    '''Return land mask for a specific area delimited by lim_idm and lim_jdm
    
    For new version of TOPAZ
    '''
    
    # get filename for corresponding area
    str_xy = f"i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}"
    
    # open file in PCA in subfolder 
    filename = os.path.join(rootdir,pca_dir,f"land_mask_{str_xy}.nc")
    
    nc = nc4.Dataset(filename, mode='r')
    maskok = xr.DataArray(nc['sithick'][:]>0, dims=('y', 'x'))
    
    return maskok

def get_land_mask(lim_idm, lim_jdm, rootdir, pca_dir):
    '''Return land mask for a specific area delimited by lim_idm and lim_jdm
    
    Out:
        1d array (stack on jdim, idim) of the land mask
        2d land mask
    
    if stacker = False, return 2d array (jdim, idim)
    pca_dir, path string, must NOT contain '/' at the beginning or will be considered absolute path
    
    '''
    
    if pca_dir[:1] == '/':
        print('WARNING: path will be considered absolute')
        pca_dir = pca_dir[1:]
    
    # get filename for corresponding area
    str_xy = f"i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}"
    
    # open file in PCA in subfolder 
    filename = os.path.join(rootdir,pca_dir,f"land_mask_{str_xy}.nc")
    nc = nc4.Dataset(filename, mode='r')
    maskok = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
    # land mask does not change as a function of variable, so we just get it using the 1st one
    data = maskok[list(maskok.keys())[0]]  

    return data.stack(z=('jdim','idim')), data  # mskok1d





def load_dataset_nc(file_config, freerun=True, train=True, verbose=0):
    '''Load all the data necessary for ML:
    Bias SIT
    non assimilated SIT TOPAZ4c
    Covariable
    Forcings
    
    sithick_TOPAZ4b23_2011_2022_FreeRun.nc
    Parameters:
    -----------
    
    file_config     :    path to .yaml file to get config parameters
    freerun         :    bool, if true, will return variables for TOPAZ4b FreeRun instead of TOPAZ4c
    train           :    bool, if true, dataset objective is training ML.
                               if false, dataset objective is ML application over period not used for training
    '''
    
    
    
    
    nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config, verbose=True)
    timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)
    
    
    # chronology 
#    chrono = np.load(os.path.join(rootdir, forcing_bdir, f'chrono_forcings_2011_2019.npy'))
 #   chrono = pd.to_datetime(chrono)

   # --------------------
   # load mask to add in Conv2D

    maskok = load_land_mask((100,550), (300,629), rootdir, 'Leo/results/pca_i100-550_j300-629')
 
    # --------- TOPAZ4 C ------------------
    data_kind = "nosit"
    n_components = load_config.get_n_components(data_kind, file_config)
    
    # filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4c_2011_2019.nc") # HERE
    filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b23_2011_2022_FreeRun.nc")
    if not train:
        filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b_FR_2000_2011.nc")
    Xf, chronof = load_nc(filename, target_field, X_only=True)
    # -------------------------------------
    
    
    # --------- TOPAZ4 err ------------------
    data_kind = "withsit"
    n_components = load_config.get_n_components(data_kind, file_config)
    
    #filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4err_2011_2019.nc")
    filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4err23_2011_2022_FreeRun.nc")
#     filename = os.path.join(rootdir,pca_dir,f"{target_field}_TOPAZ4b_2011_2019.nc")  # just for debug Conv2D
    if train:
        Xe, chronoe = load_nc(filename, target_field, X_only=True)
    else:
        chronoe = chronof  # use TOPAZ4b FR when not in training
#         chrono = chronoe
        Xe = None
    
    # -------------------------------------
    
    # keep only chronoe :    
    chrono, [Xf] = trunc_da_to_chrono(chronoe, chronof, [Xf])

    # --------- covar ------------------
#     dsCo = {} # old way
#     covar_fields = [item.split('_')[0] for item in covar_fields]
#     for covar in covar_fields:
# #            filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4c_2011_2019.nc")
#         filename = os.path.join(rootdir,pca_dir,f"{covar}_TOPAZ4b_2011_2019_FreeRun.nc")

#         tmp, chrono_co = load_nc(filename, covar, X_only=True)
#         dsCo[f'{covar}'] = tmp  # np.mean(tmp, axis=(1,2)).data[:, None]

#     _, dsCo = trunc_dict_to_chrono(chronoe, chrono_co, dsCo)
    
    
    # new way -> import also data from 2000-2011
    dsCo = {}
    for covar in covar_fields:
        if verbose == 1: print(f'\tRetrieve {covar}')
        
        filename_nc = get_fn_covar_nc(covar, freerun=freerun, train=train)
        tmp, chrono_co = load_nc(f'{rootdir}{pca_dir}/{filename_nc}', covar, X_only=True)  # load .nc
        dsCo[f'{covar}'] = tmp
    
    _, dsCo = trunc_dict_to_chrono(chronoe, chrono_co, dsCo)
    
    # -------------------------------------

    # --------- forcings ------------------

    # forcings without pca
    dsFo, Nf, chrono_fo = load_forcing(forcing_fields, forcing_bdir, return_chrono=True, train=train)
    chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})

    _, dsFo = trunc_dict_to_chrono(chronoe, chrono_fo, dsFo)        
    # -------------------------------------
    
    # ---------- sea ice age ---------------
    
    
    if verbose == 1:
        print('Loading sia...')
    
    data_kind = 'sia'
    
    # load nc   
    filename = os.path.join('/scratch/project_465000269/edelleo1/', 'Leo/sia/', 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc')  # config.nc_filenames[data_kind])
    sia, chrono_sa = load_nc(f'{filename}', 'sia', X_only=True)
    
##     load pca
##     filename = os.path.join(config.rootdir, config.pca_dir, config.pkl_filenames[data_kind])
##     pca_sa = load_pca(filename)

    
    
#     # compute PCs
# #     if train:
# #         PCs_sa = extract_pca.pca_to_PC(pca_sa, sia, maskok1d) # not working - mask does not apply well
# #         PCs_sa = extract_pca.pca_to_PC_isnan(pca_sa, sia)  # works well: just longer
#     _, [sia] = trunc_da_to_chrono(chrono, chrono_sa, [sia])
        
# #     else:  # 'apply'  # not try yet
# #         PCs_sa = pca_sa
    
        
    return Xf, Xe, dsCo, dsFo, chrono, sia, maskok
#     return Xf, Xe, dsCo, dsFo, chrono, maskok


def import_covar_nc(covar_fields, chrono_ref, rootdir, pca_dir, freerun=True, train=True, apply91=False,
                    verbose=0):
    '''
    
    Parameters:
    -----------
    
        covar_fields        :    list of string, covariables from TOPAZ freerun to import
        chrono_ref          :    pandas.DataFrame, time variable  
    '''
        # new way -> import also data from 2000-2011
    dsCo = {}
    for covar in covar_fields:
        if verbose == 1: print(f'\tRetrieve {covar}')
    
        filename_nc = get_fn_covar_nc(covar, freerun=freerun, train=train, apply91=apply91)
        if verbose == 1: print(f'\tLoading {filename_nc}')
        tmp, chrono_co = load_nc(f'{rootdir}{pca_dir}/{filename_nc}', covar, X_only=True)  # load .nc
        dsCo[f'{covar}'] = tmp
    
    _, dsCo = trunc_dict_to_chrono(chrono_ref, chrono_co, dsCo)

    return dsCo, chrono_co

def import_forcing_nc(forcing_fields, chrono_ref, rootdir, freerun=True, train=True, apply91=False,
                      verbose=0):
    '''
    Return dictionary of forcings in 'forcing_fields'
    Trunc forcing to chrono_ref
    
    Parameters:
    -----------
    
        forcing_fields        :    list of string, ERA5 forcing to import
    
    '''
    # forcings without pca
    forcing_bdir = rootdir + 'Leo/forcings_2023'
    dsFo, Nf, chrono_fo = load_forcing(forcing_fields, forcing_bdir, return_chrono=True, train=train, apply91=apply91)
    chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})

    _, dsFo = trunc_dict_to_chrono(chrono_ref, chrono_fo, dsFo) 

    return dsFo, chrono_fo

def dico_np2xr(dico, coo1, coo2, coo3):
    '''Convert a dictionnary containing 3D ndarray to 3D xr.DataArray
    
    Parameters:
    ------------
        dico      : dictionnary containing 3D ndarray (i.e. forcings)
        coo1      : coords 1:time, DataArray of dim 1
        coo2      : coords 2:y, DataArray of dim 1
        coo3      : coords 3:x, DataArray of dim 1
    '''

    new_dico = dict()
    for ky in dico:
        new_dico[ky] = xr.DataArray(dico[ky], coords={'time':coo1, 'y':coo2, 'x':coo3}, dims=['time','y','x'])
    
    return new_dico

def load_dataset_g2l(config, freerun=True, gfile=None):
    '''Load dataset from GLOBAL to LOCAL approaches
    
    Load (without PCA):
            - bias between ML - TOPAZ4b (outputs of GLOBAL correction)
            - covar
            - forcings
            - landmask
            
            
    Parameters:
    -----------
        config         : config class
        freerun        : bool, needed for loading corresponding covariables
        gfile          :    path to .nc file containing SIT corrected by GLOBAL method
                                     default empty, will use default ML algo (RF)
    
        
    '''

    
    #print('todo: dans load_data.py, ajoutez argument: filename for path and file to .nc file to global-corrected SIT')
    
    # load residual bias after global correction:
    # filename = '/cluster/work/users/leoede/Leo/results/rf_221216-141433/ml/sit_ml_2011_2019.nc'
    if gfile is None:
        gfile = f'{rootdir}Leo/results/rf_221216-141433/ml/sit_ml_2011_2019.nc'
    
    print(f'Loading global residual bias from: {gfile}')
    
    Xe, chrono = load_nc(gfile, 'bias_ml', X_only=True)
    Xna, _ = load_nc(gfile, 'sit_na', X_only=True)
    
    # --------------------------------------------------------------------------
    # data cut because RF (any algo) does not predict the last 14 timesteps
    
    # a little bit more complicated than necessary but should work
    # get mean along time axis
    # when NaN: means that the prediction is over: we cut the data there
    
    # better way would be to put the end date of the prediction in the .nc and retrieve it
    # get 'last_prediction' and get its index in chrono

    nc = nc4.Dataset(gfile, mode='r')
    if hasattr(nc, 'last_prediction'):  # for the files created after 16-02-2023
        lvp = nc.last_prediction  # last_valid_pred
        idx_cut = np.where((chrono==lvp).to_numpy()==True)[0][0] + 1 
        # +1 because last prediction is kept 
    else:
        # first index of nan value 
        idx_cut = np.where(np.isnan(Xe.mean(('y','x')) ))[0][0]
        
    print(f'Indexes after <<<{idx_cut}>>> will be removed: not predicted by ML.')
    
    Xe = Xe[:idx_cut]  # -14]
    chrono = chrono[:idx_cut]  # -14]
    Xna = Xna[:idx_cut]  # -14]
    # --------------------------------------------------------------------------
    

    # covar without pca
    dsCo = {}
    covar_fields = [item.split('_')[0] for item in config.covar_fields]
    for covar in config.covar_fields:
        #filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4c_2011_2019.nc")
        #if freerun:
        #    filename = os.path.join(config.rootdir, config.pca_dir,f"{covar}_TOPAZ4b_2011_2019_FreeRun.nc")

        filename_nc = get_fn_covar_nc(covar, freerun=freerun, train=True)
        filename = os.path.join(config.rootdir, config.pca_dir, filename_nc)
            
        # tmp, _ = load_nc(filename, covar, X_only=True)
        # no need for mean
        #dsCo[f'{covar}'] = np.mean(tmp, axis=(1,2)).data[:, None]
        dsCo[f'{covar}'], chrono_co = load_nc(filename, covar, X_only=True)
        
        
    _, dsCo = trunc_dict_to_chrono(chrono, chrono_co, dsCo)



#     # forcings without pca
    dsFo, Nf, chrono_fo = load_forcing(config.forcing_fields, config.forcing_bdir, return_chrono=True)
    chrono_fo = pd.DataFrame({'date':pd.to_datetime(chrono_fo)})

    _, dsFo = trunc_dict_to_chrono(chrono, chrono_fo, dsFo)
    
    # Converting forcings to xr.DataArray
    dsFo = dico_np2xr(dsFo, Xe['time'], Xe['y'], Xe['x'])
    
    
    # landmask
    maskok = load_land_mask(config.lim_idm, config.lim_jdm, config.rootdir, config.pca_dir)
    # maskok1d = maskok.stack(z=('y','x'))  # if necessary
    
    
    # -------- load sia ---------------------------
    
    print('Load sia')
    
    filename = os.path.join('/scratch/project_465000269/edelleo1/', 'Leo/sia/', 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc')  # config.nc_filenames[data_kind])
    sia, chrono_sa = load_nc(f'{filename}', 'sia', X_only=True)

    
    # compute PCs
#     if train:
#         PCs_sa = extract_pca.pca_to_PC(pca_sa, sia, maskok1d) # not working - mask does not apply well
#         PCs_sa = extract_pca.pca_to_PC_isnan(pca_sa, sia)  # works well: just longer
    _, [sia] = trunc_da_to_chrono(chrono, chrono_sa, [sia])
    
    
    return Xe, chrono, Xna, dsCo, dsFo, maskok, sia
    
    
