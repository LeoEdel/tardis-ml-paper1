import os
import pandas as pd
import numpy as np
import yaml

import src.utils.load_config as load_config
from src.data_preparation.running_mean import center_running_mean as crm
from src.data_preparation.loada import loada_seq
import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# See check_forcings.ipynb 
# for further analyses of the smoothed forcings

# get config parameters
config = yaml.load(open('../../config/data_proc_demo.yaml'), Loader=yaml.FullLoader)

file_config = '../../config/data_proc_demo.yaml'
template = yaml.load(open(file_template),Loader=yaml.FullLoader)
nosit_dir, withsit_dir, forcing_adir, forcing_bdir, _, _, _ = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, lim_idm, lim_jdm, _ = load_config.load_config_params(file_config)

smooth = True
ndays = 29

if ndays % 2 == 0 and smooth:  # check running windows
    print(f'Running windows (= {ndays} days) should be odd. Aborted')
    exit()
    

for field in forcing_fields:
    print(f'Forcing field: {field}')
    df = pd.read_pickle(os.path.join(rootdir,forcing_bdir,f'{field}.pkl'))
    lrec=list(df[df.time%1==timeofday].rec)  # get list of index at 12h
    print(f'--> {len(lrec)} record found')
    idm = df.attrs['idm']
    jdm = df.attrs['jdm']
    afile = os.path.join(rootdir,forcing_adir,f'{field}.a')

    if not smooth:
        print(f'Raw data at t = {timeofday}') 
        # slice for reduced domain
        forcing_data  = loada_seq(afile, lrec, idm, jdm)[:,lim_jdm[0]:lim_jdm[1],lim_idm[0]:lim_idm[1]]
        # fd0 = forcing_data[:,0,0]  # used for plot
        savefile = os.path.join(rootdir, forcing_bdir, f'{field}.npy')
        np.save(savefile, forcing_data)
        
    else: 
        print(f'Loading binary data')    
        allrec = list(range(1, len(df.time)+1))  # 1 because binary file
        # get all dataset
        forcing_data  = loada_seq(afile, allrec, idm, jdm)[:,lim_jdm[0]:lim_jdm[1],lim_idm[0]:lim_idm[1]]
        print(f'Running mean {ndays} days')    
        forcing_mean = np.empty((len(lrec), forcing_data.shape[1], forcing_data.shape[2]))

        for i in range(forcing_data.shape[2]):  # slow and ugly
            print("row: ",i)
            for j in range(forcing_data.shape[1]):
                forcing_mean[:,j,i] = crm(forcing_data[:,j,i], ndays, npd=4)

        savefile = os.path.join(rootdir, forcing_bdir, f'{field}_mean{ndays}d.npy')
        np.save(savefile, forcing_mean)
        del forcing_mean
        
    print(f'--> save to {savefile}')
    del savefile, forcing_data


    
