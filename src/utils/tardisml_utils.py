from src.data_preparation import loada
import os
import yaml 
from src.utils import save_name

# 'zone name': [lat (j), lon (i), 'full_name']
dict_area = {'artc': [(300,629), (100,550), 'Arctic'],
        'nopo':[(700,775), (335, 415), 'North Pole'],
        'nogr':[(650,725), (350,425), 'North Greenland'],
        'bfgy':[(700,750), (250,325), 'Beaufort Gyre']
}


def define_results_dir(config, rootdir, filename, verbose=False):
    '''Define subfolder name for results (figures etc.) based on the configuration file
    
    ex:
    rf_Npred15_4F_rw21d_8N_hice00_artc
    for:
    random forest, sequence prediction of 15 days, 4 forcings input, running mean windows of 21 days, PCA with 8 components
    target variable hice00, for the full Arctic area
    '''
    
    
    config['para_var_dpd']['n_components']
    
    N_str = ''.join([str(x) for x in config['para_var_dpd']['n_components']])
    H_str = ''.join([str(x) for x in config['para_var_dpd']['History']])
    Hn_str = ''.join([str(x) for x in config['para_var_dpd']['History_neg']])
    
    
#     res_dir = f"/results/{config['ml']}_Npred{config['nseq']}_{len(config['forcing_fields'])}F_rw{config['forcing_mean_days']}d_{config['n_components']}N_{config['target_field']}"
    res_dir = f"/results/{config['ml']}_Npred{config['nseq']}_{len(config['forcing_fields'])}F_rw{config['forcing_mean_days']}d_N{N_str}_H{H_str}_Hn{Hn_str}_{config['target_field']}"
    


    # for others options that may come in a +- distant future 
    opts = ''
        
    # local / global
    _, loc = get_area_name(config['lim_idm'], config['lim_jdm'])
    opts += f'_{loc}'


    res_dir = res_dir + opts
#     print(f'Results in: {res_dir}')
    
    # check if exists
    tmp_res_dir = save_name.check_folder(os.path.join(rootdir, config['user']+res_dir))
    res_dir = f'/results/{tmp_res_dir}'
    
    # update .yaml file
    config['results_dir'] = res_dir
    # write yaml file
    with open(f'{filename}', 'w') as file:
        yaml.dump(config, file)
    
    if verbose:
        print(f'Config file updated: {filename}')
    
    # create directory on rootdir
    path = os.path.join(rootdir, config['user']+res_dir)
    if verbose:
        print(f'Results in: {path}')
    if not os.path.exists(path):
        os.mkdir(path) 
        if verbose:
            print(f'Folder created\n')    
    
    # create sub folders
    for subfolder in [config['ml_dir'], config['fig_dir']]:
#     for subfolder in [config['pca_dir'], config['ml_dir'], config['fig_dir']]:
        path = os.path.join(rootdir, config['user']+res_dir+subfolder)
        if not os.path.exists(path):
            os.mkdir(path) 
#             print(f'{subfolder} created')
    return res_dir
    

    
def define_pca_dir(config, rootdir, filename, verbose=False):    
    '''Define folder name for PCA results based on the configuration file
    Folder used to store .nc containing PCA results 
    '''

    lim_idm = config['lim_idm']
    lim_jdm = config['lim_jdm']
    
    str_xy = f"i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}"
        
#     pca_dir = f"/results/pca_{config['n_components']}N_{str_xy}"
    pca_dir = f"/results/pca_{str_xy}"
    
    
    # create directory on rootdir
    path = os.path.join(rootdir, config['user']+pca_dir)
    if verbose:
        print(f'PCA results in: {path}')
    if not os.path.exists(path):
        os.mkdir(path) 
        if verbose:
            print(f'Folder created\n')   
    
    
    
    config['pca_dir'] = pca_dir
    # write yaml file
    with open(f'{filename}', 'w') as file:
        yaml.dump(config, file)
    
    if verbose:
        print(f'Config file updated (pca_dir): {filename}')
    
    return pca_dir


def get_area_name(lim_idm, lim_jdm):
    '''
    returns name of the local area define by fixed grid limits
    todo: use a central point for each area and define limits using grid size
    so the size of the local area can be changed more easily
    
    return full_name (str), short_name (str)
    
    '''
    
    # todo: areas to be refined !!
    
    for item in dict_area.keys():
#         print(dict_area[item])
        if lim_jdm == dict_area[item][0] and lim_idm == dict_area[item][1]:  # check (lat,lon)
#             print(f'Found ! Area corresponds to {dict_area[item][2]} {item}')
            return dict_area[item][2], item  # full_name, short_name
    
    # if not found in dictionnary
    return 'Undefined', 'noname'



def get_rootdir():
    host = os.getenv('HOSTNAME')
    # print(host)
    
    if 'deep-learning-tools' in host:
      #  print('nird toolkit detected')
        rootdir = '/mnt/redda-ns2993k/'
    elif 'fram.sigma2.no' in host:
       # print('fram detected')
        rootdir = '/nird/projects/nird/NS2993K/'
        if not os.path.exists(rootdir):  # if path does not exist: it's a JOB
        #    print('Job on Fram detected')
            rootdir = '/cluster/work/users/leoede/'  # i am the only one launching job for now        
    elif 'betzy.sigma2.no' in host:
       # print('betzy detected')
        rootdir = '/nird/projects/nird/NS2993K/'
    elif 'login-2.fram.sigma2.no' in host or 'login-1.fram.sigma2.no' in host:
      #  print('Interactive job on Fram')
        rootdir = '/cluster/work/users/leoede/'  # i am the only one launching job for now        
    elif 'c' in host:  # 'c59-11', 'c60-3', 'c65-11'
       # print('Job on Fram')
        rootdir = '/cluster/work/users/leoede/'  # i am the only one launching job for now        
    elif 'hugemem' in host:  # 'c59-11', 'c60-3', 'c65-11'
       # print('Huge memory job on Fram')
        rootdir = '/cluster/work/users/leoede/'  # i am the only one launching job for now        
    elif 'uan' in host:  # uan01 uan02 uan03 = LUMI - salloc or 
        # rootdir = '/project/project_465000269/edelleo1/'
        rootdir = '/scratch/project_465000269/edelleo1/'
    elif 'nid' in host:  # JOBS on LUMI
        # rootdir = '/project/project_465000269/edelleo1/'
        rootdir = '/scratch/project_465000269/edelleo1/'
    else:
        raise AssertionError("Unknown host")
        
    return rootdir
