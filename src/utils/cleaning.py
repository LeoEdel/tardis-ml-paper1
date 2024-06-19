# Contains various functions to tidy/clean folders


import glob
import os
import subprocess


def tidy_files_in_subfolder(path, extention='h5'):
     '''Put files.h5 into subfolder of the corresponding day
     
     Parameters:
     -----------
         path        :   string, path of folder to clean
         extention   :   string, extention of the files to clean
     
     '''
        
#     path = '/scratch/project_465000269/edelleo1/Leo/SIT_observations/ICESat2/'
    files = glob.glob(f'{path}*.h5')

    # put files.h5 into subfolder of the corresponding day

    for fl in files:
        str_date = os.path.basename(fl).split('_')[1][:8]  # get date=name of the folder
        folder = path+str_date
        if not os.path.exists(folder):
            os.mkdir(folder)
        subprocess.run(['mv', f'{fl}', f'{folder}'])
    