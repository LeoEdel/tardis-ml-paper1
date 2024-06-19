#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime

def check(path, name):
    """Return a new filename if file already exists (path not included)
    ex:
        mangetoi.file       exists in /dir/there/here/
        mangetoi.file       will be saved as:
        mangetoi_01.ffile
            "   _02.f
            "   _03.f          etc  
    """
    while os.path.exists('{}{}'.format(path, name)) == True:
        name_ori, ext = os.path.splitext(name)
        if name_ori[-2:].isdigit():
            num = int(name_ori[-2:])+1
            name = '{}{:02d}{}'.format(name_ori[:-2], num, ext)
        else:
            name = '{}_01{}'.format(name_ori, ext)
    
    return name



def check_folder(path):
    '''Return a new filename if folder already exists (path not included)
    ex:
        result/       exists in /dir/there/here/
        result/       will be saved as:
        result1_220816-191856/
    '''
    folder_only = os.path.basename(os.path.normpath(path))

    if os.path.exists('{}'.format(path)) == True:
        tdate = datetime.now().strftime('%y%m%d-%H%M%S')
        path = '{}_{}'.format(folder_only, tdate)
    else:
        path = folder_only

    return path