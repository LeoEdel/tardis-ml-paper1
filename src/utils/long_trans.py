#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def l180_to_360(var):
    """Convert longitude between [-180, 180] to [0, 360]
    Parameters:
    ----------
    arr, can be float, array 1D or array 2D
    """
    return var % 360

def l360_to_180(var):
    """Convert longitude between [0, 360] to [-180, 180]
    Parameters:
    ----------
    arr, can be float, array 1D or array 2D
    """
    return np.mod(var - 180., 360.) - 180
    
