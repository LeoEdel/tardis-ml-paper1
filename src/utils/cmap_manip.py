'''All functions to easily manipulate colormap
'''

import numpy as np
import matplotlib


def extract_Ncolors(Ncolors, cmap='magma'):
    '''Return X colors from the colormap
    Linearly selected between 0 and 1
    '''

    
    cmap = matplotlib.cm.get_cmap(cmap)
    return [cmap(x) for x in np.linspace(0,1,Ncolors)]
