"""common utils functions for machine learning models
"""

import pprint

def clear_output_txt(filename):  # clear file
    '''Clear .txt file
    '''
    with open(filename, 'w') as f:
        f.write('')
        
        
def add_output_2txt(filename, cap):  # save caption to text file
    '''Add string to .txt file
    '''
    with open(filename, 'a') as f:
        f.write(cap)
        
def add_pprint_2txt(filename, cap):   # save caption using pprint module
    '''Add pprint to save strings/dict to .txt file
    '''
    with open(filename, 'a') as f:
        pprint.pprint(cap, stream=f)
    