#!/usr/bin/env python
# -*- coding: utf-8 -*-

# mets tous les labels, titre, axes ticks et axes labels en taille X

import numpy as np
import matplotlib

import matplotlib.ticker as mticker

def check_list_of_empty_str(liste):
    '''check if the list is only filled with empty strings
    
    return True if list only contains empty strings
    '''
    
    for item in liste:
#         if type(item) is str:
#             print('moui')
        if(not item):
            continue
        else:
            return False
        
    return True


def resize(fig, s, rx=0, ry=0, do_annotation=True):
	""" passe tous les caracteres en taille 's'
	    s = scalaire
	    rx = angle de rotation des Xticklabels
	    ry =       ---"---         Yticklabels

	Pour tous les axes, change la taille des caracteres:
		label, ticklabels, title et suptitle (de la figure), annotations

	Fonctionne pour les colorbar si elles ont leurs propres axes

	NE FONCTIONNE PAS POUR LES AXES TWINX()
	mettre x labrl sur l'axe jumeau
	"""
	
	# Must draw the canvas to position the ticks
	# otherwise impossible to get_xticklabels()
	fig.canvas.draw()
	
	axes = fig.get_axes()
	for ax in axes:
	# check titre
		#print(f'title for ax: {ax.get_title()}')
		ax.set_title(ax.get_title(), fontsize = s + 3)
        
	# check label
		ax.set_xlabel(ax.get_xlabel(), fontsize = s + 2)
		ax.set_ylabel(ax.get_ylabel(), fontsize = s + 2)

	# check ticklabels
		lablx = [item.get_text() for item in ax.get_xticklabels()]
		ticks_loc = ax.get_xticks().tolist()  # to solve WarningUser: FixedLocator 
		ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))   
		ax.set_xticklabels(lablx, fontsize = s)
		
		lably = [item.get_text() for item in ax.get_yticklabels()]
		if check_list_of_empty_str(lably):
			continue    
		ticks_loc = ax.get_yticks().tolist()  # to solve WarningUser: FixedLocator 
		ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))   
		ax.set_yticklabels(lably, fontsize = s)
        
		for tick in ax.xaxis.get_major_ticks():  # ax.xaxis.get_minor_ticks()
			tick.label.set_fontsize(s)
			# set rotation: angle  
			tick.label.set_rotation(rx)  # 'vertical'
			if tick.label2:  # axe x 'top'=True
				tick.label2.set_fontsize(s)
				tick.label2.set_rotation(rx)
		
		for tick in ax.yaxis.get_major_ticks():  # ax.xaxis.get_minor_ticks()
			tick.label.set_fontsize(s) 
			tick.label.set_rotation(ry)
			if tick.label2:  # axe y 'right'=True
				tick.label2.set_fontsize(s)
				tick.label2.set_rotation(ry)
	# check annotation
		if do_annotation:
			child = ax.get_children()
			msq_anno = np.array([isinstance(children, matplotlib.text.Annotation) for children in child])
			annotation = [child[x] for x in np.where(msq_anno)[0]]  # get annotation
			for anno in annotation:
				anno.set_size(s)


	# check suptitle
	# if hasattr(fig, '_suptitle'):
	try:
		fig._suptitle.set_size(s+5) 
	except:
		pass        

	# pour savefig() sans marges blanches non necessaire
	# utiliser savefig(nom, bbox_inches = 'tight')
        
