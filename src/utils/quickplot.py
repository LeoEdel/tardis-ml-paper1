#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# sys.path.append(os.path.abspath("/home/leo/Climserv/utiles"))
from src.utils import modif_plot

import matplotlib
# import cmapbounds


def basemap(lons, lats, var, anta=False, plt_sh=True, savefig=False, ofile='', **kwargs):
    """
    """

    if anta:
        m = Basemap(projection='spstere', boundinglat=-60, lon_0=180, resolution='l', round=True)
    else:
        m = Basemap(projection='npstere', boundinglat=58, lon_0=0, resolution='l', round=True)

    fig = plt.figure(figsize=((1920/103)/1.5,1200/103),dpi=103)
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))

    
    cs = m.scatter(lons, lats, c=var, lw=0, latlon=True, zorder=10, **kwargs)
    cbar = plt.colorbar(cs)
    # cbar.set_label('{}'.format(label_cb))

 #    modif_plot.resize(fig, s=18)
    if savefig:
        plt.savefig(f'{ofile}')
        print(f'Figure saved as: {ofile}')
        
    if plt_sh:
        plt.show()
        
    plt.close()


def basemap_mesh(lons, lats, var, anta=False, plt_sh=True, blat=60, **kwargs):
    """
    """

    # bounds = np.delete(np.linspace(-5, 5, 21), 10)
    # bounds = np.linspace(0, 1, 11)
    # bounds, norm, cmap, ticks = cmapbounds.bds_norm(bounds=bounds, cmap=matplotlib.cm.bwr)
    # cmap.set_bad(color='grey')

    # bounds = np.linspace(224, 288, 9)
    # bounds, norm, cmap, ticks = cmapbounds.bds_norm(bounds=bounds, cmap=matplotlib.cm.jet)

    if anta:
        m = Basemap(projection='spstere', boundinglat=-blat, lon_0=180, resolution='l', round=True)
    else:
        m = Basemap(projection='npstere', boundinglat=blat, lon_0=0, resolution='l', round=True)

    fig = plt.figure(figsize=((1920/103)/1.5,1200/103),dpi=103)
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))

    nlons, nlats = m(lons, lats)  # for pcolormesh
    cs = m.pcolormesh(nlons, nlats, var, **kwargs)

#     cs = m.contourf(lons, lats, var, latlon=True, interpolation='None', **kwargs)  # , cmap=cmap, levels=bounds, **kwargs)
    cbar = plt.colorbar(cs)  # , ticks=ticks)
    cbar.set_label('(%)')
    # cbar.set_label('{}'.format(label_cb))


    modif_plot.resize(fig, s=18)
    plt.title('False Positive - False Negative (%)', y=1.03)
    # plt.title('T2m (K) ERAI-CLOUDSAT for snowfall detected by CldSat\nJanvier 2007', y=1.03)
    # plt.title('Proportion of frozen precipitation CLOUDSAT\nJanvier 2007', y=1.03)
    # plt.title('Proportion of frozen precipitation ERAI - echantillonnage CLOUDSAT\nJanvier 2007', y=1.03)
    # plt.title('T2m CLOUDSAT\nJanvier 2007', y=1.03)

    if plt_sh:
        plt.show()
        plt.close()



def basemap_empty(anta=False, plt_sh=True, **kwargs):
    """
    """

    if anta:
        m = Basemap(projection='spstere', boundinglat=-60, lon_0=180, resolution='l', round=True)
    else:
        m = Basemap(projection='npstere', boundinglat=58, lon_0=0, resolution='l', round=True)

    fig = plt.figure(figsize=((1920/103)/1.5,1200/103),dpi=103)
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.))
    m.drawmeridians(np.arange(-180.,181.,45.))

    


    # cs = m.scatter(lons, lats, c=var, lw=0, latlon=True, zorder=10, **kwargs)
    # cbar = plt.colorbar(cs)
    # cbar.set_label('{}'.format(label_cb))

    modif_plot.resize(fig, s=18)
    return fig, m
# if plt_sh:
  #       plt.show()
    #     plt.close()
