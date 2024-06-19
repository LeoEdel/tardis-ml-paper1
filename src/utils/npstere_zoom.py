#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.basemap import Basemap


def mzoom(lon_w=-10, lon_e=15, lat_s=40, lat_n=60):
    """Returns a Basemap object (NorthPoleStere) focused in a region.

    lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
                                  W and S directions are negative.
    http://code.activestate.com/recipes/578379-plotting-maps-with-polar-stereographic-projection-/

    Europe : m = npstere_zoom.mzoom(lon_w=-30, lon_e=30, lat_s=35, lat_n=70)
    Greenland : m = npstere_zoom.mzoom(lon_w=-60, lon_e=-20, lat_s=58.75, lat_n=85)
    """
    lon_0=lon_w + (lon_e - lon_w) / 2.
    lat_0=lat_n
    #
    m = Basemap(projection='npstere', lon_0=lon_0, boundinglat=0, resolution='l')
    #
    lonzm = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
    latzm = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
    xzm, yzm = m(lonzm, latzm)
    #
    ll_lon, ll_lat = m(np.min(xzm), np.min(yzm), inverse=True)
    ur_lon, ur_lat = m(np.max(xzm), np.max(yzm), inverse=True)
    #
    return Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,
                           llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                           urcrnrlon=ur_lon, urcrnrlat=ur_lat,resolution='i')



def mzoom_subplot(ax,lon_w=-10, lon_e=15, lat_s=40, lat_n=60):
    """Returns a Basemap object (NorthPoleStere) focused in a region.

    lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
                                  W and S directions are negative.
    http://code.activestate.com/recipes/578379-plotting-maps-with-polar-stereographic-projection-/

    Europe : m = npstere_zoom.mzoom(lon_w=-30, lon_e=30, lat_s=35, lat_n=70)
    Greenland : m = npstere_zoom.mzoom(lon_w=-60, lon_e=-20, lat_s=58.75, lat_n=85)
    """
    lon_0=lon_w + (lon_e - lon_w) / 2.
    lat_0=lat_n
    #
    m = Basemap(projection='npstere', lon_0=lon_0, boundinglat=0, resolution='l',ax=ax)
    #
    lonzm = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
    latzm = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
    xzm, yzm = m(lonzm, latzm)
    #
    ll_lon, ll_lat = m(np.min(xzm), np.min(yzm), inverse=True)
    ur_lon, ur_lat = m(np.max(xzm), np.max(yzm), inverse=True)
    #
    return Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,
                           llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                           urcrnrlon=ur_lon, urcrnrlat=ur_lat,resolution='l',ax=ax)
