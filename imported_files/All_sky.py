# -*- coding: utf-8 -*-
"""All_sky.py for notebooks that study all the pixels in the sky

This python file contains all the functions used for studying properties on the scale of the whole sky i.e. all the pixels in the simulations are considered

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 2nd April 2020
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# scipy modules
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

# system imports
import sys
import os
import importlib
from time import sleep

# plotting imports
import matplotlib
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import seaborn as sns


# personal imports
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import plotting_aimm02 as pt
"""

Functions begin

"""
def showProgress(idx, n):
    """
    Function prints the progress bar for a running function
    @param idx :: iterating index
    @param n :: total number of iterating variables/ total length
    """
    j = (idx+1)/n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
    sleep(0.25)
    return

def createPixelNos(total_halo_pixels=767):
    """
    Function to create string pixel nos for opening files
    """
    total_px_arr = np.arange(0, total_halo_pixels+1)
    pixel_nos_arr = []
    for t in total_px_arr:
        if len(str(t)) == 1:
            pixel_nos_arr.append('00000'+str(t))
        if len(str(t)) == 2:
            pixel_nos_arr.append('0000'+str(t))
        if len(str(t)) == 3:
            pixel_nos_arr.append('000'+str(t))
    return pixel_nos_arr

def saveMMFiles(hd_mm_halo_all_px, hd_mm_agn_all_px, num_mass_mm, z, t):
    """
    Function to save the major merger headers
    """
    np.save('Data/MM_halos_all_px_z%d_t%d.fits'%(z, t), hd_mm_halo_all_px, allow_pickle=True)
    np.save('Data/MM_agn_all_px_z%d_t%d.fits'%(z, t), hd_mm_agn_all_px, allow_pickle=True)
    np.save('Data/num_mass_mm_z%d_t%d.npy'%(z, t), num_mass_mm, allow_pickle=True)
    return 


def getMMFilesAllSky(redshift_limit, time_since_merger, agn_FX_soft=0, num_pixels=25, save_files=True):
    """
    DEPRICATED: Generate MM sample for all number of pixels set in the input, for all mass bins.
    """
    # define cosmology 
    lcdm = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)

    pixel_no_arr = createPixelNos()
    hd_mm_halo_all_px, hd_mm_agn_all_px, num_mass_mm = [], [], []
    
    len_px = pixel_no_arr[:num_pixels]
    for i, pixel_no in enumerate(len_px):        
        # get whole header files
        hd_agn, hd_halo, _ = edh.getHeaders(pixel_no, np.array(['agn', 'halo']))

        # get conditions for agns and dm halos
        _, _, conditions_agn = edh.getAgnData(hd_agn, agn_FX_soft, redshift_limit)    
        _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)

        # create MM sample (each mass bin is stored as a separate table)
        hd_mm_halo_all, num_mass_mm_halo = aimm.majorMergerSampleForAllMassBins(hd_halo, conditions_halo, lcdm, time_since_merger)
        hd_mm_agn_all, num_mass_mm_agn = aimm.majorMergerSampleForAllMassBins(hd_agn, conditions_agn, lcdm, time_since_merger)
        
        hd_mm_halo_all_px.append(hd_mm_halo_all)
        hd_mm_agn_all_px.append(hd_mm_agn_all)
        num_mass_mm.append(np.array([num_mass_mm_halo, num_mass_mm_agn], dtype=object))
        
        # show progress
        showProgress(i, len(len_px))
        
    hd_mm_halo_all_px = np.concatenate(hd_mm_halo_all_px, axis=None)
    hd_mm_agn_all_px = np.concatenate(hd_mm_agn_all_px, axis=None)
    
    if save_files:
        saveMMFiles(hd_mm_halo_all_px, hd_mm_agn_all_px, num_mass_mm, z=redshift_limit, t=time_since_merger)
    return hd_mm_halo_all_px, hd_mm_agn_all_px, num_mass_mm

def countPairsAllSky(z, t, plot=True):
    """
    DEPRICATED: Function to count pairs for different z and t
    """
    hd_mm_halo_all_px, hd_mm_agn_all_px, num_mass_mm = getMMFilesAllSky(redshift_limit=z, time_since_merger=t, save_files=False)
    
    num_pairs_halo_all_px, _, _ = aimm.getNumberDensityOfPairs(hd_mm_halo_all_px)
    num_pairs_agn_all_px, _, _ = aimm.getNumberDensityOfPairs(hd_mm_agn_all_px)
    
    if plot:
        pal_halo = ['k' for i in range(len(num_pairs_halo_all_px))]
        pal_agn = ['b' for i in range(len(num_pairs_agn_all_px))]

        pt.plotNumberDensityVsRadius(num_pairs_halo_all_px, num_mass_mm[0][0][1], pal_halo, r'DM Halos ($z< %.1f,\ \Delta t_{\rm m} = %d$ Gyr)'%(z, t), l_type=False, want_label=False)
        pt.plotNumberDensityVsRadius(num_pairs_agn_all_px, num_mass_mm[0][1][1], pal_agn, r'AGNs ($z< %.1f,\ \Delta t_{\rm m} = %d$ Gyr)'%(z, t), l_type=False, want_label=False) 
    return

def genPixelArr(ll, ul):
    """
    Function to get the pixel array 
    """
    number = np.arange(ll, ul)
    if ul <= 10:
        pixel_no_arr = ['00000'+ str(n) for n in number]
    elif ul <= 100 and ul > 10:
        pixel_no_arr = ['0000'+ str(n) for n in number]
    else:
        pixel_no_arr = ['000'+ str(n) for n in number]
    return pixel_no_arr

def allPixelNames():
    """
    Function to give the names of all the pixels of the unit simulation
    """
    hundreds = np.concatenate([genPixelArr(ll=int(i*100), ul=int(i*100 + 100) ) for i in np.arange(1, 7) ], axis=None)
    
    pixel_no_big_arr = np.array([genPixelArr(ll=0, ul=10), genPixelArr(ll=10, ul=100), hundreds,  genPixelArr(ll=700, ul=768)], dtype=object)
    pixel_no_big_arr_concat = np.concatenate(pixel_no_big_arr, axis=None)
    return pixel_no_big_arr_concat

def getHaloLengths(redshift_limit=2):
    halo_lengths = []
    pixel_arr, _ = allPixelNames()
    for px in pixel_arr[0:10]:
        print(px)
        _, hd_halo, _ = edh.getHeaders(px, np.array(['halo']))
        # halos
        _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
        
        hd_z_halo = hd_halo[conditions_halo]
        
        halo_lengths.append( len(hd_z_halo) )
    np.save('Data/all_sky_halo_lengths_z%.1f.npy'%redshift_limit, halo_lengths, allow_pickle=True)
    return 