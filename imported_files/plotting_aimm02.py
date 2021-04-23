# -*- coding: utf-8 -*-
"""Plotting.py for notebook 02_AGN_incidence_in_Major_Mergers

This python file contains all the functions used for plotting graphs and maps in the 2nd notebook (.ipynb) of the repository: 02. Creating a Major Merger (MM) catalogue to study AGN incidence due to galaxy mergers

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 30th March 2021
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

import os
import importlib

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

import Agn_incidence_from_Major_Mergers as aimm

def setLabel(ax, xlabel, ylabel, title, xlim, ylim, legend=True):
    """
    Function defining plot properties
    @param ax :: axes to be held
    @param xlabel, ylabel :: labels of the x-y axis
    @param title :: title of the plot
    @param xlim, ylim :: x-y limits for the axis
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != 'default':
        ax.set_xlim(xlim)
    
    if ylim != 'default':
        ax.set_ylim(ylim)
    
    if legend:
        l = ax.legend(loc='best',  fontsize=14)
        for legend_handle in l.legendHandles:
            legend_handle._legmarker.set_markersize(12)
            
    ax.grid(False)
    ax.set_title(title, fontsize=18)
    return

def plotCountsInMassBins(num_mass_mm_halo, num_mass_mm_agn):
    """
    Function to plot the number of objects (AGNs/Halos) found in each defined mass bin
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))

    ax.plot(num_mass_mm_halo[1][1:], num_mass_mm_halo[0], "ks", ls=":", label='DM halos')
    ax.plot(num_mass_mm_agn[1][1:], num_mass_mm_agn[0], "bs", ls=":", label='AGNs')

    setLabel(ax, r'$\log{\ m_{\rm stellar}}$', 'Counts', '', 'default', 'default', legend=True)
    return

def getError(num_pairs, num_objs):
    """
    Function to get the Poisson errors on the number of pairs
    """
    # get number of pairs from the density
    num_pairs = num_pairs*num_objs
    
    error_arr = []
    for n in num_pairs:
        if n != 0:     
            error_arr.append(1/np.sqrt(n))
        
        # if no pairs are found for the given radius
        else:
            error_arr.append(0)
    return error_arr


def plotNumberDensityVsRadius(num_pairs_all, obj_bins, pal, title, l_type=True, want_label=True, plot_shell_vol=False):
    """
    Function to plot the number density of pairs found as a function of the projected separation for a range of different mass bins
    """
    num_objs, mass_bins = obj_bins[0], obj_bins[1]
    
    # get shell volume and projected radius bins
    r_p, _, shell_volume = aimm.shellVolume()
    r_p = r_p*1e3
    
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    for i in range(len(num_pairs_all)):
        # if you want have all the mass bin labels
        if want_label:
            l = r'$M = 10^{%.1f-%.1f} M_{\odot}$'%(mass_bins[i], mass_bins[i+1])
        else:
            l = ""       
        
        # main plot
        ax.plot(r_p[1:], num_pairs_all[i], linestyle= '', marker="s",  color=pal[i], label=l, ms=9, mec='k')
        # errorbars
        ax.errorbar(r_p[1:], num_pairs_all[i], yerr=getError(num_pairs_all[i], num_objs[i]), ecolor=pal[i], fmt='none', capsize=4.5)
        if np.any(num_pairs_all[i]) != 0: ax.set_yscale("log")
    
    # plot the shell volume
    if plot_shell_vol:
        ax.plot(r_p[1:], 1/shell_volume, "grey", marker=".", mfc='k', ls="--", label='Shell Volume')    
    
    setLabel(ax, r'Separation, $r$ [kpc]', r'$f_{\rm MM pairs}}$', title, [np.min(r_p[1:]), np.max(r_p[1:])], 'default', legend=l_type)
    if l_type: ax.legend(loc=(1.04, 0), fontsize=14)
    return ax

def plotEffectOfTimeSinceMerger(num_pairs_dt_m, dt_m_arr, title, binsize=50):
    """
    Function to plot the effect of time since merger of the number of pairs found
    """
    pal_r = sns.color_palette("rocket").as_hex()
    labels = [r'$\Delta t_{\rm m}$ = %d Gyr'%dt for dt in dt_m_arr]
    
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    ax.hist(num_pairs_dt_m,  bins=binsize, color=pal_r[1:len(dt_m_arr)+1], label=labels)
    
    setLabel(ax, r'$n_{\rm pairs}$ [kpc$^{-3}$]', r'Number of counts', title, 'default', 'default', legend=False)
    ax.legend(loc='upper right', fontsize=14)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    return 

def generateColorPalattes(length, ctype):
    return sns.color_palette(ctype, length).as_hex()

def plotTimeSinceMergerMassBins(dt_m_arr, mass_bins, num_pairs, title="DM Halos"):
    """
    Function to study the mass and merger dependence simultaneously
    """
    # get shell volume and projected radius bins
    r_p, _, _ = aimm.shellVolume()
    
    # initiating plot params
    c_types,l = ["mako", "coolwarm", "Spectral", "hls"], len(r_p)+1
    color_palatte = [generateColorPalattes(l, c_t) for c_t in c_types]
    
    for t, dt in enumerate(dt_m_arr):
        ax = plotNumberDensityVsRadius(num_pairs[t], mass_bins, color_palatte[t], r'%s ($\Delta t_{\rm m} = %d$ Gyr)'%(title,dt))
    return 