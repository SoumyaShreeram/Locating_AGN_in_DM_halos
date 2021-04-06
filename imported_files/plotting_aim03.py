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
            
    ax.set_title(title, fontsize=18)
    ax.grid(False)
    return


def plotNumberDensityVsRadius(num_pairs_all0, num_pairs_all1, title, plot_shell_vol=False):
    """
    Function to plot the number density of pairs found as a function of the projected separation for a range of different mass bins
    """
    # get shell volume and projected radius bins
    r_p, _, shell_volume = aimm.shellVolume()
    
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    ax.plot(r_p[1:], num_pairs_all0, "s",  color='k', label='DM halos', ms=9, mec='k')
    ax.plot(r_p[1:], num_pairs_all1, "s",  color='b', label='AGNs', ms=9, mec='b')
    
    # errorbars
    ax.errorbar(r_p[1:], num_pairs_all[i], yerr=getError(num_pairs_all[i]), ecolor=pal[i], fmt='none', capsize=4.5)
    ax.errorbar(r_p[1:], num_pairs_all[i], yerr=getError(num_pairs_all[i]), ecolor=pal[i], fmt='none', capsize=4.5)
        if np.any(num_pairs_all[i]) != 0: ax.set_yscale("log")
            
    # plot the shell volume
    if plot_shell_vol:
        ax.plot(r_p[1:], 1/shell_volume, "grey", marker=".", mfc='k', ls="--", label='Shell Volume')    
    
    setLabel(ax, r'Separation, $r$ [kpc/h]', r'$n_{\rm pairs}}$ [$h^{3}/{\rm kpc}^{-3}$]', title, [np.min(r_p[1:])-1, np.max(r_p[1:])+1], 'default', legend=True)
    ax.set_yscale("log")
    return ax

def plotEffectOfTimeSinceMerger(num_pairs_dt_m, dt_m_arr, title, binsize=15):
    """
    Function to plot the effect of time since merger of the number of pairs found
    """
    pal_r = sns.color_palette("rocket", len(dt_m_arr)).as_hex()
    labels = [r'$\Delta t_{\rm m}$ = %d Gyr'%dt for dt in dt_m_arr]
    
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    n_p = [num_pairs_dt_m[0], num_pairs_dt_m[1], num_pairs_dt_m[2], num_pairs_dt_m[3]]
    ax.hist(n_p,  bins=binsize, color=pal_r, label=labels)
    
    setLabel(ax, r'$n_{\rm pairs}$ [kpc$^{-3}$]', r'Number of counts', title, 'default', 'default', legend=False)
    ax.legend(loc='upper right', fontsize=14)
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    return 

def plotTimeSinceMergerMassBins(dt_m_arr, num_pairs, title="DM Halos"):
    """
    Function to study the mass and merger dependence simultaneously
    """
    # get shell volume and projected radius bins
    r_p, _, shell_volume = aimm.shellVolume()
    
    # initiating plot params
    color_palatte = sns.color_palette("magma", len(dt_m_arr)).as_hex()
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    for t, dt in enumerate(dt_m_arr):
        ax.plot(r_p[1:], num_pairs[t], "s",  mfc=color_palatte[t], ms=9, mec='k', label= r'$\Delta t_{\rm m} = %d$ Gyr'%(dt))
    
    # plot the shell volume
    ax.plot(r_p[1:], 1/shell_volume, "grey", marker=".", mfc='k', ls="--", label='Shell Volume')
    
    setLabel(ax, r'Separation, $r$ [kpc/h]', r'$n_{\rm pairs}$ [kpc$^{-3}$]', title, [np.min(r_p[1:])-1, np.max(r_p[1:])+1], 'default', legend=True)
    ax.legend(loc='upper right', fontsize=14)
    
    ax.set_yscale("log")
    return 