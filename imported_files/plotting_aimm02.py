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

def plotNumberDensityVsRadius(num_pairs_all, mass_bins, r_p, shell_volume, pal, title):
    """
    Function to plot the number density of pairs found as a function of the projected separation for a range of different mass bins
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    for i in range(len(num_pairs_all)):
        ax.plot(r_p[1:], num_pairs_all[i], "s",  color=pal[i], label=r'$M = 10^{%.1f-%.1f} M_{\odot}$'%(mass_bins[i], mass_bins[i+1]))
    ax.plot(r_p[1:], 1/shell_volume, "grey", ls="--", label='Shell Volume')

    setLabel(ax, r'Projected separation, $r_p$ [kpc/h]', r'$n_{\rm pairs}}$ [kpc$^{-3}$]', title, 'default', 'default', legend=True)
    ax.set_yscale("log")
    return 