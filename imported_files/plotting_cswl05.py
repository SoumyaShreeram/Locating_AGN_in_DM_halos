# -*- coding: utf-8 -*-
"""Plotting.py for notebook 05_Preliminary_comparison_of_simulations_AGN_fraction_with_data

This python file contains all the functions used for plotting graphs and maps in the 2nd notebook (.ipynb) of the repository: 05. Preliminary comparison of the 𝑓MM between simulation and data

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 27th April 2021
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

def plotFpairs(ax, r_p, f_pairs, f_pairs_err, label, color='r', errorbar = True):
    # changing all unit to kpc
    r_p_kpc, f_pairs = 1e3*r_p[1:], 1e9*f_pairs

    # plotting the results
    ax.plot( r_p_kpc , f_pairs, 's', color=color, label = label)
    if errorbar:
        ax.errorbar(r_p_kpc , np.array(f_pairs), yerr=f_pairs_err, ecolor='k', fmt='none', capsize=4.5)
    return ax