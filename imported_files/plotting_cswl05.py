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
import Comparison_simulation_with_literature_data as cswl

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
        l = ax.legend(loc='best',  fontsize=14, frameon=False)
        for legend_handle in l.legendHandles:
            legend_handle._legmarker.set_markersize(12)
            
    ax.grid(False)
    ax.set_title(title, fontsize=18)
    return

def plotFpairs(ax, r_p, f_pairs, f_pairs_err, label, color='r', errorbar = True):
    # changing all unit to kpc
    r_p_kpc, f_pairs = 1e3*r_p[1:], f_pairs

    # plotting the results
    ax.plot( r_p_kpc , f_pairs, 's', ls='--', color=color, label = label)
    if errorbar:
        ax.errorbar(r_p_kpc , np.array(f_pairs), yerr=f_pairs_err, ecolor='k', fmt='none', capsize=4.5)
    return ax


def plotScaleMMdistribution(halo_m_scale_arr_all_r, cosmo, dt_m_arr):
    """
    Function plots the number of objects in pairs as a function of the scale of last MM
    --> the cuts on delta t_mm are overplotted to see the selection criterion
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    bins = 20
    hist_all_r = np.zeros((0, bins))

    for i in range(len(halo_m_scale_arr_all_r)):
        hist_counts, a = np.histogram(halo_m_scale_arr_all_r[i], bins=bins)
        hist_all_r = np.append(hist_all_r, [hist_counts], axis=0)

        ax.plot(a[1:], hist_counts, '--', marker = 'd', color='k')

    scale_mm = cswl.tmmToScale(cosmo, dt_m_arr)
    pal1 = sns.color_palette("Spectral", len(scale_mm)+1).as_hex()

    for j, l in enumerate(scale_mm):
        ax.vlines(l, np.min(hist_all_r), np.max(hist_all_r), colors=pal1[j], label=r'$t_{\rm MM}$ = %.1f Gyr'%dt_m_arr[j])

    setLabel(ax, r'Scale factor, $a$', r'Counts', '', 'default',[np.min(hist_all_r), np.max(hist_all_r)], legend=False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_yscale('log')
    return

def plotNpSep(ax, hd_z_halo, pairs_all, color, label, mec, errorbars = False):
    """
    Function plots the n_p as a function of separation
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, shell_volume = aimm.shellVolume()
    
    # get number density of pairs with and without selection cuts
    n_pairs, n_pairs_err = cswl.nPairsToFracPairs(hd_z_halo, pairs_all[1])
    good = np.nonzero(n_pairs)[0]
    
    # changing all unit to kpc
    r_p_kpc, n_pairs =  1e3*r_p[1:][good], n_pairs[good]
    
    # plotting the results
    ax.plot( r_p_kpc , n_pairs, 'd', mec = mec, ms = 9, color=color, label=label)
    
    # errorbars
    if errorbars:
        n_pairs_err = np.array(n_pairs_err)[good]
        ax.errorbar(r_p_kpc , np.array(n_pairs), yerr=n_pairs_err, ecolor='k', fmt='none', capsize=4.5)
    return ax, n_pairs, n_pairs_err