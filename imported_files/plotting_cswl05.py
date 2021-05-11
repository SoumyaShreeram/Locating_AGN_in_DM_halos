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
        ax.errorbar(r_p_kpc , f_pairs.value, yerr=np.array(f_pairs_err), ecolor='k', fmt='none', capsize=4.5)
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

def plotNpSep(ax, hd_z_halo, pairs_all, color, label, mec, errorbars = True):
    """
    Function plots the n_p as a function of separation
    """
    pairs_all = np.array(pairs_all)
    
    # get shell volume and projected radius bins [Mpc]
    r_p, shell_volume = aimm.shellVolume()
    
    # get number density of pairs with and without selection cuts
    n_pairs, n_pairs_err = cswl.nPairsToFracPairs(hd_z_halo, pairs_all)
    
    # changing all unit to kpc
    r_p_kpc, n_pairs =  1e3*r_p[1:], n_pairs
    
    # plotting the results
    ax.plot( r_p_kpc , n_pairs, 'd', mec = mec, ms = 10, color=color, label=label)
    
    # errorbars
    if errorbars:
        n_pairs_err = np.array(n_pairs_err)
        ax.errorbar(r_p_kpc , np.array(n_pairs), yerr=n_pairs_err, ecolor=mec, fmt='none', capsize=4.5)
    return ax, n_pairs, n_pairs_err

def plotFracNdensityPairs(hd_z_halo, pairs_all, pairs_mm_all, pairs_dv_all, pairs_mm_dv_all, num_mm_control_pairs):
    """
    Function to plot the fractional number density of pairs for different selection criteria
    """
    flare = sns.color_palette("pastel", 5).as_hex()
    mec = ['k', '#05ad2c', '#db5807', '#a30a26', 'b']
    fig, ax = plt.subplots(1,1,figsize=(7,6))

    # plotting the 4 cases with the 4 different cuts
    ax, n_pairs, n_pairs_err = plotNpSep(ax, hd_z_halo, pairs_all[1], 'k', 'All pairs', mec[0]) 
    ax, n_mm_pairs, n_pairs_mm_err = plotNpSep(ax, hd_z_halo, pairs_mm_all[1], flare[1], r'Mass ratio 3:1', mec[2])
    ax, n_dv_pairs, n_pairs_dv_err = plotNpSep(ax, hd_z_halo, pairs_dv_all[1], flare[2], r'$\Delta z_{R\ and\ S} < 10^{-3} $', mec[1])
    ax, n_mm_dv_pairs, n_pairs_mm_dv_err = plotNpSep(ax, hd_z_halo, pairs_mm_dv_all[1], flare[3], r'Mass ratio 3:1, $\Delta z_{R\ and\ S }  < 10^{-3} $', mec[3])
    ax, n_mz_control_pairs, n_mz_control_err = plotNpSep(ax, hd_z_halo, num_mm_control_pairs, flare[4],  'Control sample', mec[3])

    ax.set_yscale("log")
    setLabel(ax, r'Separation, $r$ [kpc]', r'$n_{\rm halo\ pairs}}$ [Mpc$^{-3}$]', '', 'default', 'default', legend=False)
    ax.legend(bbox_to_anchor=(1.05, 1),  loc='upper left', fontsize=14, frameon=False)
    
    pairs_arr = np.array([n_pairs, n_mm_pairs, n_dv_pairs, n_mm_dv_pairs, n_mz_control_pairs], dtype=object)
    pairs_arr_err = np.array([n_pairs_err, n_pairs_mm_err, n_pairs_dv_err, n_mz_control_err], dtype=object)
    return pairs_arr, pairs_arr_err

def plotCumulativeDist(vol, dt_m_arr, pairs_mm_all, pairs_mm_dv_all, n_pairs_mm_dt_all, n_pairs_mm_dv_dt_all, param = 't_mm'):
    """
    Function to plot the cumulative number of pairs for the total vol (<z=2) for pairs with dz and mass ratio criteria
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, _ = aimm.shellVolume()

    fig, ax = plt.subplots(1,2,figsize=(17,6))
    pal = sns.color_palette("hls", len(dt_m_arr)+1).as_hex()

    ax[0].plot( (1e3*r_p[1:]), (pairs_mm_all[1][1:]/(2*vol)), 'X', color='k', label='No criterion')
    ax[1].plot( (1e3*r_p[1:]), (pairs_mm_dv_all[1][1:]/(2*vol)), 'X', color='k', label='No criterion')

    for t_idx in range(len(dt_m_arr)):
        np_mm_dt, np_mm_dv_dt = n_pairs_mm_dt_all[t_idx], n_pairs_mm_dv_dt_all[t_idx]    
        if param == 't_mm':
            label = r'$t_{\rm MM}$ = %.1f Gyr'%(dt_m_arr[t_idx])
        else:
            label = r'$\tilde{X}_{\rm off}$ = %.2f'%dt_m_arr[t_idx]
            
        ax[0].plot( (1e3*r_p[1:]), (np_mm_dt[1:]/(2*vol)), 'kX', label = label, color=pal[t_idx])
        ax[1].plot( (1e3*r_p[1:]), (np_mm_dv_dt[1:]/(2*vol)), 'kX', color=pal[t_idx])

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    setLabel(ax[0], r'Separation, $r$ [kpc]', 'Cumulative number of halo pairs\n'+r'[Mpc$^{-3}$]', r'Mass ratio 3:1, $\Delta z_{\rm R, S} < 10^{-3}$', 'default', 'default', legend=True)
    setLabel(ax[1], r'Separation, $r$ [kpc]', r'', 'Mass ratio 3:1', 'default', 'default', legend=False)
    return pal