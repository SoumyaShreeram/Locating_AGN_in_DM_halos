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


from scipy.stats import norm

def setLabel(ax, xlabel, ylabel, title='', xlim='default', ylim='default', legend=True):
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
    r_p_kpc, n_pairs =  1e3*r_p[1:len(n_pairs)+1], n_pairs
    
    # plotting the results
    ax.plot( r_p_kpc , n_pairs, 'd', mec = mec, ms = 10, color=color, label=label)
    
    # errorbars
    if errorbars:
        n_pairs_err = np.array(n_pairs_err)
        ax.errorbar(r_p_kpc , np.array(n_pairs), yerr=n_pairs_err, ecolor=mec, fmt='none', capsize=4.5)
    return ax, n_pairs, n_pairs_err

def plotFracNdensityPairs(hd_z_halo, pairs_all, pairs_mm_dv_all, pairs_selected_all, plot_selected_pairs=True):
    """
    Function to plot the fractional number density of pairs for different selection criteria
    """
    flare = sns.color_palette("pastel", 5).as_hex()
    mec = ['k', '#05ad2c', '#db5807', '#a30a26', 'b']
    fig, ax = plt.subplots(1,1,figsize=(5,4))

    # plotting the 4 cases with the 4 different cuts
    ax, n_pairs, n_pairs_err = plotNpSep(ax, hd_z_halo, pairs_all[1], 'k', r' $\mathbf{\Gamma}_{m;\  \Delta v;\ t_{\rm MM};\  \tilde{X}_{\rm off}}(r)\ $', mec[0]) 
    
    ax, n_mm_dv_pairs, n_pairs_mm_dv_err = plotNpSep(ax, hd_z_halo, pairs_mm_dv_all[1], flare[3], r'$\mathbf{\Gamma}_{t_{\rm MM};\  \tilde{X}_{\rm off}}(r|\ m;\  \Delta v)$', mec[3])
    
    if plot_selected_pairs:
        ax, n_selected_pairs, n_selected_err = plotNpSep(ax, hd_z_halo, pairs_selected_all[1], flare[2], r'$\mathbf{\Gamma}(r|\ m;\  \Delta v;\ t_{\rm MM};\  \tilde{X}_{\rm off} )$'+'\n'+r'$t_{\rm MM} \in [0.6-1.2]$ Gyr, $\tilde{X}_{\rm off} \in [0.17, 0.54]$', mec[1])
    

    ax.set_yscale("log")
    setLabel(ax, r'Separation, $r$ [kpc]', r'$\mathbf{\Gamma}(r)$ [Mpc$^{-3}$]', '', 'default', 'default', legend=False)
    ax.legend(bbox_to_anchor=(1.05, 1),  loc='upper left', fontsize=15, frameon=False)
    
    pairs_arr = np.array([n_pairs, n_mm_dv_pairs, n_selected_pairs], dtype=object)
    pairs_arr_err = np.array([n_pairs_err, n_pairs_mm_dv_err, n_selected_err], dtype=object)
    return pairs_arr, pairs_arr_err, ax

def plotCumulativeDist(vol, dt_m_arr, pairs_mm_all, pairs_mm_dv_all, n_pairs_mm_dt_all, n_pairs_mm_dv_dt_all, param = 't_mm'):
    """
    Function to plot the cumulative number of pairs for the total vol (<z=2) for pairs with dz and mass ratio criteria
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, _ = aimm.shellVolume()

    fig, ax = plt.subplots(1,2,figsize=(17,6))
    pal = sns.color_palette("coolwarm", len(dt_m_arr)+1).as_hex()

    ax[0].plot( (1e3*r_p[1:]), (pairs_mm_all[1][1:]/(2*vol)), 'X', color='k', label='No criterion')
    ax[1].plot( (1e3*r_p[1:]), (pairs_mm_dv_all[1][1:]/(2*vol)), 'X', color='k', label='No criterion')

    for t_idx in range(len(dt_m_arr)):
        np_mm_dt, np_mm_dv_dt = n_pairs_mm_dt_all[t_idx], n_pairs_mm_dv_dt_all[t_idx]    
        if param == 't_mm':
            label = r'$t_{\rm MM} \in$ %.1f-%.1f Gyr'%(dt_m_arr[t_idx][0], dt_m_arr[t_idx][1])
        else:
            label = r'$\tilde{X}_{\rm off} \in$ %.1f-%.1f Gyr'%(dt_m_arr[t_idx][0], dt_m_arr[t_idx][1])
        ax[0].plot( (1e3*r_p[1:]), (np_mm_dt[1:]/(2*vol)), 'kX', label = label, color=pal[t_idx])
        ax[1].plot( (1e3*r_p[1:]), (np_mm_dv_dt[1:]/(2*vol)), 'kX', color=pal[t_idx])

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    setLabel(ax[0], r'Separation, $r$ [kpc]', 'Cumulative number of halo pairs\n'+r'[Mpc$^{-3}$]', r'Mass ratio 3:1, $\Delta z_{\rm R, S} < 10^{-3}$', 'default', 'default', legend=False)
    setLabel(ax[1], r'Separation, $r$ [kpc]', r'', 'Mass ratio 3:1', 'default', 'default', legend=False)
    ax[0].legend(bbox_to_anchor=(-0.5, -0.7), loc='lower left', ncol=4, frameon=False)
    return pal


def plotParameterDistributions(xoff_all, string=r'$\tilde{X}_{\rm off}$', xmax=5, filestring='xoff'):
    """
    Function to plot the parameter distribution i.e. SF and PDF
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    sf_xoff = norm.sf(np.sort(xoff_all))
    if string == r'$\tilde{X}_{\rm off}$':
        ax.plot(np.sort(xoff_all), sf_xoff, 'r-', label=r'Survival Function of '+string)
        xmax = np.max(xoff_all)
    else:
        ax.plot(np.sort(xoff_all), 1-sf_xoff, 'r-', label=r'CDF of '+string)
    
    pdf_xoff = norm.pdf(np.sort(xoff_all))
    ax.plot(np.sort(xoff_all), pdf_xoff, 'k-', label=r'PDF of '+string)
    
    setLabel(ax, string, 'Distribution of '+string,  '', [np.min(xoff_all), xmax], 'default', legend=True)
    plt.savefig('../figures/'+filestring+'_function.png', facecolor='w', edgecolor='w', bbox_inches='tight')
    return ax

def axId(i):
    if i == 0: m, n = 0, 0
    if i == 1: m, n = 0, 1
    if i == 2: m, n = 1, 0
    if i == 3: m, n = 1, 1
    return int(m), int(n)

def plotPdf(ax, arr, string, color):
    pdf_arr = norm.pdf(np.sort(arr))
    ax.plot(np.sort(arr), pdf_arr, '-', color=color, label=r'PDF of '+string, lw=4)
    return

def saveFig(filename):
    plt.savefig('../figures/'+filename, facecolor='w', edgecolor='w', bbox_inches='tight')
    return

def plotContour(u_pix, matrix_2D, xmin=10, xmax=150, ymin=0, ymax=2, ax=None, cmap='YlGnBu'):
    """
    Function plots a contour map 
    @u_pix :: number of pixels in the FOV
    @Returns :: 2D matrix
    """
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    if isinstance(u_pix, (int, float)):
        X, Y = np.meshgrid(np.linspace(0, u_pix, u_pix), np.linspace(0, u_pix, u_pix))
    if isinstance(u_pix, (list, tuple, np.ndarray)): # if FOV is a rectangle
        X, Y = np.meshgrid(np.linspace(xmin, xmax, u_pix[0]), np.linspace(ymin, ymax, u_pix[1]))
    
    plot = ax.contourf(X, Y, matrix_2D, cmap=cmap, origin='image')
    return ax, plot

def labelMZTmmXoff(ax, ylabel, redshift_limit=2):
    setLabel(ax[0, 0], r'Stellar mass, $\log{M^*}$', ylabel, '', 'default', 'default', legend=False)
    setLabel(ax[0, 1], 'Redshift, $z$', '', '', [0, redshift_limit], 'default', legend=False)
    setLabel(ax[1, 0], r'$t_{\rm MM}$', ylabel, '', 'default', 'default', legend=False)
    ax[1,0].set_xscale('log')

    setLabel(ax[1, 1], r'$\tilde{X}_{\rm off}$', '', '', 'default', 'default', legend=False)
    return

def plotBinsMZdistribution(mz_mat_tmm0, mz_mat_tmm1, tmm_bins, param=r'$t_{\rm MM} = $'):
    
    fig, ax = plt.subplots(2,2,figsize=(15,15))

    ax0, pt0 = plotContour((mz_mat_tmm0[0].shape[1], mz_mat_tmm0[0].shape[0]), mz_mat_tmm0[0], ymin=0.8, ymax=1.3, cmap='terrain', ax=ax[0, 0])
    ax1, pt1 = plotContour((mz_mat_tmm0[1].shape[1], mz_mat_tmm0[1].shape[0]), mz_mat_tmm0[1], ymin=0., ymax=2, cmap='terrain', ax=ax[1, 0])
    setLabel(ax[0, 0], '', 'Mass ratio', param+' %.2f - %.2f'%(tmm_bins[0][0], tmm_bins[0][1]), 'default', 'default', legend=False)
    setLabel(ax[1, 0], r'Separation, $r_p$ [kpc]', 'Mean redshift', '', 'default', 'default', legend=False)

    ax2, pt2 = plotContour((mz_mat_tmm1[0].shape[1], mz_mat_tmm1[0].shape[0]), mz_mat_tmm1[0], ymin=0.8, ymax=1.3, cmap='terrain', ax=ax[0, 1])
    ax3, pt3 = plotContour((mz_mat_tmm1[1].shape[1], mz_mat_tmm1[1].shape[0]), mz_mat_tmm1[1], ymin=0., ymax=2, cmap='terrain', ax=ax[1, 1])
    setLabel(ax[0, 1], '', '', param+ ' %.2f - %.2f'%(tmm_bins[1][0], tmm_bins[1][1]), 'default', 'default', legend=False)
    setLabel(ax[1, 1], r'Separation, $r_p$ [kpc]', '', '', 'default', 'default', legend=False)
    return

def snsPlotLabels():
    plt.xlabel(r'$t_{\rm MM}$ [Gyr]', fontsize=20)
    plt.ylabel(r'$\tilde{X}_{\rm off}$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return

def plotGaussianKde(param_arr, Z, string, i, j, set_xy_lim=True):
    xmin, xmax = np.min(param_arr[i]), np.max(param_arr[i])
    ymin, ymax = np.min(param_arr[j]), np.max(param_arr[j])

    fig, ax = plt.subplots(1,1,figsize=(5, 5))
    ax.plot(param_arr[i], param_arr[j], 'k.', markersize=.02)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    if set_xy_lim:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    
    setLabel(ax, string[i], string[j], '', 'default', 'default', legend=False)
    return ax


def plotModelResults(ax, hd_halo, pairs_all, pairs_selected, vol):
    """
    Plots the models generated for bins of Tmm and Xoff
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, shell_volume = aimm.shellVolume()

    #  plotting the cumulative pairs
    norm = vol*len(hd_halo)
    np_all, np_selected = pairs_all/norm, pairs_selected[1]/norm    

    ax[0].plot( (1e3*r_p), (np_selected), 'rX', ls = '--', ms=9, label='Selected pairs')
    ax[0].plot( (1e3*r_p), (np_all), 'kX', ls = '--', label = 'All pairs', ms = 9)

    setLabel(ax[0], r'', r'Cumulative $n_{\rm halo\ pairs}}$  [Mpc$^{-3}$]', '', 'default', 'default', legend=True)
    
    # plotting the pairs in bins of radius 
    np_all_bins, np_all_bins_err = cswl.nPairsToFracPairs(hd_halo, pairs_all)
    np_selected_bins, np_selected_bins_err = cswl.nPairsToFracPairs(hd_halo, pairs_selected[1])

    _ = plotFpairs(ax[1], r_p, np_all_bins, np_all_bins_err, label = 'All pairs', color='k')
    _ = plotFpairs(ax[1], r_p, np_selected_bins, np_selected_bins_err, label = 'Selected pairs')
    ax[1].set_yscale('log')
    setLabel(ax[1], r'', r'$n_{\rm halo\ pairs}}$  [Mpc$^{-3}$]', '', 'default', 'default', legend=True)
    
    # plotting the pairs in bins with respect to the control
    _ = plotFpairs(ax[2], r_p, np_selected_bins/np_all_bins, np_selected_bins_err, label='wrt all pairs', color='orange')
    
    setLabel(ax[2], r'Separation, $r$ [kpc]', r'Fraction of pairs, $f_{\rm halo\ pairs}}$ ', '', 'default', 'default', legend=False)

    return np_selected_bins/np_all_bins