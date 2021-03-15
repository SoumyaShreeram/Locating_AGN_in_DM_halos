# -*- coding: utf-8 -*-
"""Plotting.py
## Plotting functions for simulating stars using Slitless-Spectroscopy
This python file contains all the functions used for plotting graphs and density maps across the various notebooks.
**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 8th October 2020
## 1. Imports
"""

import astropy.units as u
import astropy.io.fits as fits

import numpy as np
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

import Exploring_DM_Haloes as edh
"""
1. Functions for labeling plots

"""

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
        ax.legend(loc='best',  fontsize=14)
    ax.grid(False)
    ax.set_title(title, fontsize=18)
    return

def plotAgnClusterDistribution(pos_z_clu, pos_z_AGN, pos_z_gal, min_cluster_mass):
    """
    Function to plot the AGN cluster distribution
    @pos_z_clu :: postion and redshifts of all the selected 'clusters'
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    @pos_z_gal :: postion and redshifts of all the selected galaxies
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    # plotting galaxies
    galaxy = ax.plot(pos_z_gal[0], pos_z_gal[1], '.',  color='#fcd16d', markersize=0.08, label=r'Galaxies', alpha=0.2)
    
    # plotting clusters
    cluster = ax.plot(pos_z_clu[0], pos_z_clu[1], 'o', color= '#03a351', markersize=3, label=r'Halos $M_{500c}> 10^{%.1f} M_\odot$ '%(np.log10(min_cluster_mass)))
    
    # plotting AGNs
    agn = ax.plot(pos_z_AGN[0], pos_z_AGN[1], '.',  color='k', markersize=2.5, label=r'AGN', alpha=0.7)

    # labeling axes and defining limits
    xlim = [np.min(pos_z_gal[0]), np.max(pos_z_gal[0])]
    ylim = [np.min(pos_z_gal[1]), np.max(pos_z_gal[1])]
    setLabel(ax, 'R.A. (deg)', 'Dec (deg)', 'Redshift $z<%.2f$'%(np.max(pos_z_clu[2])+0.1), xlim, ylim, legend=False)
    
    legend = ax.legend(loc='best',  fontsize=14)
    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(12)
    return

def plotHostSubHalos(hd_halo0, hd_halo1, hd_halo2, cluster_params, redshift_limit, pos_z_AGN):
    """
    Function to plot the host and satellite halo distribution
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them 
    --> divided into 3 because each hd_halo holds info on 1000 halos alone
    @cluster_params :: contains clu_FX_soft, galaxy_mag_r, min_cluster_mass where
        @min_cluster_mass :: min mass for halo to be called a cluster
    @redshift_limit :: upper limit on redshift
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    """ 
    ra_cen, dec_cen, ra_sat, dec_sat = edh.getPositionsHostSatHalos(hd_halo0, hd_halo1, hd_halo2, cluster_params, redshift_limit)
    
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    # plotting host halos
    host_halos = ax.plot(ra_cen, dec_cen, 'o', color= 'grey', markersize=2.5, label=r'Host-halos $P_{id}=-1$')
    
    # plotting sat halos
    host_halos = ax.plot(ra_sat, dec_sat, 'o', color= 'b', markersize=5, label=r'Satellite halos $P_{id} \neq -1$')
    
    
    # plotting AGNs
    agn = ax.plot(pos_z_AGN[0], pos_z_AGN[1], '*',  color='#07d9f5', markersize=2.5, label=r'AGN', alpha=0.7)

    # labeling axes and defining limits
    xlim = [np.min(pos_z_AGN[0]), np.max(pos_z_AGN[0])]
    ylim = [np.min(pos_z_AGN[1]), np.max(pos_z_AGN[1])]
    label = r'Redshift $z<%.2f$, Halos $M_{500c}> 10^{%.1f} M_\odot$ '%(redshift_limit, np.log10(cluster_params[0]))
    
    setLabel(ax, 'R.A. (deg)', 'Dec (deg)', label, xlim, ylim, legend=False)    
    legend = ax.legend(loc='best',  fontsize=14)
    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(12)
        
    print('AGNs: %d, Host halos: %d, Sub-halos: %d'%(len(pos_z_AGN[0]), len(ra_cen), len(ra_sat)))
    return

def plotAGNfraction(pos_z_clu, pos_z_AGN, pos_z_gal, redshift_limit_agn):
    """
    Function to plot the agn fraction in the given pixel
    @pos_z_clu :: postion and redshifts of all the selected 'clusters'
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    @pos_z_gal :: postion and redshifts of all the selected galaxies
    @redshift_limit_agn :: upper limit on redshift based on the clusters found
    """
    fig, ax = plt.subplots(2,1,figsize=(7,14))
    
    # getting the useful histogram properties
    counts_agn, redshift_bins_agn, _ = ax[0].hist(pos_z_AGN[2], density=False, color='#ffffff', edgecolor='#ffffff')
    counts_gal, redshift_bins_gal, _ = ax[0].hist(pos_z_gal[2], density=False, color='#ffffff', edgecolor='#ffffff')

    # plotting the galaxy and agn distribution as a function of redshift
    ax[0].plot(redshift_bins_agn[1:], counts_agn, 'o', markerfacecolor='k', linestyle='--', color='grey',linewidth=2, markersize=7, label=r'AGNs')
    ax[0].plot(redshift_bins_gal[1:], counts_gal, 'o', markerfacecolor='b', linestyle='--', color='#68a6a5',linewidth=2, markersize=7, label=r'Galaxies')
    
    # axis properties - 0
    xlim = [np.min(redshift_bins_agn[1:]), np.max(redshift_bins_agn[1:])]
    setLabel(ax[0], r'Redshift$_R$', 'Counts', '$z<%.2f$'%redshift_limit_agn, xlim, 'default', legend=True)
    ax[0].set_yscale("log")

    # agn fraction as a function of redshift
    ax[1].plot(redshift_bins_gal[1:], ((counts_agn*100)/counts_gal), 'o', markerfacecolor='b', linestyle='--', color='#68a6a5',linewidth=2, markersize=5, label=r'$z<%.2f$'%redshift_limit_agn)
    
    # axis properties - 1
    xlim = [np.min(redshift_bins_agn[1:])-0.02, np.max(redshift_bins_agn[1:])]
    setLabel(ax[1], r'Redshift$_R$', 'AGN fraction (percent)', '', xlim, 'default', legend=True)
    return