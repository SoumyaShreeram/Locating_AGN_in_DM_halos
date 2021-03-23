# -*- coding: utf-8 -*-
"""Plotting.py
## Plotting functions for simulating stars using Slitless-Spectroscopy
This python file contains all the functions used for plotting graphs and maps across the various notebooks in the repository.

**Script written by**: Soumya Shreeram <br>
**Project supervised by**: Johan Comparat <br>
**Date**: 23rd February 2021
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
        l = ax.legend(loc='best',  fontsize=14)
        for legend_handle in l.legendHandles:
            legend_handle._legmarker.set_markersize(12)
            
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
    galaxy = ax.plot(pos_z_gal[0], pos_z_gal[1], '.',  color='#fcd16d', markersize=0.8, label=r'Galaxies', alpha=0.2)
    
    # plotting clusters
    cluster = ax.plot(pos_z_clu[0], pos_z_clu[1], 'o', color= '#03a351', markersize=3, label=r'Halos $M_{500c}> 10^{%.1f} M_\odot$ '%(np.log10(min_cluster_mass)))
    
    # plotting AGNs
    agn = ax.plot(pos_z_AGN[0], pos_z_AGN[1], '.',  color='k', markersize=2.5, label=r'AGN', alpha=0.7)

    # labeling axes and defining limits
    xlim = [np.min(pos_z_gal[0]), np.max(pos_z_gal[0])]
    ylim = [np.min(pos_z_gal[1]), np.max(pos_z_gal[1])]
    setLabel(ax, 'R.A. (deg)', 'Dec (deg)', 'Redshift $z<%.2f$'%(np.max(pos_z_clu[2])), xlim, ylim, legend=True)
    return

def plotHostSubHalos(pos_z_cen, pos_z_sat, pos_z_AGN, redshift_limit, cluster_params):
    """
    Function to plot the host and satellite halo distribution
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them 
    --> divided into 3 because each hd_halo holds info on 1000 halos alone
    @cluster_params :: contains clu_FX_soft, galaxy_mag_r, min_cluster_mass where
        @min_cluster_mass :: min mass for halo to be called a cluster
    @redshift_limit :: upper limit on redshift
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    """ 
    ra_cen, dec_cen = pos_z_cen[0], pos_z_cen[1]
    ra_sat, dec_sat = pos_z_sat[0], pos_z_sat[1]
    
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    # plotting host halos
    host_halos = ax.plot(ra_cen, dec_cen, 'o', color= 'grey', markersize=2.5, label=r'Host-halos $P_{id}=-1$')
    
    # plotting sat halos
    sat_halos = ax.plot(ra_sat, dec_sat, 'o', color= 'b', markersize=5, label=r'Satellite halos $P_{id} \neq -1$')
    
    
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
        
    print('AGNs: %d, Host (central) halos: %d, Sattelite halos: %d'%(len(pos_z_AGN[0]), len(ra_cen), len(ra_sat)))
    return

def plotAGNfraction(pos_z_clu, pos_z_AGN, pos_z_gal, redshift_limit_agn, bin_size):
    """
    Function to plot the agn fraction in the given pixel
    @pos_z_clu :: postion and redshifts of all the selected 'clusters'
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    @pos_z_gal :: postion and redshifts of all the selected galaxies
    @redshift_limit_agn :: upper limit on redshift based on the clusters found
    """
    fig, ax = plt.subplots(1,2,figsize=(17,7))
    
    # getting the useful histogram properties
    counts_agn, redshift_bins_agn = np.histogram(pos_z_AGN[2], bins = bin_size)
    counts_gal, redshift_bins_gal = np.histogram(pos_z_gal[2], bins = bin_size)
    
    # plotting the galaxy and agn distribution as a function of redshift    
    ax[0].plot(redshift_bins_gal[1:], counts_gal, 'bs', ms=4, label=r'Galaxies')
    ax[0].plot(redshift_bins_agn[1:], counts_agn, 'ks', ms=4, label=r'AGNs')
    
    # axis properties - 0
    xlim = [np.min(redshift_bins_agn[1:]), np.max(redshift_bins_agn[1:])]
    setLabel(ax[0], r'Redshift$_R$', 'Counts','', xlim, 'default', legend=True)
    ax[0].set_yscale("log")

    # agn fraction as a function of redshift
    f_agn, idx = [], []
    for c, c_gal in enumerate(counts_gal):
        if c_gal != 0:
            f_agn.append(((counts_agn[c]*100)/c_gal))
            idx.append(c)
    z_bin_modified = redshift_bins_gal[1:][np.array(idx)]
    
    # plot agn fraction
    ax[1].plot(z_bin_modified, f_agn, 'rs', ms=4, label=r'$z<%.2f$'%redshift_limit_agn)
    
    # axis properties - 1
    xlim = [np.min(redshift_bins_agn[1:])-0.02, np.max(redshift_bins_agn[1:])]
    setLabel(ax[1], r'Redshift$_R$', 'AGN fraction (percent)', '', xlim, 'default', legend=True)
    ax[1].set_yscale("log")
    
    plt.savefig('figures/agn_frac.pdf', facecolor='w', edgecolor='w')
    print( 'z<%.2f'%redshift_limit_agn )
    return redshift_bins_gal[1:]

def plotRedshiftComovingDistance(cosmo, redshift_limit, resolution = 0.0001):
    """Function to plot the relation between redshift and the comoving distance
    @cosmo :: cosmology package loaded
    @redshift_limit :: upper limit in redshift --> end point for interpolation
    @resolution :: resolution of time steps (set to e-4 based of simulation resolution)
    
    @Returns :: plot showing the dependence of redshift on comoving distance
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))

    distance_Mpc = cosmo.comoving_distance(np.arange(0,redshift_limit, resolution))
    redshifts = np.arange(0,redshift_limit, resolution)

    ax.plot(redshifts, distance_Mpc, 'k.', ms=1)
    setLabel(ax, 'Redshift (z)', 'Comoving distance (Mpc)', '', 'default', 'default', legend=False)
    print('Redshift-Comoving distance relationship')
    return

def plotMergerDistribution(merger_val_gal, counts_gal, merger_val_agn, counts_agn, cosmo, redshift_limit):
    """
    Function to plot the distribution (counts) of the merger scale factor/redshift 
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    # plot the merger distribution for galaxies and agns
    ax1.plot(merger_val_gal, counts_gal, 'bx', label='Galaxies') 
    ax1.plot(merger_val_agn, counts_agn, 'kx', label='AGNs') 

    setLabel(ax1, r'Scale, $a(t)$, of last Major Merger', 'Counts', '', 'default', 'default', legend=True)
    ax.set_yscale("log")
    
    # setting the x-label on top
    a_min, a_max = np.min(merger_val_gal), np.max(merger_val_gal)
    scale_factor_arr = [a_min, a_min*2, a_min*4, a_max]
    ax2.set_xticks([z_at_value(cosmo.scale_factor, a) for a in scale_factor_arr])
    ax2.set_xlabel('Redshift (z)')
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    print("Objects with merger redshifts z < %.2f"%z_at_value(cosmo.scale_factor, a_min))
    plt.savefig('figures/merger_distribution_z%.2f.pdf'%redshift_limit, facecolor='w', edgecolor='w')
    return 

def plotTimeSinceMergerDist(scale_merger_AGN, scale_merger_gal, pos_z_AGN, pos_z_gal, cosmo, bin_size, redshift_limit):
    """
    Plot the distribution of halos with respective galaxies & agns given the time since merger
    """
    # get the time difference since merger events in the halos
    t_merger_agn = edh.getMergerTimeDifference(scale_merger_AGN, pos_z_AGN[2], cosmo)
    t_merger_gal = edh.getMergerTimeDifference(scale_merger_gal, pos_z_gal[2], cosmo)

    # get the t since merger bins and counts
    t_merger_bins_agn, counts_t_merger_agn = np.histogram(t_merger_agn, bins = bin_size)
    t_merger_bins_gal, counts_t_merger_gal = np.histogram(t_merger_gal, bins = bin_size)

    fig, ax = plt.subplots(1,1,figsize=(7,6))

    # plot the time since merger distribution for galaxies and agns
    ax.plot(t_merger_bins_gal, np.cumsum(counts_t_merger_gal[:-1]), 'b^', label='Galaxies', ms=4) 
    ax.plot(t_merger_bins_agn, np.cumsum(counts_t_merger_agn[:-1]), 'k^', label='AGNs', ms=4) 

    # set labels/legends
    setLabel(ax, r'$\Delta t_{merger} = t(z_{merger})-t(z=0)$ [Gyr]', 'Cumulative counts', '', 'default', 'default', legend=True)
    
    ax.set_xscale("log")
    plt.savefig('figures/t_since_merger_distribution_z%.2f.pdf'%redshift_limit, facecolor='w', edgecolor='w')
    return ax