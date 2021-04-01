# -*- coding: utf-8 -*-
"""Plotting.py for notebook 01_Exploring_DM_Halos

This python file contains all the functions used for plotting graphs and maps in the 1st notebook (.ipynb) of the repository: 01. Exploring parameters in DM halos and sub-halos

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 23rd February 2021
Last updated on 30th March 2021
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

def plotAgnClusterDistribution(pos_z_clu, pos_z_AGN, pos_z_halo, cluster_params):
    """
    Function to plot the AGN cluster distribution
    @pos_z_clu :: postion and redshifts of all the selected 'clusters'
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    @pos_z_gal :: postion and redshifts of all the selected galaxies
    """
    halo_m_500c = cluster_params[0]
    
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    # plotting halos
    halos = ax.plot(pos_z_halo[0], pos_z_halo[1], '.',  color='#fcd16d', markersize=0.2, label=r'All DM Halos', alpha=0.2)
    
    # plotting clusters
    cluster = ax.plot(pos_z_clu[0], pos_z_clu[1], 'o', color= '#03a351', markersize=3, label=r'Clusters $M_{500c}> 10^{%.1f} M_\odot$ '%(np.log10(halo_m_500c)))
    
    # plotting AGNs
    agn = ax.plot(pos_z_AGN[0], pos_z_AGN[1], '*',  color='k', markersize=3.5, label=r'AGN', alpha=0.7)

    # labeling axes and defining limits
    xlim = [np.min(pos_z_halo[0]), np.max(pos_z_halo[0])]
    ylim = [np.min(pos_z_halo[1]), np.max(pos_z_halo[1])]
    setLabel(ax, 'R.A. (deg)', 'Dec (deg)', '', xlim, ylim, legend=True)
    print('Redshift z<%.2f'%(np.max(pos_z_clu[2])))
    return

def plotHostSubHalos(pos_z_cen_halo, pos_z_sat_halo, pos_z_AGN):
    """
    Function to plot the host and satellite halo distribution
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them 
    --> divided into 3 because each hd_halo holds info on 1000 halos alone
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    """ 
    ra_cen, dec_cen = pos_z_cen_halo[0], pos_z_cen_halo[1]
    ra_sat, dec_sat = pos_z_sat_halo[0], pos_z_sat_halo[1]
    
    fig, ax = plt.subplots(1,1,figsize=(9,8)) 
    # plotting host halos
    host_halos = ax.plot(ra_cen, dec_cen, '.', color= 'k', markersize=0.06, label=r'Host-halos $P_{id}=-1$', alpha=0.4)
    
    # plotting sat halos
    sat_halos = ax.plot(ra_sat, dec_sat, 'o', color='#07d9f5', markersize=0.07, label=r'Satellite halos $P_{id} \neq -1$', alpha=0.7)
    
    # plotting AGNs
    agn = ax.plot(pos_z_AGN[0], pos_z_AGN[1], '*',  color='#fff717', markersize=6.5, label=r'AGN', markeredgecolor='w', markeredgewidth=0.4)

    # labeling axes and defining limits
    xlim = [np.min(pos_z_AGN[0]), np.max(pos_z_AGN[0])]
    ylim = [np.min(pos_z_AGN[1]), np.max(pos_z_AGN[1])]
    setLabel(ax, 'R.A. (deg)', 'Dec (deg)', '', xlim, ylim, legend=True)    
        
    print('AGNs: %d, Host (central) halos: %.2e, Sattelite halos: %.2e'%(len(pos_z_AGN[0]), len(ra_cen), len(ra_sat)))
    return

def plotAGNfraction(pos_z_AGN, pos_z_gal, redshift_limit_agn, bin_size):
    """
    Function to plot the agn fraction in the given pixel
    @pos_z_AGN :: postion and redshifts of all the selected AGNs
    @pos_z_gal :: postion and redshifts of all the selected galaxies
    @redshift_limit_agn :: upper limit on redshift based on the clusters found
    """
    fig, ax = plt.subplots(1,2,figsize=(19,7))
    
    # getting the useful histogram properties
    counts_agn, redshift_bins_agn = np.histogram(pos_z_AGN[2], bins = bin_size)
    counts_gal, redshift_bins_gal = np.histogram(pos_z_gal[2], bins = bin_size)
    
    # plotting the galaxy and agn distribution as a function of redshift    
    ax[0].plot(redshift_bins_gal[1:], counts_gal, 'ks', ms=4, label=r'DM Halos')
    ax[0].plot(redshift_bins_agn[1:], counts_agn, 'bs', ms=4, label=r'AGNs')
    
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
    ax[1].plot(z_bin_modified, f_agn, 's', color='#6b0385', ms=4)
    
    # axis properties - 1
    xlim = [np.min(redshift_bins_agn[1:])-0.02, np.max(redshift_bins_agn[1:])]
    setLabel(ax[1], r'Redshift$_R$', r'$f_{AGN}$ (%s)'%"%", '', xlim, 'default', legend=False)
    ax[1].set_yscale("log")
    
    plt.savefig('figures/agn_frac.pdf', facecolor='w', edgecolor='w')
    print( 'Reddhift z<%.2f'%redshift_limit_agn )
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
    ax1.plot(merger_val_gal, counts_gal, 'kx', label='DM Halos') 
    ax1.plot(merger_val_agn, counts_agn, 'bx', label='AGNs') 

    setLabel(ax1, r'Scale, $a(t)$, of last Major Merger', 'Counts', '', 'default', 'default', legend=True)
    ax.set_yscale("log")
    
    # setting the x-label on top (converting a to redshift)
    a_min, a_max = np.min(merger_val_gal), np.max(merger_val_gal)    
    scale_factor_arr = [a_max, a_min*4, a_min*2, a_min]
    ax2.set_xticks([(1/a) -1 for a in scale_factor_arr])
    
    ax2.invert_xaxis()
    ax2.set_xlabel('Redshift (z)')
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    print("Objects with merger redshifts z < %.2f"%z_at_value(cosmo.scale_factor, a_min))
    plt.savefig('figures/merger_distribution_z%.2f.pdf'%redshift_limit, facecolor='w', edgecolor='w')
    return 


def plotCentralSatelliteScaleMergers(cen_sat_AGN, cen_sat_halo, redshift_limit):
    """
    Function to plot the central and sattelite scale factors for mergers
    
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6))

    labels = [r'central AGNs', r'satellite AGNs', 'central DM halos', 'satellite  DM halos']
    c, m, ms = ['b', '#38cee8', 'k', 'grey'], ['^', '*', '^', '*'], [9, 15, 5, 9]
    mec, mew = ['w', 'k', 'k', '#abaeb3'], [0.7, 0.4, 1, 0.7]
    
    for i in [0, 1]:
        s_m_agn, c_agn = np.unique(cen_sat_AGN[i]['HALO_scale_of_last_MM'], return_counts=True)
        s_m_gal, c_gal = np.unique(cen_sat_halo[i]['HALO_scale_of_last_MM'], return_counts=True)
        
        # agns
        ax.plot(s_m_agn, c_agn, color=c[i], marker=m[i], ls='', ms=ms[i], label=labels[i], markeredgecolor=mec[i], markeredgewidth=mew[i])
        
        # DM halos
        j = i + 2
        ax.plot(s_m_gal, c_gal, color=c[j], marker=m[j], ls='', ms=ms[j], label=labels[j], markeredgecolor=mec[j], markeredgewidth=mew[j])
    
    # set label
    setLabel(ax, r'Scale, $a(t)$, of last Major Merger', 'Counts', '', 'default', 'default', legend=True)
    ax.set_yscale("log")

    plt.savefig('figures/merger_dist_cenAndsat_z%.2f.pdf'%redshift_limit, facecolor='w', edgecolor='w')
    print('Objects below z: ', redshift_limit)
    return [labels, c, m, ms, mec, mew]


def plotTimeSinceMergerDist(scale_merger_AGN, scale_merger_gal, z_AGN, z_gal, cosmo, bin_size, redshift_limit):
    """
    Plot the distribution of halos with respective galaxies & agns given the time since merger
    """
    # get the time difference since merger events in the halos
    t_merger_agn = edh.getMergerTimeDifference(scale_merger_AGN, z_AGN, cosmo)
    t_merger_gal = edh.getMergerTimeDifference(scale_merger_gal, z_gal, cosmo)

    # get the t since merger bins and counts
    if bin_size[0]:
        c_t_agn, merger_bins_agn = np.histogram(np.array(t_merger_agn), bins = bin_size[1])
        c_t_gal, merger_bins_gal = np.histogram(np.array(t_merger_gal), bins = bin_size[1])
        merger_bins_agn = merger_bins_agn[:-1]
        merger_bins_gal = merger_bins_gal[:-1]
        
    else:
        merger_bins_agn, c_t_agn = np.unique(t_merger_agn, return_counts=True)
        merger_bins_gal, c_t_gal = np.unique(t_merger_gal, return_counts=True)

    fig, ax = plt.subplots(1,1,figsize=(7,6))

    # plot the time since merger distribution for galaxies and agns
    ax.plot(merger_bins_gal, np.cumsum(c_t_gal), 'k^', label='DM Halos', ms=4) 
    ax.plot(merger_bins_agn, np.cumsum(c_t_agn), 'b^', label='AGNs', ms=4) 
    
    # set labels/legends
    setLabel(ax, r'$\Delta t_{merger} = t(z_{merger})-t(z_{current})$ [Gyr]', 'Cumulative counts', '', 'default', 'default', legend=False)
    
    ax.legend(loc='lower left',  fontsize=14)
    ax.set_yscale("log")
    ax.set_xscale("log")    
    return ax, fig, t_merger_agn, t_merger_gal


def mergerRedshiftPlot(cen_sat_AGN, cen_sat_halo, dt_m, plot_params, redshift_limit):
    """
    Function to plot the time since merger as a function of the redshift
    @cen_sat_AGN(gal) :: handels to access the central and satellite AGNs(galaxies)
    @dt_m :: time difference after merger for cen/sat AGNs(galaxies)
    @plot_params :: to keep consistency between plots, array containing [labels, c, m, ms]
    """
    fig, ax = plt.subplots(1,1,figsize=(7,6)) 
    
    # change marker size for central DM halos
    plot_params[3][1] = 9
    
    z_R = [cen_sat_AGN[0]['redshift_R'], cen_sat_AGN[1]['redshift_R'], cen_sat_halo[0]['redshift_R'], cen_sat_halo[1]['redshift_R']]
    
    # plot central, satellite merger distributions as per visual preference
    for i in [2, 3, 0, 1]:
        ax.plot(dt_m[i], z_R[i], plot_params[2][i], color=plot_params[1][i], ms=plot_params[3][i], label=plot_params[0][i], markeredgecolor=plot_params[4][i], markeredgewidth=plot_params[5][i])
    
    # set labels/legends
    setLabel(ax, r'$\Delta t_{merger} = t(z_{merger})-t(z_{current})$ [Gyr]', r'Redshift$_R$', '', 'default', 'default', legend=True)
    ax.set_xscale("log")
    plt.savefig('figures/t_since_merger_z_plot_%.2f.pdf'%redshift_limit, facecolor='w', edgecolor='w')    
    return ax


def plotMergerTimeCuts(ax, t_merger_cut_arr, l):
    """
    Function to plot the defined cuts in merger times within the concerned plot
    @t_merger_cut_arr :: array that defines the cuts in the merger times
    @l :: array that defines the linestyles used to denote these cuts (refer to the initial codeblock in the notebook)
    """
    for i, t_m_cut in enumerate(t_merger_cut_arr):
        ax.axvline(x=t_m_cut, color='r', linestyle= l[i], label='%.1f Gyr'%t_m_cut)

    ax.legend(fontsize=14, loc='lower left') 
    return