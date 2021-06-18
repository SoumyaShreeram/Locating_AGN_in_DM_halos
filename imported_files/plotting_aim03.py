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

import Modelling_AGN_fractions_from_literature as mafl

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

def plotSatyapal(ax, Satyapal_14, r_p_err_S14, f_agn_err_S14, color_S14):
    "Plot taken from Satyapal et al. 2014"
    r_p_S14, f_agn_S14 = mafl.getXY(Satyapal_14)
    ax.plot(r_p_S14, f_agn_S14, 'o', label='Satyapal et al. 2014', color=color_S14, ms=9, mec='#8637b8')

    xerr, yerr = mafl.getErr(r_p_err_S14, r_p_S14, f_agn_err_S14, f_agn_S14)
    ax.errorbar(r_p_S14, f_agn_S14, yerr = yerr, fmt='none', ecolor='#8637b8', capsize=2)
    return ax, np.array([r_p_S14, f_agn_S14, xerr, yerr], dtype=object)

def plotLiu(ax, r_p_L12, f_agn_L12, Liu_12_err, color_E11):
    """
    Plot taken from Liu et al 2012
    """
    ax.plot(r_p_L12, f_agn_L12, 'd', label='Liu et al. 2012', color=color_E11, ms=9, mec='#487ab8')

    yerr = np.abs(np.transpose(Liu_12_err)-f_agn_L12)
    ax.errorbar(r_p_L12, f_agn_L12, yerr = yerr, fmt='none', ecolor='#487ab8', capsize=2)
    return ax, np.array([r_p_L12, f_agn_L12, [1e-3*np.ones(len(yerr[0])), 1e-3*np.ones(len(yerr[0]))], yerr], dtype=object) 

def plotSilverman(ax, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, color_Sil11, xmax=150):
    """
    Plot taken from Silverman et al. 2011 
    """
    r_p_Sil11, f_agn_Sil11 = mafl.getXY(Silverman_11)
    excess = ax.plot(r_p_Sil11, f_agn_Sil11, 'o', label='Silverman et al. 2011', color=color_Sil11, ms=9, mec='k')
    control = ax.hlines(0.05, 0, xmax, colors='k', linestyles=':')

    xerr, yerr = mafl.getErr(r_p_err_Sil11, r_p_Sil11, f_agn_err_Sil11, f_agn_Sil11)
    ax.errorbar(r_p_Sil11, f_agn_Sil11, yerr = yerr, xerr = xerr, fmt='none', ecolor='k', capsize=2)   
    return ax, np.array([r_p_Sil11, f_agn_Sil11, xerr, yerr], dtype=object)

def plotEllison(ax, r_p_E11, f_agn_E11, r_p_err_E11, f_agn_err_E11, color_E11, xmax=150, mec_E11 = '#0b8700'):
    """
    Plot taken from Ellison et al. 2011
    """
    ax.plot(r_p_E11, f_agn_E11, 'o', label='Ellison et al. 2011', color=color_E11, ms=9, mec=mec_E11)
    control = ax.hlines(0.0075, 0, xmax, colors=mec_E11, linestyles='--')
    
    # errorbars
    xerr, yerr = mafl.getErr(r_p_err_E11, r_p_E11, f_agn_err_E11, f_agn_E11)
    ax.errorbar(r_p_E11, f_agn_E11, yerr = yerr, xerr = xerr, fmt='none', ecolor=mec_E11, capsize=2)
    return ax, np.array([r_p_E11, f_agn_E11, xerr, yerr], dtype=object)

def plotAllLiteraturePlots(Satyapal_14, r_p_err_S14, f_agn_err_S14, r_p_L12, f_agn_L12, Liu_12_err, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, r_p_E11, f_agn_E11, r_p_err_E11, f_agn_err_E11, ax = None, xmax= 150,ymax = 0.22):
    """
    Function plots all the data points obtained from literature 
    """
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(7,6))
    
    
    ax.set_xticks(ticks=np.arange(0, xmax, step=10), minor=True)
    ax.set_yticks(ticks=np.arange(0, ymax, step=1e-2), minor=True)
    
    color_E11, color_S14, color_Sil11 ='#d5ff03', '#ff6803', '#ff0318'
    

    # Satyapal et al. 2014
    ax, Satyapal_14_all = plotSatyapal(ax, Satyapal_14, r_p_err_S14, f_agn_err_S14, color_S14)
    
    # Liu et al. 2012
    ax, Liu_12_all = plotLiu(ax, r_p_L12, f_agn_L12, Liu_12_err, color_E11)
    
    # Silverman et al 2011
    ax, Silverman_11_all = plotSilverman(ax, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, color_Sil11)
    
    # Ellison et al. 2011
    ax, Ellison_11_all = plotEllison(ax, r_p_E11, f_agn_E11, r_p_err_E11, f_agn_err_E11, color_E11)
    
    setLabel(ax, r'Projected separation, $r_{\rm p}$ [kpc]', r'Fraction of AGNs, $f_{\rm AGN}$', xlim=[0, xmax], ylim=[0, ymax])
    plt.savefig('../figures/close_p_lit_combined.pdf', facecolor='w', edgecolor='w', bbox_inches='tight')
    return ax, np.array([Satyapal_14_all, Liu_12_all, Silverman_11_all, Ellison_11_all], dtype=object)