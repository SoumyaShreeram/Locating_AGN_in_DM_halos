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
import sys

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
from scipy import interpolate

sys.path.append('../imported_files/')
import Scaling_relations as sr

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

def saveFig(filename):
    plt.savefig('../figures/'+filename, facecolor='w', edgecolor='w', bbox_inches='tight')
    return

    
def plotBinnedM500cLx(ax, scaled_Lx, pixel_no='000000', c='k', label='clusters only',\
 full_sky=True, ls='-', dlog_M500c=0.05, model_name='Model_A0', frac_cp=0.2):
    """
    Function to bin the M500c and rest frame x-ray luminosity, Lx
    """
    # defining the bins in Lx and M500c
    log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = sr.getBinsLxM500c(scaled_Lx,\
     pixel_no=pixel_no, full_sky=full_sky, dlog_M500c=dlog_M500c,\
     model_name=model_name, frac_cp=frac_cp)
    
    ll_log_Lx, ul_log_Lx =  log_Lx_mean - log_Lx_std,log_Lx_mean + log_Lx_std
    
    _ = ax.fill_between(log_M500c_bins, ll_log_Lx, ul_log_Lx, color=c, alpha=0.1)
    ax.plot(log_M500c_bins+dlog_M500c/2., log_Lx_mean, color=c, lw=2, label=label, ls=ls)
    
    xlim = [np.round(np.min(log_M500c_bins), 0), np.round(np.max(log_M500c_bins), 0)]
    return ax, xlim

def plotBinnedM500cLxDifference(ax, scaled_Lx, unscales_Lx, pixel_no='000000', c='k', label='clusters only',\
 full_sky=True, ls='-', dlog_M500c=0.15,  model_name='Model_A0', frac_cp=0.2, plot_label=False):
    """
    Function to bin the M500c and rest frame x-ray luminosity, Lx
    """
    # defining the bins in Lx and M500c
    scaled = sr.getBinsLxM500c(scaled_Lx, pixel_no=pixel_no, full_sky=full_sky,\
    dlog_M500c=dlog_M500c, model_name=model_name, frac_cp=frac_cp)
    log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = scaled

    og = sr.getBinsLxM500c(unscales_Lx, pixel_no=pixel_no, full_sky=full_sky,\
    dlog_M500c=dlog_M500c, model_name=model_name, frac_cp=frac_cp)
    log_M500c_bins_og, dlog_M500c_og, log_Lx_mean_og, log_Lx_std_og = og
    print(log_M500c_bins, len(log_Lx_mean))
    print(log_Lx_mean)
    
    ax.plot(log_M500c_bins+dlog_M500c/2, log_Lx_mean-log_Lx_mean_og, color=c, lw=2.5, label=label, ls=ls, zorder=2)
    
    xlim = [np.round(np.min(log_M500c_bins), 0), np.round(np.max(log_M500c_bins), 0)]
    return ax, xlim, np.array(log_Lx_mean-log_Lx_mean_og)

def plotBinnedM500cLxScatter(ax, scaled_Lx, unscales_Lx, pixel_no='000000', c='k', label='clusters only',\
 full_sky=True, ls='-', dlog_M500c=0.15,  model_name='Model_A0', frac_cp=0.2, plot_label=False):
    """
    Function to bin the M500c and rest frame x-ray luminosity, Lx
    """
    # defining the bins in Lx and M500c
    scaled = sr.getBinsLxM500c(scaled_Lx, pixel_no=pixel_no, full_sky=full_sky,\
    dlog_M500c=dlog_M500c, model_name=model_name, frac_cp=frac_cp)
    log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = scaled

    og = sr.getBinsLxM500c(unscales_Lx, pixel_no=pixel_no, full_sky=full_sky,\
    dlog_M500c=dlog_M500c, model_name=model_name, frac_cp=frac_cp)
    log_M500c_bins_og, dlog_M500c_og, log_Lx_mean_og, log_Lx_std_og = og

    # print the mean increase in scatter in the low mass/high mass end
    grp_idx, clu_idx = np.where(log_M500c_bins<=14)[0], np.where(log_M500c_bins>=14)[0]
    groups_sigma = log_Lx_mean[grp_idx]-log_Lx_mean_og[grp_idx]
    clusters_sigma = log_Lx_mean[clu_idx]-log_Lx_mean_og[clu_idx]
    print('groups: %.3f'%np.mean(groups_sigma), '+/- %.3f'%np.std(groups_sigma))
    print('clusters: %.3f'%np.mean(clusters_sigma), '+/- %.3f'%np.std(clusters_sigma))

    # plot the mean increase in scatter in the low mass end
    if plot_label:
        ax.plot(log_M500c_bins+dlog_M500c/2, log_Lx_std_og, 'k-.', lw=1, zorder=2, label='clusters only')
    else:
        ax.plot(log_M500c_bins+dlog_M500c/2, log_Lx_std_og, 'k-.', lw=1, zorder=2)
    
    ax.plot(log_M500c_bins+dlog_M500c/2, log_Lx_std, color=c, lw=2.5, label=label, ls=ls, zorder=2)
    
    #print('mean scatter/std:', np.mean(frac_log_Lx_std))
    xlim = [np.round(np.min(log_M500c_bins), 0), np.round(np.max(log_M500c_bins), 0)]
    return ax, xlim