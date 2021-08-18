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
import pandas as pd

# scipy modules
from scipy.spatial import KDTree
from scipy import interpolate 

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
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import All_sky as sky


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

def saveFig(filename):
    plt.savefig('../figures/'+filename, facecolor='w', edgecolor='w', bbox_inches='tight')
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
    ax.plot(r_p_S14, f_agn_S14, 'o', label='Satyapal et al. 2014', color=color_S14, ms=9, mec='#8637b8', zorder=2)

    xerr, yerr = mafl.getErr(r_p_err_S14, r_p_S14, f_agn_err_S14, f_agn_S14)
    ax.errorbar(r_p_S14, f_agn_S14, yerr = yerr, fmt='none', ecolor='#210340', capsize=2, zorder=2)
    return ax, np.array([r_p_S14, f_agn_S14, xerr, yerr], dtype=object)

def plotLiu(ax, r_p_L12, f_agn_L12, Liu_12_err, color_E11):
    """
    Plot taken from Liu et al 2012
    """
    ax.plot(r_p_L12, f_agn_L12, 'd', label='Liu et al. 2012', color=color_E11, ms=9,\
     mec='#487ab8', zorder=2)

    yerr = np.abs(np.transpose(Liu_12_err)-f_agn_L12)
    ax.errorbar(r_p_L12, f_agn_L12, yerr = yerr, fmt='none', ecolor='#759c00',\
     capsize=2, zorder=2)
    return ax, np.array([r_p_L12, f_agn_L12, [1e-3*np.ones(len(yerr[0])), 1e-3*np.ones(len(yerr[0]))], yerr], dtype=object) 

def plotSilverman(ax, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, color_Sil11, xmax=150):
    """
    Plot taken from Silverman et al. 2011 
    """
    r_p_Sil11, f_agn_Sil11 = mafl.getXY(Silverman_11)
    excess = ax.plot(r_p_Sil11, f_agn_Sil11, 'o', label='Silverman et al. 2011',\
     color=color_Sil11, ms=9, mec='k', zorder=2)
    control = ax.hlines(0.05, 0, xmax, colors='k', linestyles=':', zorder=2)

    xerr, yerr = mafl.getErr(r_p_err_Sil11, r_p_Sil11, f_agn_err_Sil11, f_agn_Sil11)
    ax.errorbar(r_p_Sil11, f_agn_Sil11, yerr = yerr, xerr = xerr, fmt='none',\
     ecolor='k', capsize=2, zorder=2)   
    return ax, np.array([r_p_Sil11, f_agn_Sil11, xerr, yerr], dtype=object)

def plotEllison(ax, r_p_E11, f_agn_E11, r_p_err_E11, f_agn_err_E11, color_E11, xmax=150, mec_E11 = '#0b8700'):
    """
    Plot taken from Ellison et al. 2011
    """
    ax.plot(r_p_E11, f_agn_E11, 'o', label='Ellison et al. 2011', color=color_E11, ms=9,\
     mec=mec_E11, zorder=2)
    control = ax.hlines(0.0075, 0, xmax, colors=mec_E11, linestyles='--', zorder=2)
    
    # errorbars
    xerr, yerr = mafl.getErr(r_p_err_E11, r_p_E11, f_agn_err_E11, f_agn_E11)
    ax.errorbar(r_p_E11, f_agn_E11, yerr = yerr, xerr = xerr, fmt='none', ecolor=mec_E11, capsize=2)
    return ax, np.array([r_p_E11, f_agn_E11, xerr, yerr], dtype=object)

def plotAllLiteraturePlots(Satyapal_14, r_p_err_S14, f_agn_err_S14, r_p_L12, f_agn_L12,\
 Liu_12_err, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, r_p_E11, f_agn_E11, r_p_err_E11,\
  f_agn_err_E11, axs = None, xmax= 150,ymax = 0.22):
    """
    Function plots all the data points obtained from literature 
    """
    if np.any(axs) == None:
        fig, axs = plt.subplots(1,2,figsize=(15,6))
        fig.patch.set_facecolor('white')
    ax, ax1 = axs[0], axs[1]
    
    ax.set_xticks(ticks=np.arange(0, xmax, step=10), minor=True)
    ax.set_yticks(ticks=np.arange(0, ymax, step=1e-2), minor=True)
    
    color_E11, color_S14, color_Sil11 ='#d5ff03', '#ff6803', '#ff0318'
    

    # Satyapal et al. 2014
    ax, Satyapal_14_all = plotSatyapal(ax, Satyapal_14, r_p_err_S14, f_agn_err_S14, color_S14)
    
    # Liu et al. 2012
    ax, Liu_12_all = plotLiu(ax, r_p_L12, f_agn_L12, Liu_12_err, color_E11)
    
    # Silverman et al 2011
    ax1, Silverman_11_all = plotSilverman(ax1, Silverman_11, r_p_err_Sil11, f_agn_err_Sil11, color_Sil11)
    
    # Ellison et al. 2011
    ax, Ellison_11_all = plotEllison(ax, r_p_E11, f_agn_E11, r_p_err_E11, f_agn_err_E11, color_E11)
    
    setLabel(ax, r'Projected separation, $r_{\rm p}$ [kpc]', r'Fraction of AGNs, $f_{\rm AGN}$', title='z<0.2', xlim=[0, xmax], ylim=[0, ymax])
    setLabel(ax1, r'Projected separation, $r_{\rm p}$ [kpc]', '', title='z<1', xlim=[0, xmax], ylim=[0, ymax])
    plt.savefig('../figures/close_p_lit_combined.pdf', facecolor='w', edgecolor='w', bbox_inches='tight')
    return ax, np.array([Satyapal_14_all, Liu_12_all, Silverman_11_all, Ellison_11_all], dtype=object)


def plotChangesCatAGN(ax, g_cp, g_rand, redshift_limit=.2, c='r', label_idx = 3, num_rp_bins=12,frac_cp_agn=0.03):
    """
    Function to see the changes in the new AGN cat wrt the old one 
    @c :: color of the lines
    """
    # get shell volume and projected radius bins
    r_p, shell_volume = aimm.shellVolume()
    r_p_half, shell_volume_half = aimm.shellVolume(num_bins=num_rp_bins)
    
    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()

    halo_lens = np.load('../Data/all_sky_halo_lengths_z%.1f.npy'%redshift_limit)
    rand_agn, cp_agn = mafl.getAGNlengths(redshift_limit=redshift_limit, frac_cp_agn=frac_cp_agn, all=False)
    
    # get the total number of possible AGN-halo pairs
    data_dir = '../Data/pairs_z%.1f/Major_dv_pairs/'%1
    gamma_all = np.load(data_dir+'gamma_all_pixels.npy', allow_pickle=True)

    r_kpc, r_kpc_half = (1e3*r_p[1:]), (1e3*r_p_half[1:])
    
    
    label1 = r'$t_{\rm MM}, \tilde{X}_{\rm off}$ assigned AGN-halo pairs'
    label2 = 'randomly assigned AGN-halo pairs'
    label3 = 'all halo-halo pairs'
    # --------- plot 1 -------------------------
    l1, = ax[0].plot(r_kpc, g_cp['Gamma_mean_CP'], '-', color=c, label=label1)
    l2, = ax[0].plot(r_kpc_half, g_rand['Gamma_mean_RAND'], '--', color=c, label=label2)
    l3, = ax[0].plot(r_kpc, gamma_all[0], ':', color=c, label=label3)

    ax[0].set_yscale('log')
    #xlim = [(1e3*np.min(r_p[1:])), (1e3*np.max(r_p[1:]))]
    
    xlabel, t = r'Separation, $r$ [kpc]', r'z<%.1f, $f_{\rm cp AGN}$ = %.2f'%(redshift_limit,frac_cp_agn)
    ylabel = r'$\Gamma_{\rm t_{\rm MM}; \ \tilde{X}_{\rm off}}(m;\ \Delta v)$ [Mpc$^{-3}$]'
    
    if label_idx == 3: 
        handles, labels = [l1, l2, l3], [label1, label2 ,label3 ]
        l = ax[0].legend(handles, labels, loc='best',  fontsize=14, frameon=False)        
    setLabel(ax[0], xlabel, ylabel, title=t, legend=True)

    # --------- plot 2 -------------------------
    # interpolate the halo-halo numberty to divide by old agn
    f = interpolate.interp1d(r_kpc, gamma_all[0])
    f_err = interpolate.interp1d(r_kpc, gamma_all[1])
    gamma_all_inter = np.array([f(r_kpc_half), f_err(r_kpc_half)])

    ax[1].plot(r_kpc, (g_cp['Gamma_mean_CP']/gamma_all[0]), '-', color=c)
    ax[1].plot(r_kpc_half, (g_rand['Gamma_mean_RAND']/gamma_all_inter[0]), '--', color=c)
    
    ylabel1 = r'$f_{\rm AGN}$'
    ax[1].set_yscale('log')
    setLabel(ax[1], xlabel, ylabel1, title=t, legend=False)
    return gamma_all_inter

def getFracError(da, a, db, b):
    return a/b, (a/b)*np.sqrt( (da/a)**2 + (db/b)**2)

def plotErrors(ax, r_p, r_p_half, g_cp_z1 , g_rand_z1 ):
    # --- plot to the left ----------
    top, bottom = g_cp_z1['Gamma_mean_CP']+g_cp_z1['Gamma_std_CP']/2, g_cp_z1['Gamma_mean_CP']-g_cp_z1['Gamma_std_CP']/2
    ax[0].fill_between((1e3*r_p[1:]), top, bottom, color='#5b7c85', alpha=0.3)

    top_r, bottom_r = g_rand_z1['Gamma_mean_RAND']+g_rand_z1['Gamma_std_RAND']/2, g_rand_z1['Gamma_mean_RAND']-g_rand_z1['Gamma_std_RAND']/2
    ax[0].fill_between((1e3*r_p_half[1:]), top_r, bottom_r, color='#5b7c85', alpha=0.3)

    top_all, bottom_all = g_cp_z1['Gamma_meanALL']+g_cp_z1['Gamma_stdALL']/2, g_cp_z1['Gamma_meanALL']-g_cp_z1['Gamma_stdALL']/2
    ax[0].fill_between((1e3*r_p[1:]), top_all, bottom_all, color='#5b7c85', alpha=0.3)

    # get fractional errors
    f_cp, err_cp = getFracError(g_cp_z1['Gamma_std_CP'], g_cp_z1['Gamma_mean_CP'], g_cp_z1['Gamma_stdALL'], g_cp_z1['Gamma_meanALL'])
    f_rand, err_rand = getFracError(g_rand_z1['Gamma_std_RAND'], g_rand_z1['Gamma_mean_RAND'], g_rand_z1['Gamma_stdALL'], g_rand_z1['Gamma_meanALL'])
    
    t, b = f_cp+err_cp/2, f_cp-err_cp/2
    ax[1].fill_between((1e3*r_p[1:]), t, b, color='#5b7c85', alpha=0.3)

    t_r, b_r = f_rand+err_rand/2, f_rand-err_rand/2
    ax[1].fill_between((1e3*r_p_half[1:]),t_r, b_r, color='#5b7c85', alpha=0.3)

    col_0 = Column(data=err_cp, name='frac_std_CP')
    col_1 = Column(data=err_rand, name='frac_std_RAND')
    g_cp_z1.add_column(col_0)
    g_rand_z1.add_column(col_1)
    return g_cp_z1, g_rand_z1

def label(t, x, offset):
    label_tmm = r'$\langle t_{\rm MM}^{(%d)}$'%t
    label_xoff = r'$ + \tilde{X}_{\rm off}^{(%d)}\rangle$'%x
    if offset == 0:
        label=label_tmm + label_xoff
    if offset != 0:
        label=label_tmm + label_xoff + ' + %.2f'%offset
    return label


def plotModels(axs, models, std, r_kpc, left=True, asymotote_value=[0.01, 0.05]):
    pal = sns.color_palette("Wistia", models.shape[0]+1).as_hex()
    greys = sns.color_palette("Wistia", models.shape[0]+1).as_hex()
    
    for i in range(models.shape[0]):
        if left:
            m, s = mafl.normalizeAsymptote(models[i], asymotote_value=asymotote_value[0]), std[i]
        
            axs[0].plot(r_kpc[1:], m[1:], color='k', lw=0.3, alpha=0.2, zorder=1)
            top, bottom = m.astype(None) + s.astype(None)/2, m.astype(None) - s.astype(None)/2
            axs[0].fill_between(r_kpc[1:],  top[1:], bottom[1:], color=pal[i], alpha=0.09, zorder=1)

        else:
            m, s = mafl.normalizeAsymptote(models[i], asymotote_value=asymotote_value[1]), std[i]
        
            axs[1].plot(r_kpc[1:], m[1:], color='k', lw=0.4, alpha=0.2, zorder=1)
            top, bottom = m.astype(None) + s.astype(None)/2, m.astype(None) - s.astype(None)/2
            axs[1].fill_between(r_kpc[1:],  top[1:], bottom[1:], color=greys[i], alpha=0.09, zorder=1)
    return axs

def plotAGNModelZ(axs, idx, r_p, g_cp_z0_2, ls='-', redshift_limit=2, c='b', if_cp=True):
    # fraction of agn line
    if if_cp:
        axs[idx].plot((1e3*r_p[1:]), g_cp_z0_2['Gamma_mean_CP']/g_cp_z0_2['Gamma_meanALL'], \
            ls=ls, lw=2, color=c, label='AGN model z<%.1f'%redshift_limit)
        
        # fill the standard deviation
        m, std = g_cp_z0_2['Gamma_mean_CP']/g_cp_z0_2['Gamma_meanALL'], g_cp_z0_2['frac_std_CP']
        axs[idx].fill_between((1e3*r_p[1:]), m-std/2, m+std/2, color='#5b7c85', alpha=0.4 )
    else:
        axs[idx].plot((1e3*r_p[1:]), g_cp_z0_2['Gamma_mean_RAND']/g_cp_z0_2['Gamma_meanALL'], \
            ls=ls, lw=2, color=c)
        
        # fill the standard deviation
        m, std = g_cp_z0_2['Gamma_mean_RAND']/g_cp_z0_2['Gamma_meanALL'], g_cp_z0_2['frac_std_RAND']
        axs[idx].fill_between((1e3*r_p[1:]), m-std/2, m+std/2, color='#5b7c85', alpha=0.4 )
    return axs

def plotMSEdist(ax, mse, pal, i, names, label):
    ax.plot(mse, lw=2, ls='--', color=pal[i], zorder=1)
    
    min_mse_idx = np.where(mse == np.min(mse))
    min_x = np.arange(len(mse))[min_mse_idx]
    min_model_name = names[i][1][min_mse_idx[0][0]]
    label = label[i]+min_model_name
    ax.plot(min_x, np.min(mse), 's', ms=12, mec='k', mew=.5, color=pal[i], label=label, zorder=2)
    return ax, min_mse_idx[0][0], label


def plotSelectedModels(axs, selected_A, selected_B, r_kpc):
    "Function plots all the physical models"
    # overplot with the data
    pt.plotModels(axs, select_E11_A[0], select_E11_A[1], r_kpc)
    pt.plotModels(axs, select_S14_A[0], select_S14_A[1], r_kpc)
    pt.plotModels(axs, select_L12_A[0], select_L12_A[1], r_kpc)
    pt.plotModels(axs, select_Sil11_A[0], select_Sil11_A[1], r_kpc, left=False)

    pt.plotModels(axs, select_E11_B[0], select_E11_B[1], r_kpc)
    pt.plotModels(axs, select_S14_B[0], select_S14_B[1], r_kpc)
    pt.plotModels(axs, select_L12_B[0], select_L12_B[1], r_kpc)
    pt.plotModels(axs, select_Sil11_B[0], select_Sil11_B[1], r_kpc, left=False)
    return


def plotMatParameterSpace2d(names_E11_A, mse_E11_A, title='A1 (E11)', model_z=2, second_min=False, row_pix=4):
    "Function to plot the parameter space"
    fig, ax = plt.subplots(1,1,figsize=(6,5))

    tmm_dec = np.load('../Data/pairs_z%.1f/t_mm_deciles.npy'%model_z)
    xoff_dec = np.load('../Data/pairs_z%.1f/xoff_deciles.npy'%model_z)
    
    tmm_dec = [i+(j-i)/2 for i, j in zip(tmm_dec[:-1], tmm_dec[1:])]
    xoff_dec =  [i+(j-i)/2 for i, j in zip(xoff_dec[:-1], xoff_dec[1:])]
    
    tmm_dec, xoff_dec = ['%.1f'%t for t in tmm_dec],  ['%.2f'%t for t in xoff_dec]

    mse_2d = mafl.makeMatrix2D(names_E11_A, mse_E11_A)

    df = pd.DataFrame(mse_2d,
                 index=tmm_dec,
                 columns=xoff_dec)

    ax = sns.heatmap(df, cbar_kws={'label': 'MSE values'}, cmap='ocean')
    
    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    row, col = np.where(mse_2d == np.min(mse_2d))[0][0], np.where(mse_2d == np.min(mse_2d))[1][0]
    ax.add_patch(Rectangle((col, row), 1, 1, edgecolor='#fc521e', fill=False, lw=3))

    if second_min:
        #second_try = np.min(mse_2d[mse_2d != np.min(mse_2d)])
        row2 = row_pix
        col2 = np.where(mse_2d == second_try)[1][0]
        print(row2, col2)
        
        ax.add_patch(Rectangle((col2, row2), 1, 1, edgecolor='#fc521e', ls='--', fill=False, lw=3))

    setLabel(ax, r'$\tilde{X}_{\rm off}$', r'$t_{\rm MM}$ [Gyr]', 'Model '+title, legend=False)
    fig.patch.set_facecolor('white')
    
    saveFig(title+'parameter_space.png')
    return ax