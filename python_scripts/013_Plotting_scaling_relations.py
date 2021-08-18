# astropy modules
import astropy.units as u
import astropy.io.fits as fits

from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# system imports
import os
import sys
import importlib as ib
import glob

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import norm
from scipy import interpolate
import pandas as pd

font = {'family' : 'serif',
        'weight' : 'medium',
        'size'   : 20}
matplotlib.rc('font', **font)

sys.path.append('../imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import Scaling_relations as sr
import plotting_sr_agn_clu as pt
import All_sky as sky

"""
Input parameters
"""
redshift_limit = 2

# plotting parameters
pal = sns.hls_palette(10, h=.5, s=1).as_hex()
sns.set_context("paper", font_scale=2, rc={"lines.linewidth":2})

# cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777
L_box = 1000.0 / h

vol = cosmo.comoving_volume(redshift_limit)*(53)/(129600/np.pi)

# loading files parameters
model_names_arr = np.array(['Model_A%d'%idx for idx in np.arange(4)])
frac_cp_agn_arr = [0.03, 0.10, 0.15, 0.2]

# radial bins for studying the changed flux
frac_r500c_arr = [0, .2, 0.5, 1]

# bin names of the flux that is upscaled
flux_up_R500c_arr = ['f_AGN_(%.1f-%.1f)R_500c'%(frac_r500c_arr[i], f) for i, f in enumerate(frac_r500c_arr[1:])]

# open file without AGN models
fname = '../Data/pairs_z%.1f/Scaling_relations/CLU_with_scaled_Lx_all_sky_%s.fit'%(redshift_limit, 'ModelNone')
CLU_og = Table.read(fname, format='fits')


"""
Plotting functions
"""
def getClusterFile(redshift_limit=2, model_name='Model_A0', frac_cp_agn=0.2):
    # opens model directory
    model_dir = '../Data/pairs_z%.1f/Scaling_relations/%s/'%(redshift_limit, model_name)

    # name of the model file for the chosen fraction of cp agn
    fname = model_dir+'CLU_with_scaled_Lx_all_sky_frac_cp_%.2f.fit'%(frac_cp_agn)
    hd_clu_params_all_model = Table.read(fname, format='fits')
    return hd_clu_params_all_model

def plotLabels(fig, ax, title, x=True, y=True):
    "Function for labelling all the Lx-M500c plots"
    fig.patch.set_facecolor('white')
    if x and y:
        pt.setLabel(ax, r'$\log_{10}\ M_{\rm 500c}\ [M_\odot]\times E(z)$',\
            r'$\log_{10}\ \rm L_{\rm X}/E(z)\ [erg/s] $ ',\
            title=title,\
            legend=False, xlim=[13.0, 15.])

    if x and not y:
        pt.setLabel(ax, r'$\log_{10}\ M_{\rm 500c}\ [M_\odot]\times E(z)$',\
            '', title=title, legend=False, xlim=[13.0, 15.])
    if not x and y:
        pt.setLabel(ax, '',\
            r'$\log_{10}\ \rm L_{\rm X}/E(z)\ [erg/s] $ ',\
             title=title, legend=False, xlim=[13.0, 15.])

    if not x and not y:
        pt.setLabel(ax, '', '', title=title, legend=False, xlim=[13.0, 15.])
    return 

def plotScatterLabels(fig, ax, title, xl=r'$\log_{10}\ M_{\rm 500c}\ [M_\odot]\times E(z)$',\
 yl=r'$\sigma ( \log_{10}\ \rm L_{\rm X}/E(z)\ )$ ', x=True, y=True):
    "Function for labelling all the Lx-M500c plots"
    fig.patch.set_facecolor('white')
    if x and y:
        pt.setLabel(ax, xl, yl, title=title, legend=False, xlim=[13.0, 15.])

    if x and not y:
        pt.setLabel(ax, xl, '', title=title, legend=False, xlim=[13.0, 15.])

    if not x and y:
        pt.setLabel(ax, '', yl, title=title, legend=False, xlim=[13.0, 15.])

    if not x and not y:
        pt.setLabel(ax, '', '', title=title, legend=False, xlim=[13.0, 15.])
    return 

def plotModelsGivenFracCpAndR500cWthAGN(ax, model_names_arr, f_cp, in_r500c):
    pal = ['#34ed9a', '#e35a20', '#591496', '#42030f']
    #pal = sns.hls_palette(10, h=.5, s=1).as_hex()

    for m, model_name in enumerate(model_names_arr):
        # open cluster file
        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = 'Model A%d'%m
        ax, _ = pt.plotBinnedM500cLx(ax, CLU_model[in_r500c], c=pal[m],\
                                     label=label, model_name=model_name, frac_cp=f_cp)
    ax.legend(loc='lower right', fontsize=14, frameon=False)
    return ax

def plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c):
    pal = ['#34ed9a', '#e35a20',  '#42030f', 'b']
    #pal = sns.hls_palette(10, h=.5, s=1).as_hex()

    for f, f_cp in enumerate(frac_cp_agn_arr):
        # open cluster file
        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = r'$f_{\rm cp}=$ %d %s'%((f_cp*100), '%')
        ax, _ = pt.plotBinnedM500cLx(ax, CLU_model[in_r500c], c=pal[f],\
                                     label=label, model_name=model_name, frac_cp=f_cp)
    ax.legend(loc='lower right', fontsize=14, frameon=False)
    return ax

def plotModelsScatter(ax, model_names_arr, hd_clu_params_all, f_cp, in_r500c):
    "Function to plot the scatter for different models, keeping other params fixed"
    
    papers = ['Chiu et al. 2021', 'Lovisari et al. 2020', 'Lovisari et al. 2015', 'Bulbul et al. 2019', 'Barnes et a. 2017', 'Vikhlinin et al. 2009','Mantz et al. 2016']
    scatters = [0.18, 0.2, 0.245, 0.25, 0.3, 0.396, 0.43]
    mass_lower_limits = [13, 14, 13, 14, 14, 14, 14]
    greys = sns.color_palette("autumn", len(papers)+1)

    for i,y in enumerate(scatters):
        ax.hlines(y, mass_lower_limits[i], 15.2, colors=greys[i], linestyles='-', alpha=0.6, label=papers[i], zorder=1, linewidths=3)

    pal = ['#34ed9a', '#e35a20',  '#42030f', 'b']
    #pal = sns.color_palette("cubehelix", 4)

    for m, model_name in enumerate(model_names_arr):
        # open cluster file

        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = 'Model A%d'%m
        ax, _ = pt.plotBinnedM500cLxScatter(ax, CLU_model[in_r500c],\
        hd_clu_params_all['CLUSTER_LX_soft_RF'], c=pal[m],\
        label=label, model_name=model_name, frac_cp=f_cp)
    return ax

def plotRelativeModelsScatter(ax, model_names_arr, hd_clu_params_all, f_cp, in_r500c):
    "Function to plot the RELATIVE scatter for different models, keeping other params fixed"
    
    pal = ['#34ed9a', '#e35a20',  '#42030f', 'b']
    #pal = sns.color_palette("cubehelix", 4)

    for m, model_name in enumerate(model_names_arr):
        # open cluster file
        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = 'Model A%d'%m
        scaled = sr.getBinsLxM500c(CLU_model[in_r500c], model_name=model_name, frac_cp=f_cp)
        log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = scaled

        og = sr.getBinsLxM500c(hd_clu_params_all['CLUSTER_LX_soft_RF'], model_name='ModelNone')
        log_M500c_bins_og, dlog_M500c_og, log_Lx_mean_og, log_Lx_std_og = og

        ax.plot(log_M500c_bins+dlog_M500c, np.abs(log_Lx_std-log_Lx_std_og), color=pal[m], lw=2,\
        label=label, ls='-', zorder=2)
        ax.plot(log_M500c_bins+dlog_M500c, np.abs(log_Lx_std_og-log_Lx_std_og), color='r', lw=2, ls='--', zorder=2)   
    
    return ax

def plotRelativeFcpMean(ax, frac_cp_agn_arr, hd_clu_params_all, model_name, in_r500c, a):
    "Function to plot the RELATIVE scatter for different models, keeping other params fixed"
    
    pal = ['#34ed9a', '#e35a20',  '#42030f', 'b']
    #pal = sns.color_palette("cubehelix", 4)

    for f, f_cp in enumerate(frac_cp_agn_arr):
        # open cluster file
        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = r'$f_{\rm cp}$= %d %s'%(f_cp*100, '%')
        scaled = sr.getBinsLxM500c(CLU_model[in_r500c], model_name=model_name, frac_cp=f_cp)
        log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = scaled

        og = sr.getBinsLxM500c(hd_clu_params_all['CLUSTER_LX_soft_RF'], model_name='ModelNone')
        log_M500c_bins_og, dlog_M500c_og, log_Lx_mean_og, log_Lx_std_og = og

        ax.plot(log_M500c_bins+dlog_M500c, log_Lx_mean-log_Lx_mean_og, color=pal[f], ls=a,\
        zorder=2)
        #ax.plot(log_M500c_bins+dlog_M500c, np.abs(log_Lx_std_og), color='k', lw=2, ls='-.', zorder=2)   
    xl = r'$\log_{10}\ M_{\rm 500c}\ [M_\odot] E(z)$'
    yl = r'$\log_{10}\ L_{\rm X, with\ AGN}\ - \log_{10}\ L_{\rm X,\ cluster\ only}$ '
    return ax, xl, yl

def plotFcpScatter(ax, frac_cp_agn_arr, hd_clu_params_all, model_name, in_r500c, a):
    "Function to plot the scatter for different models, keeping other params fixed"
    papers = ['Chiu et al. 2021', 'Lovisari et al. 2020', 'Lovisari et al. 2015', 'Bulbul et al. 2019', 'Barnes et a. 2017', 'Vikhlinin et al. 2009','Mantz et al. 2016']
    scatters = [0.18, 0.2, 0.245, 0.25, 0.3, 0.396, 0.43]
    mass_lower_limits = [13, 14, 13, 14, 14, 14, 14]
    greys = sns.color_palette("autumn", len(papers)+1)

    for i,y in enumerate(scatters):
        ax.hlines(y, mass_lower_limits[i], 15.2, colors=greys[i], linestyles='-', alpha=0.6, label=papers[i], zorder=1, linewidths=3)

    pal = ['#34ed9a', '#e35a20', '#591496', '#42030f']

    for f, f_cp in enumerate(frac_cp_agn_arr):
        # open cluster file
        CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=f_cp)

        # plot Lx-M500c for the model, given the radial bin and cp fraction
        label = r'$f_{\rm cp}$= %d %s'%(f_cp*100, '%')
        scaled = sr.getBinsLxM500c(CLU_model[in_r500c], model_name=model_name, frac_cp=f_cp)
        log_M500c_bins, dlog_M500c, log_Lx_mean, log_Lx_std = scaled

        og = sr.getBinsLxM500c(hd_clu_params_all['CLUSTER_LX_soft_RF'], model_name='ModelNone')
        log_M500c_bins_og, dlog_M500c_og, log_Lx_mean_og, log_Lx_std_og = og

        ax.plot(log_M500c_bins+dlog_M500c, np.abs(log_Lx_std), color=pal[f], alpha=a,\
        label=label, ls='-', zorder=2)
        ax.plot(log_M500c_bins+dlog_M500c, np.abs(log_Lx_std_og), color='k', lw=2, ls='--', zorder=2)   
        
    return ax


"""
### 1. Effect of radial bins

Start by seeing the impact as we go farther in radius from the cluster center
"""
radial_bins = False
if radial_bins:
    fig, ax = plt.subplots(figsize=(6, 5))
    model_name = 'Model_A0'
    frac_cp_agn = 0.2
    CLU_model = getClusterFile(model_name=model_name, frac_cp_agn=frac_cp_agn)

    # for different radial bils
    for i, n in enumerate(flux_up_R500c_arr):
        label = r'with AGN in %.1f$\times R_{500c}$'%(frac_r500c_arr[i+1])    
        ax, _ = pt.plotBinnedM500cLx(ax, CLU_model[n], c=pal[i],\
                                     label=label, model_name=model_name, frac_cp=frac_cp_agn)

    # cluster only contribution
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')
    ax.legend(loc='lower right', fontsize=14, frameon=False)

    # no AGN model contribution
    for i, n in enumerate(flux_up_R500c_arr):
        ax, _ = pt.plotBinnedM500cLx(ax, CLU_og[n], c=pal[i], ls='--', \
                                    model_name='ModelNone')
    title = r'$f_{\rm cp}$ = %d %s, %s'%(100*frac_cp_agn, '%', model_name)
    plotLabels(fig, ax, title=title)
    pt.saveFig('Radial_bins_f_cp_%.2f_all_sky_%s.png'%(frac_cp_agn, model_name))
    print('created '+ 'Radial_bins_f_cp_%.2f_all_sky_%s.png'%(frac_cp_agn, model_name))



"""
### 1. Effect of different models on scaling relations

Start by seeing the impact as we go farther in radius from the cluster center
"""
see_effect_sr_changed_models = False
if see_effect_sr_changed_models:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    r = 1
    in_r500c = flux_up_R500c_arr[r-1] 
    rbin_no = frac_r500c_arr[r]

    #-----------------------------------
    # f_cp = 20%, radial bin 0.5 x R500c
    ax = axs[0, 0]
    f_cp = frac_cp_agn_arr[3]
    ax = plotModelsGivenFracCpAndR500cWthAGN(ax, model_names_arr, f_cp, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotLabels(fig, ax, title=title, x=False)

    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = axs[1, 0]
    f_cp = frac_cp_agn_arr[2]
    ax = plotModelsGivenFracCpAndR500cWthAGN(ax, model_names_arr, f_cp, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotLabels(fig, ax, title=title)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = axs[0, 1]
    f_cp = frac_cp_agn_arr[1]
    ax = plotModelsGivenFracCpAndR500cWthAGN(ax, model_names_arr, f_cp, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotLabels(fig, ax, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 3%, radial bin 0.x x R500c
    ax = axs[1, 1]
    f_cp = frac_cp_agn_arr[0]
    ax = plotModelsGivenFracCpAndR500cWthAGN(ax, model_names_arr, f_cp, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotLabels(fig, ax, title=title, y=False)

    #-----------------------------------
    pt.saveFig('Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))
    print('created'+'Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))




"""
### 3. Effect of different models on scaling relations

Start by seeing the impact as we go farther in radius from the cluster center
"""
see_effect_scatter_changed_models = False
if see_effect_scatter_changed_models:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    r = 1
    in_r500c = flux_up_R500c_arr[r-1] 
    rbin_no = frac_r500c_arr[r]
    #-----------------------------------
    # f_cp = 20%, radial bin 0.x x R500c
    ax = axs[0, 0]
    f_cp = frac_cp_agn_arr[3]
    ax = plotModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, title=title, x=False)
    ax.legend(fontsize=14, frameon=False, ncol=2)
    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = axs[1, 0]
    f_cp = frac_cp_agn_arr[2]
    ax = plotModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, title=title)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = axs[0, 1]
    f_cp = frac_cp_agn_arr[1]
    ax = plotModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 3%, radial bin 0.5 x R500c
    ax = axs[1, 1]
    f_cp = frac_cp_agn_arr[0]
    ax = plotModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, title=title, y=False)

    #-----------------------------------
    pt.saveFig('Scatter_Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))
    print('created '+'Scatter_Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))


"""
### 4. Extreme case for seeing model differences in scatter
"""
extreme_case_models = False
if extreme_case_models:
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    # radial bin 0.2 x R500c
    r = 1
    in_r500c = flux_up_R500c_arr[r-1] 
    rbin_no = frac_r500c_arr[r]
    
    # f_cp = 20%
    f_cp = frac_cp_agn_arr[3]
    
    ax = axs
    ax = plotModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, title=title, x=False)
    ax.legend(bbox_to_anchor=(1, 0.05), loc='lower left', fontsize=14, frameon=False)
    
    pt.saveFig('Scatter_Radial_bin_%.1f_f_cp_%.2f_models_all.png'%(rbin_no, f_cp))
    print('created extreme case models')
    

"""
### 5. Relative Effect of scatter from different models on scaling relations

Start by seeing the impact as we go farther in radius from the cluster center
"""
see_rel_scatter_changed_models = False
if see_rel_scatter_changed_models:
    yl = r'Res$\{ \sigma ( \log_{10}\ \rm L_{\rm X}/E(z)\ )\}$ '
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    r = 1
    in_r500c = flux_up_R500c_arr[r-1] 
    rbin_no = frac_r500c_arr[r]
    #-----------------------------------
    # f_cp = 20%, radial bin 0.x x R500c
    ax = axs[0, 0]
    f_cp = frac_cp_agn_arr[3]
    ax = plotRelativeModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, yl=yl, title=title, x=False)
    ax.legend(fontsize=14, frameon=False, ncol=2)
    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = axs[1, 0]
    f_cp = frac_cp_agn_arr[2]
    ax = plotRelativeModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, yl=yl, title=title)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = axs[0, 1]
    f_cp = frac_cp_agn_arr[1]
    ax = plotRelativeModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, yl=yl, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 3%, radial bin 0.5 x R500c
    ax = axs[1, 1]
    f_cp = frac_cp_agn_arr[0]
    ax = plotRelativeModelsScatter(ax, model_names_arr, CLU_og, f_cp, in_r500c)
    
    title = r'$f_{\rm cp}$ = %d %s, <%.1f $\times R_{500c}$'%(f_cp*100, '%', rbin_no)
    plotScatterLabels(fig, ax, yl=yl, title=title, y=False)

    #-----------------------------------
    pt.saveFig('relative_scatter_Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))
    print('created '+'relative_scatter_Radial_bin_%.1f_f_cp_all_models_all.png'%(rbin_no))


"""
### 1. Effect of different fractions on scaling relations

Start by seeing the impact as we go farther in radius from the cluster center
"""
see_effect_sr_changed_models = False
if see_effect_sr_changed_models:
    gs = gridspec.GridSpec(2, 12)
    gs.update(wspace=1.5)
    fig, _ = plt.subplots(1, 1, figsize=(13.5, 10))
    model_name = model_names_arr[1]

    #-----------------------------------
    # f_cp = 20%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[0,:6])
    r = 1
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False)

    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[0,6:])
    r = 2
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[1,3:9])
    r = 3
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title)
    
    #-----------------------------------
    pt.saveFig('Radial_bin_all_f_cp_all_models_A0.png')
    print('created'+'Radial_bin_all_f_cp_all_models_A0')



"""
### 1. Effect of different fractions on scaling relations (with last plot addition)

Start by seeing the impact as we go farther in radius from the cluster center
"""
see_effect_sr_changed_models = True
if see_effect_sr_changed_models:
    gs = gridspec.GridSpec(2, 12)
    gs.update(wspace=1.5)
    fig, _ = plt.subplots(1, 1, figsize=(13.5, 10))
    model_name = model_names_arr[1]

    #-----------------------------------
    # f_cp = 20%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[0,:5])
    r = 1
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False)

    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[0,6:])
    r = 2
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[1,0:5])
    r = 3
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title)
    
    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = plt.subplot(gs[1,6:])
    r = 3
    for r, a in zip([1, 2, 3], ['-', '-.', ':']):
        in_r500c = flux_up_R500c_arr[r-1] 
        rbin_no = frac_r500c_arr[r]
        
        #-----------------------------------
        # f_cp = 20%, radial bin 0.x x R500c
        
        ax, xl, yl = plotRelativeFcpMean(ax, frac_cp_agn_arr, CLU_og, model_name, in_r500c, a)
        if r == 1:
            ax.legend(bbox_to_anchor=(1, 0.05), loc='lower left', fontsize=14, frameon=False)
    
    title = r'Model A1, <[0.2, 0.5, %.1f] $\times R_{500c}$'%(rbin_no)
    plotScatterLabels(fig, ax, xl=xl, yl=yl, title=title)
    
    #-----------------------------------
    pt.saveFig('Radial_bin_all_f_cp_all_models_A1.png')
    print('created'+'Radial_bin_all_f_cp_all_models_A1')

"""
### 6. Effect of scatter on changing fraction of close pair agn
"""

see_effect_scatter_changed_frac_cp = False
if see_effect_scatter_changed_frac_cp:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    model_name='Model_A1'

    r = 3
    for r, a in zip([1, 2, 3], [1, 0.4, 0.1]):
        in_r500c = flux_up_R500c_arr[r-1] 
        rbin_no = frac_r500c_arr[r]
        
        #-----------------------------------
        # f_cp = 20%, radial bin 0.x x R500c
        
        ax = plotFcpScatter(ax, frac_cp_agn_arr, CLU_og, model_name, in_r500c, a)
        if r == 1:
            ax.legend(bbox_to_anchor=(1, 0.05), loc='lower left', fontsize=14, frameon=False)
    title = r'Model A1, <[0.2, 0.5, %.1f] $\times R_{500c}$'%(rbin_no)
    plotScatterLabels(fig, ax, title=title)
    

    pt.saveFig('Scatter_Radial_bin_all_f_cp_all_%s.png'%model_name)
    print('created '+'Scatter_Radial_bin_all_f_cp_all_%s.png'%model_name)

"""
Plot for changed in fcp with sr and scatter

"""
see_effect_sr_scatter = False
if see_effect_sr_scatter:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    model_name = model_names_arr[1]

    #-----------------------------------
    # f_cp = 20%, radial bin 0.5 x R500c
    ax = axs[0,0]
    r = 1
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='--', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False)

    #-----------------------------------
    # f_cp = 15%, radial bin 0.5 x R500c
    ax = axs[0,1]
    r = 2
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title, x=False, y=False)

    #-----------------------------------
    # f_cp = 10%, radial bin 0.5 x R500c
    ax = axs[1,0]
    r = 3
    rbin_no = frac_r500c_arr[r]
    in_r500c = flux_up_R500c_arr[r-1]
    ax = plotFcpGivenModelAndR500cWthAGN(ax, frac_cp_agn_arr, model_name, in_r500c)
    ax, _ = pt.plotBinnedM500cLx(ax,  CLU_og['CLUSTER_LX_soft_RF'], ls='-.', \
                                model_name='ModelNone')

    title = r'with AGN in %.1f $\times R_{500c}$'%(rbin_no)
    plotLabels(fig, ax, title=title)
    #-----------------------------------
    ax = axs[1, 1]
    model_name='Model_A0'

    r = 3
    for r, a in zip([1, 2, 3], [1, 0.4, 0.1]):
        in_r500c = flux_up_R500c_arr[r-1] 
        rbin_no = frac_r500c_arr[r]
        
        #-----------------------------------
        # f_cp = 20%, radial bin 0.x x R500c
        
        ax = plotFcpScatter(ax, frac_cp_agn_arr, CLU_og, model_name, in_r500c, a)
        if r == 1:
            ax.legend(fontsize=12, loc='lower left', frameon=False, ncol=2)
    title = r'Model A1, <[0.2, 0.5, %.1f] $\times R_{500c}$'%(rbin_no)
    plotScatterLabels(fig, ax, title=title)
    
    #-----------------------------------
    pt.saveFig('Radial_bin_all_f_cp_all_models_A0_with_scatter.png')
    print('created'+'Radial_bin_all_f_cp_all_models_A0_with_scatter')

