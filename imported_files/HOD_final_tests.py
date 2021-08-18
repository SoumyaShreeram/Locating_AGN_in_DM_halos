"""
File for studying HOD of different AGN catalogues 
"""
# scipy modules
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np
from scipy.stats import gaussian_kde
import os
import glob

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# personal imports
import Agn_incidence_from_Major_Mergers as aimm
import plotting_cswl05 as pt
import All_sky as sky
import Scaling_relations as sr 


def getModelDir(frac_cp=0.2, pixel_no='000000'):
    "Function to get the directory names"
    if frac_cp == 0.2 or frac_cp == 0.1:
       string = '%.1f'%frac_cp
    else:
        string = '%.2f'%frac_cp

    data_dir = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS',\
         'fixedAmp_InvPhase_001')

    list_model_names = np.array(glob.glob(os.path.join(data_dir, 'CP_10_sigma_1.0_frac_'+string+'_*_tmm_*_xoff_*')))
    return list_model_names


def sumHist(arr, cut_columns):
    "Function sums over the redshifts in the array"
    arr_new = arr[:, cut_columns]
    arr_summed = np.sum(arr_new, axis=1)
    return np.mean(arr_summed, axis=1), np.std(arr_summed, axis=1)

def concatHISTallPIX(model_dir, cut_columns, num_m_bins=40, num_z_bins=21, catGal=False):
    "Function to concatenate HIST files for all pixels"
    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()

    cen_hist = np.zeros((len(pixel_no_cont_arr), num_m_bins, num_z_bins))
    hist_all =  np.zeros((len(pixel_no_cont_arr), num_m_bins, num_z_bins))
    
    for p, pixel_no in enumerate(pixel_no_cont_arr):    
        if not catGal:
            file_cen_hist = os.path.join(model_dir, 'HIST', 'HistCen_Mvir_zz_'+pixel_no+'.ascii')
            hd_cenMvir_model = np.loadtxt(file_cen_hist)
            hd_cenMvir_model = np.sum(hd_cenMvir_model[:, cut_columns], axis=1)
            cen_hist[p] = hd_cenMvir_model
            
        file_hist_all = os.path.join(model_dir, 'HIST', 'Hist_Mvir_zz_'+pixel_no+'.ascii')   
        hd_Mvir_model = np.loadtxt(file_hist_all)
        hd_Mvir_model = np.sum(hd_Mvir_model[:, cut_columns], axis=1)
        hist_all[p] = hd_Mvir_model

    if catGal:
        return hist_all
    else:
        return cen_hist, hist_all

def concatHISTallPIXLminZmax(model_dir, cut_columns, Lmin, Zmax, num_m_bins=40, num_z_bins=21, catGal=False):
    "Function to concatenate HIST files for all pixels"
    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()

    cen_hist = np.zeros((len(pixel_no_cont_arr), num_m_bins, num_z_bins))
    hist_all =  np.zeros((len(pixel_no_cont_arr), num_m_bins, num_z_bins))
    
    for p, pixel_no in enumerate(pixel_no_cont_arr):    
        filename = pixel_no+'_LX_%d_Z_%.1f'%(Lmin, Zmax)

           
        file_cen_hist = os.path.join(model_dir, 'HIST', 'HistCen_Mvir_zz_'+filename+'.ascii')
        hd_cenMvir_model = np.loadtxt(file_cen_hist)
        hd_cenMvir_model = np.sum(hd_cenMvir_model[:, cut_columns], axis=1)
        cen_hist[p] = hd_cenMvir_model
        
        file_hist_all = os.path.join(model_dir, 'HIST', 'Hist_Mvir_zz_'+filename+'.ascii')   
        hd_Mvir_model = np.loadtxt(file_hist_all)
        hd_Mvir_model = np.sum(hd_Mvir_model[:, cut_columns], axis=1)
        hist_all[p] = hd_Mvir_model

    if catGal:
        return hist_all
    else:
        return cen_hist, hist_all

def concatAllPixels(cut_columns, list_model_names, save_dir, model_names_arr, catGal=False):
    """Function to concat the histograms for all the pixels
    @cut_columns :: get rid of all z bins above redshift limit
    @list_model_names :: list of the directories with the different instances of the CP catAGN
    @save_dir :: local dir that saves the combined histrogram
    """
    
    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()

    for model_dir, model_name in zip(list_model_names, model_names_arr):
        cen_hist_all, hist_all = concatHISTallPIX(model_dir, cut_columns, catGal=catGal)
        np.save(save_dir+'cenHIST_'+ model_name +'.npy', cen_hist_all, allow_pickle=True)
        np.save(save_dir+'HIST_'+ model_name +'.npy', hist_all, allow_pickle=True)
    return 

def concatAllPixelsLminZmax(cut_columns, list_model_names, save_dir, model_names_arr,\
 Lmin, Zmax, catGal=False):
    """Function to concat the histograms for all the pixels
    @cut_columns :: get rid of all z bins above redshift limit
    @list_model_names :: list of the directories with the different instances of the CP catAGN
    @save_dir :: local dir that saves the combined histrogram
    """
    
    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()
    filename = '_LX_%d_Z_%.1f'%(Lmin, Zmax)
    for model_dir, model_name in zip(list_model_names, model_names_arr):
        cen_hist_all, hist_all = concatHISTallPIXLminZmax(model_dir, cut_columns, Lmin, Zmax, catGal=catGal)
        np.save(save_dir+'cenHIST_'+ model_name+ filename +'.npy', cen_hist_all, allow_pickle=True)
        np.save(save_dir+'HIST_'+ model_name+ filename +'.npy', hist_all, allow_pickle=True)
    return

def getHistDir(model_name = 'Model_A0', frac_cp=0.2, pixel_no='000000'):
    "Function to get the histograms of the AGN"

    list_model_names = getModelDir(frac_cp=frac_cp, pixel_no=pixel_no)

    cen_histAGN_cpAGN, histAGN_cpAGN = [], []
    for model_dir in list_model_names:
        file_cen_histAGN = os.path.join(model_dir, 'HIST', 'HistCen_Mvir_zz_'+pixel_no+'.ascii')
        file_histAGN = os.path.join(model_dir, 'HIST', 'Hist_Mvir_zz_'+pixel_no+'.ascii')   

        hd_cenMvir_model = np.loadtxt(file_cen_histAGN)
        hd_Mvir_model = np.loadtxt(file_histAGN)
        
        cen_histAGN_cpAGN.append(hd_cenMvir_model)
        histAGN_cpAGN.append(hd_Mvir_model)
    return cen_histAGN_cpAGN, histAGN_cpAGN 

def loadCPcatAGN(frac_cp=0.2, pixel_no='000000'):
    "Function to load the different CP cat AGN"
    
    list_model_names = getModelDir(frac_cp=frac_cp, pixel_no=pixel_no)

    hd_cp_agn_all = []
    for model_dir in list_model_names:
        catAGN_filename = os.path.join(model_dir, pixel_no+'.fit')
        hd_cp_agn = Table.read(catAGN_filename, format='fits')
        hd_cp_agn_all.append(hd_cp_agn)

    return hd_cp_agn_all

def plotForAllZbins(ax, m_mid, cenN, color, ls='-', label=''):
    # plot the lines for every z-bin
    ax.plot(10**m_mid, cenN, ls=ls, color=color, lw=1.5, label=label)
    return ax

def plotModelsFixedFrac(ax, model_names_arr, histAGN_ALL, cen_histAGN_ALL, cen_histGAL,\
 m_mid, zz=3, frac_cp=0.2 ):
    color = sns.color_palette("bright", histAGN_ALL.shape[0]+1)

    for m in range(len(model_names_arr)):
        label = 'Model A%d'%m 
        hod_fixed_fraction = histAGN_ALL[m, frac_cp, :, zz]/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, hod_fixed_fraction,\
                             color[m], label=label)

        cen_fixed_fraction = cen_histAGN_ALL[m, frac_cp, :, zz]/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, cen_fixed_fraction,\
                             color[m], ls='--')

        sat_fixed_fraction = (histAGN_ALL[m, frac_cp, :, zz]-cen_histAGN_ALL[m, frac_cp, :, zz])/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, sat_fixed_fraction,\
                             color[m], ls='-.')
    return ax


def plotFracFixedModels(ax, frac_cp_arr, histAGN_ALL, cen_histAGN_ALL, cen_histGAL,\
 m_mid, zz=3, model_no=0):
    color = sns.color_palette("bright", histAGN_ALL.shape[0]+1)

    for f in range(len(frac_cp_arr)):
        
        label = r'$f_{\rm cp}=$ %.2f'%frac_cp_arr[f]
        hod_fixed_model = histAGN_ALL[model_no, f, :, zz]/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, hod_fixed_model,\
                             color[f], label=label)

        cen_hod_fixed_model = cen_histAGN_ALL[model_no, f, :, zz]/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, cen_hod_fixed_model,\
                             color[f], ls='--')

        sat_hod_fixed_model = (histAGN_ALL[model_no, f, :, zz]-cen_histAGN_ALL[model_no, f, :, zz])/cen_histGAL[:, zz]
        ax = plotForAllZbins(ax, m_mid, sat_hod_fixed_model,\
                             color[f], ls='-.')
    return ax

def plotOgHOD(ax, m_mid, histAGN, cen_histAGN, cen_histGAL, zz=3, label=r'$f_{\rm cp}=$ 0.00'):
    ax.plot(10**m_mid, histAGN[:, zz]/cen_histGAL[:, zz], 'k-', alpha=0.5,lw=1.5, label=label)
    ax.plot(10**m_mid, cen_histAGN[:, zz]/cen_histGAL[:, zz], 'k--', alpha=0.5,lw=1.5)
    ax.plot(10**m_mid, (histAGN[:, zz]-cen_histAGN[:, zz])/cen_histGAL[:, zz], 'k-.', alpha=0.5,lw=1.5)
    return ax