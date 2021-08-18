# -*- coding: utf-8 -*-
"""Plotting.py for notebook 04_Modelling_AGN_fractions_from_literature

This python file contains all the functions used for modelling the AGN fraction based on measurements from literature

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 20th April 2021
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np
import seaborn as sns

# scipy modules
import scipy.odr as odr
from scipy import interpolate 

import os
import sys
import importlib

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('../imported_files/')
import Agn_incidence_from_Major_Mergers as aimm
import All_sky as sky
import Comparison_simulation_with_literature_data as cswl
import plotting_aim03 as pt
"""
Functions begin
"""

def getXY(arr):
    "Function to get x and y arr from the parent arr"
    x = [np.vstack(arr)[i][0] for i in range(len(arr))]
    y = [np.vstack(arr)[i][1] for i in range(len(arr))]
    return x, y

def getErrArrays(x):
    "Function to get the error arrays from the parent array"
    x_err_arr = [x[i][:2] for i in range(len(x))]
    y_err_arr = [x[i][2:] for i in range(len(x))]
    return x_err_arr, y_err_arr

def getErr(x_err_arr, x, y_err_arr, y):
    "Function to transform the errors arrays into readable pyplot formats"
    xerr = np.abs(np.transpose(np.vstack(x_err_arr)) - x)
    yerr = np.abs(np.transpose(np.vstack(y_err_arr)) - y)
    return xerr, yerr

def powerLaw(beta, x):
    return  -beta[0]/np.power(x, beta[1])

def performODR(X, Y, xerr_all, yerr_all, func=powerLaw):
    "Function to fit the empirical data"
    # model object
    power_law_model = odr.Model(func)

    # data and reverse data object
    data = odr.RealData(X, Y, sx=xerr_all, sy=yerr_all)
    
    # odr with model and data
    myodr = odr.ODR(data, power_law_model, beta0=[0.2, 0.])
    
    out = myodr.run()
    out = myodr.restart(iter=1000)
    return out

def getAGNlengths(redshift_limit=2, frac_cp_agn=0.03, all=True):
    agn_lens = np.load('../Data/all_sky_agn_lengths_z%.1f_fracCP_%.2f.npy'%(redshift_limit, frac_cp_agn), allow_pickle=True)
    rand_agn = np.array([agn_lens[i][0] for i in range(len(agn_lens))])
    if all:
        cp_agn = np.array([agn_lens[i][1] for i in range(len(agn_lens))])
    else:
        cp_agn = np.array([agn_lens[i][1] - rand_agn[i] for i in range(len(agn_lens))])

    print(rand_agn[-2:-1])
    return rand_agn, cp_agn

def loadPairsAllSky(r_p, end_points, data_dir, pixel_no_cont_arr, frac_cp_agn=0.03, cp_agn=True):
    n_pairs_separation = np.zeros((0, len(r_p)))
    
    for i, e in enumerate(end_points[:-1]):
        ll, ul = pixel_no_cont_arr[int(e)], pixel_no_cont_arr[int(end_points[i+1]-1)]
        
        if cp_agn:
            n_pairs = np.load(data_dir+'np_tmm2_xoff6_pixels_%s-%s_fracAGN%.2f.npy'%(ll, ul, frac_cp_agn), allow_pickle=True)
        else:
            n_pairs = np.load(data_dir+'np_pixels_%s-%s.npy'%(ll, ul), allow_pickle=True)

        n_pairs_separation = np.vstack((n_pairs_separation, n_pairs))
    return n_pairs_separation

def openAGNhaloPairsFile(file_divide=16, redshift_limit=0.2, num_rp_bins=12, last_px=750,\
 old_agn_pairs=True, frac_cp_agn=0.03):
    """
    @file_divide :: number of files into which the pair counting is divided for the all sky
    @redshift_limit :: lookback into redshift until
    @num_rp_bins :: number of r bins that was used when counting pairs
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, shell_volume = aimm.shellVolume()
    r_p_half, shell_volume_half = aimm.shellVolume(num_bins=num_rp_bins )

    # pixel number from the simulation file
    pixel_no_cont_arr = sky.allPixelNames()

    end_points = np.linspace(0, last_px, file_divide)
    end_points = np.append(end_points, [767], axis=None)

    halo_lens = np.load('../Data/all_sky_halo_lengths_z%.1f.npy'%redshift_limit)
    rand_agn, cp_agn = getAGNlengths(redshift_limit=redshift_limit, frac_cp_agn=frac_cp_agn, all=False)
    
    data_cp_dir = '../Data/pairs_z%.1f/cat_AGN_halo_pairs/'%redshift_limit
    
    # get the total number of possible AGN-halo pairs
    lens_rand, lens_cp = np.array([halo_lens, rand_agn]), np.array([halo_lens, cp_agn])
    tot_p_rand_agn = cswl.GammaDenominator(lens_rand, 0, -1, num_sets=2)
    tot_p_cp_agn = cswl.GammaDenominator(lens_cp, 0, -1, num_sets=2)
        # repeat same process for old catAGN (without close pairs)
    if old_agn_pairs:
        r_p_half, shell_volume_half = aimm.shellVolume(num_bins=num_rp_bins )
        data_dir = data_cp_dir + 'cat_without_CP/'
        n_p_sep = loadPairsAllSky(r_p_half, end_points, data_dir, pixel_no_cont_arr, cp_agn=False)
        
        print(tot_p_rand_agn.shape)
        frac_rand_agn_mean = [np.mean(n_p_sep[:, i]/tot_p_rand_agn) for i in range(len(r_p_half))]
        frac_rand_agn_std = [np.std(n_p_sep[:, i]/tot_p_rand_agn) for i in range(len(r_p_half))]
        
        mean_gamma_rand, std_gamma_rand = frac_rand_agn_mean[1:]/shell_volume_half, frac_rand_agn_std[1:]/shell_volume_half

    n_pairs_sep = loadPairsAllSky(r_p, end_points, data_cp_dir, pixel_no_cont_arr, frac_cp_agn=frac_cp_agn)
    
    # get mean and std of the number density
    frac_cp_agn_mean = [np.mean(n_pairs_sep[:, i]/tot_p_cp_agn)  for i in range(len(r_p))]
    frac_cp_agn_std  = [np.std(n_pairs_sep[:, i]/tot_p_cp_agn)  for i in range(len(r_p))]
    
    mean_gamma_cp, std_gamma_cp = frac_cp_agn_mean[1:]/shell_volume, frac_cp_agn_std[1:]/shell_volume
    
    names_cp, names_rand = ['Gamma_mean_CP', 'Gamma_std_CP'], ['Gamma_mean_RAND', 'Gamma_std_RAND']
    return Table([mean_gamma_cp, std_gamma_cp], names=names_cp), Table([mean_gamma_rand, std_gamma_rand], names=names_rand)

def getFracAgnHaloPairsCp(ax, frac_cp_agn_arr, z=1, num_rp_bins=12):
    """
    Function to get the fraction of pairs  
    """
    # get shell volume and projected radius bins [Mpc]
    r_p, _ = aimm.shellVolume()
    r_p_half, _ = aimm.shellVolume(num_bins=num_rp_bins )
    pal = sns.color_palette("viridis", 4).as_hex()

    data_dir = '../Data/pairs_z%.1f/Major_dv_pairs/'%1
    gamma_all = np.load(data_dir+'gamma_all_pixels.npy', allow_pickle=True)

    f_cp_agn_halo_pairs, f_rand_agn_halo_pairs = np.zeros((0, len(r_p)-1)), np.zeros((0, len(r_p_half)-1))

    for f, frac_cp_agn in enumerate(frac_cp_agn_arr):
        # read the files that count pairs (AGN-halo and halo-halo) for the new catAGN and old catAGN
        g_cp, g_rand = openAGNhaloPairsFile(redshift_limit=z, frac_cp_agn=frac_cp_agn, num_rp_bins=num_rp_bins)
        
        # plot the results and the chanes wrt the old catAGN
        gamma_all_inter = pt.plotChangesCatAGN(ax, g_cp, g_rand, label_idx = f, num_rp_bins=num_rp_bins, \
                                                                  redshift_limit=z, c=pal[f], frac_cp_agn=frac_cp_agn)

        # append these values 
        cols_cp0 = Column(data=gamma_all[0], name='Gamma_meanALL')
        cols_cp1 = Column(data=gamma_all[1], name='Gamma_stdALL'  )
        
        cols_rand0 = Column(data=gamma_all_inter[0], name='Gamma_meanALL'  )
        cols_rand1 = Column(data=gamma_all_inter[1], name='Gamma_stdALL'  )
        
        g_cp.add_columns([cols_cp0, cols_cp1])
        g_rand.add_columns([cols_rand0,cols_rand1])
    return g_cp, g_rand

def combineFracTmmXoff(t0, std_t0, frac_xoff_z2, num_decs=20):
    models_all = np.zeros((0, len(t0) ))
    std_all = np.zeros((0, len(t0) ))

    for d in np.arange(0, num_decs, 2):
        x, std_x = frac_xoff_z2.columns[d], frac_xoff_z2.columns[d+1]
    
        models_all = np.append(models_all, [t0+x], axis=0)
        std = np.sqrt(std_x**2 + std_t0**2)
        std_all = np.append(std_all, [std], axis=0)
    return models_all, std_all

def generateDecileModels(frac_tmm, frac_xoff, num_decs=20):
    models = np.zeros((0, len(frac_tmm)))
    std = np.zeros((0, len(frac_tmm)))
    for d in np.arange(0, num_decs, 2):
        # best possible models for z<2
        tmm, std_tmm = frac_tmm.columns[d], frac_tmm.columns[d+1]
        
        models_tmm, std_tmm = combineFracTmmXoff(tmm, std_tmm, frac_xoff, num_decs=num_decs)
        
        models = np.append(models, models_tmm, axis=0)
        std =  np.append(std, std_tmm, axis=0)
    return models, std

def interpolateModelWRTdata(model, std, data_x, r_kpc):
    "Function to get the value of the model at the data x-points"
    func_model = interpolate.interp1d(r_kpc, model)
    func_std = interpolate.interp1d(r_kpc, std)
    return func_model(data_x), func_std(data_x)

def MSE(data, model):
    "Function calculated the goodness of the model"
    if len(data)>len(model):
        start_at = len(data)-len(model) 
        data = data[start_at:]
    mse = np.sum( (data-model)**2)/len(data)
    return mse

def normalizeAsymptote(array, asymotote_value=0.01):
    "Function to normalize the array according to its asymtotic value"
    if array[-1] != asymotote_value:
        diff = array[-1] - asymotote_value
        if diff > 0:
            array_normalized = array - diff
        elif diff < 0:
            array_normalized = array + np.abs(diff)
    else:
        array_normalized = array
    return np.array(array_normalized)

def getBestLookingModels(models, std, data, r_kpc, num_decs=10, asymotote_value=[.01, .05], left=True):
    "Function to get the best looking models given a set of 100 models"
    colnames = ['t%d_x%d'%(t, x) for t in np.arange(num_decs) for x in np.arange(num_decs)]
    selected_model_names, selected_models = [], np.zeros((0, models.shape[1])) 
    selected_std = np.zeros((0, models.shape[1]))
    
    discarded_models, mse_arr = [], []
    for i in range(models.shape[0]):
        # know with respect to which control one must normalize
        if left:
            asymotote_val = asymotote_value[0]
            m_id, s = normalizeAsymptote(models[i], asymotote_value=asymotote_val), std[i]
        else:
            asymotote_val = asymotote_value[1]
            m_id, s = normalizeAsymptote(models[i], asymotote_value=asymotote_val), std[i]
        
        model_is_physical = (m_id[1:] > asymotote_val)

        if  np.all(model_is_physical):
            # interpolate the model
            x, y, yerr = data['r_p'], data['f_agn'], data['f_agn_err']
            m_inter, s_inter = interpolateModelWRTdata(m_id, s, x[(x>r_kpc[1]) & (x<r_kpc[-1])], r_kpc)
            
            # save chi-sqs
            mse = MSE(y, m_inter) 
            mse_arr.append(mse)

            selected_model_names.append(colnames[i])
            selected_models = np.append(selected_models, [m_id], axis=0)
            selected_std = np.append(selected_std, [s], axis=0)
        else:
            # discard model
            discarded_models.append(colnames[i])
        
    selected = np.array([selected_models, selected_std], dtype=object)
    print('Best model out of %d'%len(mse_arr), selected_model_names[np.where(mse_arr==np.min(mse_arr))[0][0]])
    return mse_arr, np.array([discarded_models, selected_model_names], dtype=object), selected

def makeMatrix2D(names_E11_A, mse_E11_A):
    mse_A_mat2d = np.zeros((10, 10))

    for i in np.arange(10):
        for j in np.arange(10):
            selected_names = np.array(names_E11_A[1])
            if np.any(selected_names == 't%d_x%d'%(i, j)):
                match_idx = np.where(selected_names == 't%d_x%d'%(i, j))
                mse_A_mat2d[i, j] = mse_E11_A[match_idx[0][0]] 
            else:
                mse_A_mat2d[i, j] = np.max(mse_E11_A)
    return mse_A_mat2d