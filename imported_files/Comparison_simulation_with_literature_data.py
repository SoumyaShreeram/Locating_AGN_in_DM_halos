"""
02. Creating a Major Merger (MM) catalogue to study AGN incidence due to galaxy mergers

This python file contains the function of the corresponding notebook '02_AGN_incidence_from_Major_Mergers'.

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 30th March 2021
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
import Agn_incidence_from_Major_Mergers as aimm


def countSelectedPairs(all_selected_idx, print_msg = True, string = 'Major merger cut: '):
    """
    Function to count selected pairs from the list of lists outputted by ball tree
    @all_selected_idx :: 
    """
    count_selected_pairs = 0
    
    for j, mm in enumerate(all_selected_idx):
        
        if len(mm) >= 1:
            for m in mm:
                if j != m:
                    count_selected_pairs += 1
    if print_msg:
        print(string+'%d selected pairs'%(count_selected_pairs/2))
    return count_selected_pairs

def deltaVelSelection(hd_halo, all_mm_idx, dz_cut=0.001):
    """
    Function to choose the pairs that meet the delta v criterion
    @mm :: are you passing the index array after selecting major mergers?
    """
    all_dz_idx = []
    for i, mm  in enumerate(all_mm_idx):
        dz_idx = []
        
        # going through all the pairs to see their delta v criteria
        if len(mm) >= 1:
            for m in mm:
                if i != m:
                    dz_r = np.abs(hd_halo[i]['redshift_R'] - hd_halo[m]['redshift_R'])
                    dz_s = np.abs(hd_halo[i]['redshift_S'] - hd_halo[m]['redshift_S'])

                    # applying the selection criteria
                    if dz_r < dz_cut and dz_s < dz_cut:
                        dz_idx.append(m)
        all_dz_idx.append(dz_idx)
    count_dz_major_mergers = countSelectedPairs(all_dz_idx, string = 'Delta z %d cut: '%dz_cut)
    return all_dz_idx, count_dz_major_mergers


def majorMergerSelection(hd_halo, pairs_idx, mass_min = 0.33, mass_max = 3, keyword='mm and dv'): 
    """
    Function to choose the pairs that classify as major mergers
    @hd_halo :: header file with all the DM halos
    @pairs_idx :: list of lists containing the idx of the pairs for every halo
    """
    # object arr to save major pairs for every DM halo
    all_mm_idx = []
    
    for i, p in enumerate(pairs_idx): 
        # list to save indicies of pairs classified as major mergers
        mm_idx = []
       
        if len(p) >= 1:
            for p_idx in p:
                if i != p_idx:
                    mass_ratio = hd_halo[i]['galaxy_SMHMR_mass']/hd_halo[p_idx]['galaxy_SMHMR_mass']

                    # only consider pairs that pass mass ratios criterion
                    if float(mass_ratio) >= mass_min and float(mass_ratio) <= mass_max:
                        mm_idx.append(p_idx)

        # save this info for the given halo in the object array
        all_mm_idx.append(mm_idx)
    
    count_major_mergers = countSelectedPairs(all_mm_idx, string = 'Major merger %d : 1 cut '%mass_max)
    return all_mm_idx, count_major_mergers

def indexArray(all_mm_idx):
    idx_arr = []
    for i in range(len(all_mm_idx)):
        if len(all_mm_idx[i]) != 0:
            idx_arr.append(i)
    return idx_arr

def openPairsFiles(key, data_dir = 'Data/pairs_z2.0/', redshift_limit = 2, mass_max = 3, dz_cut = 0.001):
    """
    Function to open all the files with 
    """
    # get shell volume and projected radius bins
    r_p, shell_volume = aimm.shellVolume()
    pairs_idx_all, n_pairs_arr = [], []
    
    for i in range(len(r_p)):
        if key == 'mm':
            filename = 'Major_pairs/pairs_idx_r%.3f_mm%d.npy'%(r_p[i], mass_max)
        if key == 'mm and dv':
            filename = 'Major_dv_pairs/pairs_idx_r%.3f_mm%d_dz%.3f.npy'%(r_p[i], mass_max, dz_cut)
        if key == 'dv':
            filename = 'dv_pairs/pairs_idx_r%0.3f_dz%.3f.npy'%(r_p[i], dz_cut)
        if key == 'all':
            filename = 'pairs_idx_r%.3f.npy'%(r_p[i])
            
        pairs_idx = np.load(data_dir+filename, allow_pickle=True)
        n_pairs = countSelectedPairs(pairs_idx, print_msg = False)
        
        n_pairs_arr.append(n_pairs)
        pairs_idx_all.append(pairs_idx)        
    return np.array([pairs_idx_all, n_pairs_arr], dtype=object)

def saveTmmFiles(key, dt, arr , redshift_limit = 2, param='t_mm'):
    """
    Function decides where to save the tmm evaluated files
    """
    if key == 'mm and dv':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/all_pairs_%s%.2f-%.2f.npy'%(redshift_limit, param, dt[0], dt[1]), arr, allow_pickle=True)
    
    if key == 'mm':
        np.save('Data/pairs_z%.1f/Major_pairs/all_pairs_%s%.2f-%.2f.npy'%(redshift_limit, param, dt[0], dt[1]), arr, allow_pickle=True)
        
    if key == 'dv':
        np.save('Data/pairs_z%.1f/dv_pairs/all_pairs_%s%.2f-%.2f.npy'%(redshift_limit, param,  dt[0], dt[1]), arr, allow_pickle=True)

    # if you want to save all the pairs
    if key == 'all':
        np.save('Data/pairs_z%.1f/all_pairs_%s%.2f-%.2f.npy'%(redshift_limit, param, dt[0], dt[1]), arr, allow_pickle=True)
    return


def tmmToScale(cosmo, dt_m_arr):
    "Function to convert time time since MM array to scale factor of last MM"
    scale_mm = []
    for t in dt_m_arr:
        # converting the time since merger into scale factor
        merger_z = z_at_value(cosmo.lookback_time, t*u.Gyr)
        merger_scale = 1/(1 + merger_z)
        scale_mm.append(merger_scale)
    return scale_mm

def calTmm(cosmo, a, z):
    # convert the merger scale factor into redshift
    merger_z = z_at_value(cosmo.scale_factor, a)
    
    # convert the merger & current redshifts into lookback time
    merger_time = cosmo.lookback_time(merger_z)
    current_time = cosmo.lookback_time(z)
    
    # difference in lookback time between the merger and AGN redshift
    diff_time = merger_time-current_time
    return diff_time

def defineTimeSinceMergeCut(hd_obj, pairs_idx, cosmo, time_since_merger = 1):
    """
    Define the time since merger cut 
    """
    # object arr to save major pairs for every DM halo
    all_t_mm_idx = []
    
    for i, p in enumerate(pairs_idx): 
        # list to save indicies of pairs classified as major mergers
        t_mm_idx = []
        diff_time = calTmm(cosmo, hd_obj[i]['HALO_scale_of_last_MM'], hd_obj[i]['redshift_R'])
        
        # if there are more than 'self'-pairs
        if len(p) >= 1:
            for p_idx in p:                
                diff_time_pair = calTmm(cosmo, hd_obj[p_idx]['HALO_scale_of_last_MM'], hd_obj[p_idx]['redshift_R'])
                # only consider pairs that pass this time since merger-scale criterion
                if (diff_time_pair <= time_since_merger*u.Gyr) or  (diff_time <= time_since_merger*u.Gyr):
                    t_mm_idx.append(p_idx)
                    
        # save this info for the given halo in the object array
        all_t_mm_idx.append(t_mm_idx)
    
    count_t_mm = countSelectedPairs(all_t_mm_idx, print_msg = False, string = 'T_mm = %d Gyr: '%time_since_merger)
    return all_t_mm_idx, count_t_mm

def decideBins(a, a_max):
    "Function to decide the upper and lower limits of the bins"
    diff_a = [a[0:2], a[1:3], a[2:4], a[3:5], a[4:], [a[5], a_max]]
    
    return diff_a


def selectParameterPairs(hd_obj, pairs_idx, cosmo, diff_t_mm_arr, param, redshift_limit = 2, string_param = 't_mm'):
    """
    Select pairs that pass the parameter cuts
    @param :: list with [lower_limit, upper_limit] of the parameter bin
    """
    # object arr to save major pairs for every DM halo
    all_t_mm_idx = []
    
    
    for i, p in enumerate(pairs_idx): 
        # list to save indicies of pairs classified as major mergers
        t_mm_idx = []
        diff_time = diff_t_mm_arr[i]
        
        # if there are more than 'self'-pairs
        if len(p) >= 1:
            for p_idx in p:
                if i != p_idx:
                    diff_time_pair = diff_t_mm_arr[p_idx]
                    
                    # only consider pairs that pass this time since merger-scale criterion
                    if string_param == 't_mm':
                        if (param[0] <= diff_time_pair <= param[1]) or  ( param[0] <= diff_time <= param[1]):
                            t_mm_idx.append(p_idx)
                    if string_param == 'x_off':
                        if (param[0] >= diff_time_pair >= param[1]) or  (param[0] >= diff_time >= param[1]):
                            t_mm_idx.append(p_idx)

        # save this info for the given halo in the object array
        all_t_mm_idx.append(t_mm_idx)
        
    count_t_mm = countSelectedPairs(all_t_mm_idx, print_msg = False, string = '%s = %.1f - %.1f Gyr: '%(string_param, param[0], param[1]))
    return all_t_mm_idx, count_t_mm


def concatAllTmmFiles(dt_m_arr, key, redshift_limit=2, param='t_mm'):
    """
    Function to concatenate all the files containing pairs for different T_mm criteria
    """
    r_p, _ = aimm.shellVolume()
    n_pairs_t_mm_all = np.zeros( (0, len(r_p) ) )
    
    if key == 'mm and dv':
        data_dir = 'Data/pairs_z%.1f/Major_dv_pairs/'%redshift_limit
    if key == 'dv':
        data_dir = 'Data/pairs_z%.1f/dv_pairs/'%redshift_limit
    if key == 'mm':
        data_dir = 'Data/pairs_z%.1f/Major_pairs/'%redshift_limit
    if key == 'all':
        data_dir = 'Data/pairs_z%.1f/'%redshift_limit    
        
    for dt_m in dt_m_arr:
        if param == 't_mm':
            n_pairs_t_mm = np.load(data_dir+'all_pairs_%s%.1f.npy'%(param, dt_m), allow_pickle=True)
        if param == 'x_off':
            n_pairs_t_mm = np.load(data_dir+'all_pairs_%s%.2f.npy'%(param, dt_m), allow_pickle=True)
            
        # save the counts for all radius bins for a given time since merger
        n_pairs_t_mm_all = np.append(n_pairs_t_mm_all, [n_pairs_t_mm], axis=0)
    return n_pairs_t_mm_all

def error(n_pairs):
    "Calculates the error on the pairs"    
    err = []
    for n in n_pairs:
        if n != 0:
            err.append(1/np.sqrt(n))
        else:
            err.append(0)
    return err

def nPairsToFracPairs(hd_obj, all_pairs_vs_rp, redshift_limit = 2):
    """
    Function to convert the number of pairs into a fractional number density per shell
    @redshift_limit :: the initial redshift limit set on the sample (needed for opening dir)
    """
    num_pairs = all_pairs_vs_rp[1:] - all_pairs_vs_rp[:-1]
    
    # get shell volume and projected radius bins
    r_p, shell_volume = aimm.shellVolume()
    
    # normalization
    total_num_pairs = len(hd_obj)    
    N = total_num_pairs*(total_num_pairs - 1)
    
    # fractional number density
    f_pairs = num_pairs/(N*shell_volume)
    return f_pairs, error(num_pairs)/(N*shell_volume)

def getAllMMscales(hd_obj, pairs_mm_all, r_p):
    "Function to get the scale of last MM of all the pairs for all radius"
    halo_m_scale_arr_all_r = []
    for i in range(len(r_p)):
        halo_m_scale_arr = []

        for i, p in enumerate(pairs_mm_all[0][i]): 
            # list to save indicies of pairs classified as major mergers
            t_mm_idx = []
            halo_merger_scale0 = hd_obj[i]['HALO_scale_of_last_MM']
            halo_m_scale_arr.append(halo_merger_scale0)
            # get scale factor of the companion
            if len(p) > 1:
                for p_idx in p:
                    halo_merger_scale1 = hd_obj[p_idx]['HALO_scale_of_last_MM']
                    halo_m_scale_arr.append(halo_merger_scale1)

        # save the info
        halo_m_scale_arr_all_r.append(halo_m_scale_arr)
    return halo_m_scale_arr_all_r

def getPairIndicies(pairs_idx, r):
    "Function generates an array that holds all the pair indicies"
    pairs_arr = np.zeros((0, 2))
    two_idx = []
    
    # get the indicies of the pair
    for j in range(pairs_idx[r].shape[0]):
        if len(pairs_idx[r][j]) >= 1:
            for p in pairs_idx[r][j]:
                # don't want a pair with itself
                if j != p: 
                    two_idx = sorted([j, p])
                    pairs_arr = np.append(pairs_arr, [two_idx], axis=0)
        
    pairs_arr = np.unique(pairs_arr, axis=0)
    return pairs_arr

def massRatios(pairs, m_arr):
    """
    Function calculates the mass ratio between the pairs
    """
    m1, m2 = m_arr[int(pairs[0])], m_arr[int(pairs[1])]
    return m1/m2

def meanZ(pairs, z_arr):
    """
    Function calculates the mean z between the pairs
    """
    z1, z2 = z_arr[int(pairs[0])], z_arr[int(pairs[1])]
    return (z1+z2)/2

def getMZmatchedPairs(hd_halo, pairs_all, pairs_selected, r, mr_min = 0.15, mr_max = 2, redshift_limit=2, step_z = 0.01, param = 't_mm'):
    """
    Function matches the mass of pairs 'with selection cuts' with those 'with no selections cuts'
    """
    m_arr, z_arr = hd_halo['galaxy_SMHMR_mass'], hd_halo['redshift_R']
    
    if param == 't_mm':
        param_arr =  np.load('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), allow_pickle=True)
        step_param = 0.02 # Gyr
    #else:
        # neet to figure it out for xoff
        
    # get pair indicies for the given r
    pairs_selected_arr = getPairIndicies(pairs_selected[0], r)
    pairs_all_arr = getPairIndicies(pairs_all[0], r)
    
    count_mz_matched_pairs = []
    
    # loop over all the pairs at the given separation
    for pairs in pairs_selected_arr:
        count_per_pair = 0
        # get mass ratio and mean z of the selection cut pair
        m_ratio = massRatios(pairs, m_arr)
        mean_z, mean_param = meanZ(pairs, z_arr), meanZ(pairs, param_arr)
        
        # count all halo pairs in the same mass and z bin as this pair
        for i in pairs_all_arr:
            mass_condition = (m_ratio - mr_min <= massRatios(i, m_arr) <= m_ratio + mr_max)
            z_condition = (mean_z - step_z) < meanZ(i, z_arr) < (mean_z + step_z)
            param_condition =  (mean_param - step_param) < meanZ(i, param_arr) < (mean_param + step_param)
            
            # count pairs that pass the conditions
            if mass_condition and z_condition and param_condition:
                count_per_pair += 1
        count_mz_matched_pairs.append(count_per_pair)
    np.save('Data/pairs_z%.1f/Major_dv_pairs/Controls/control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, r), count_mz_matched_pairs, allow_pickle=True)
    return 

def decideBools(keyword = 'all'):
    """
    Function decides the values of the booleans based on the keyword
    @keyword :: keyword decides where to save the file 
    takes three values -- 
    @keyword == 'dv' :: only redshift (velocity) criteria
    @keyword == 'mm and dv' :: major merger and redshift criteria
    @keyword == 'all' :: considers all pairs (not major merger or redshift cuts)
    """
    if keyword == 'dv':
        major_mergers_only, delta_v_cut = False, True
    if keyword == 'mm':
        major_mergers_only, delta_v_cut = True, False
    if keyword == 'mm and dv':
        major_mergers_only, delta_v_cut = True, True
    if keyword == 'all':
        major_mergers_only, delta_v_cut = False, False
    return major_mergers_only, delta_v_cut 