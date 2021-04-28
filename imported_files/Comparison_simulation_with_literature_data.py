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

def countPairs(pairs_idx, print_msg=True):
    """
    Function to count pairs, for every DM halo, without including self
    """
    count_pairs, count_no_pairs = 0, 0
    for p in pairs_idx:
        if len(p) > 1:
            count_pairs += len(p)
        # ignore the halo itself in the counting of pairs
        else:
            count_no_pairs += 1
    fraction_of_pairs = count_pairs/count_no_pairs
    
    if print_msg:
        print('%d pairs found among %d halos; fraction of pairs: %.4f'%(count_pairs, len(pairs_idx), fraction_of_pairs))
        
    return count_pairs, fraction_of_pairs

def countSelectedPairs(all_mm_idx, print_msg = True, string = 'Major merger cut: '):
    count_selected_pairs = 0
    
    for j in range( len(all_mm_idx) ):
        
        if len(all_mm_idx[j]) >= 1:
            count_selected_pairs += len(all_mm_idx[j][1:])
    
    if print_msg:
        print(string+'%d selected pairs'%count_selected_pairs)
    return count_selected_pairs

def majorMergerSelection(hd_halo, pairs_idx, mass_min = 0.33, mass_max = 3): 
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
        
        # if there are more than 'self'-pairs
        if len(p) > 1:
            for p_idx in p[1:]:
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

def deltaVelSelection(hd_halo, all_mm_idx, dv_cut=500, mm = True):
    """
    Function to choose the pairs that meet the delta v criterion
    @mm :: are you passing the index array after selecting major mergers?
    """
    all_dv_idx = []
    for i, mm  in enumerate(all_mm_idx):
        dv_idx = []
        
        # if major pairs are not selected before the all_mm_idx enters this function
        if not mm:
            # going through all the pairs to see their delta v criteria
            if len(mm) > 1:
                for m in mm[1:]:
                    dv = 3e5*(hd_halo[i]['redshift_R'] - hd_halo[m]['redshift_R']) 

                    # applying the selection criteria
                    if dv < dv_cut:
                        dv_idx.append(m)
        else:
            # going through all the major pairs to see their delta v criteria
            if len(mm) >= 1:
                for m in mm:
                    dv = 3e5*(hd_halo[i]['redshift_R'] - hd_halo[m]['redshift_R']) 

                    # applying the selection criteria
                    if dv < dv_cut:
                        dv_idx.append(m)
            
        all_dv_idx.append(dv_idx)
    count_dv_major_mergers = countSelectedPairs(all_dv_idx, string = 'Delta v %d cut: '%dv_cut)
    return all_dv_idx, count_dv_major_mergers

def openPairsFiles(data_dir = 'Data/Major_pairs/pairs_z2/', redshift_limit = 2, mass_max = 3, dv_cut = 500):
    """
    Function to open all the files with 
    """
    # get shell volume and projected radius bins
    r_p, dr_p, shell_volume = aimm.shellVolume()
    pairs_idx_all, n_pairs_arr = [], []
    
    for i in range(len(r_p)):
        pairs_idx = np.load(data_dir+'pairs_idx_r%d_mm%d_dv%d.npy'%(i, mass_max, dv_cut), allow_pickle=True)
        n_pairs = countSelectedPairs(pairs_idx, print_msg = False)
        
        n_pairs_arr.append(n_pairs)
        pairs_idx_all.append(pairs_idx)        
    return np.array([pairs_idx_all, n_pairs_arr], dtype=object)

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

def concatAllTmmFiles(dt_m_arr, redshift_limit=2, data_dir = 'Data/pairs_z2/'):
    """
    Function to concatenate all the files containing pairs for different T_mm criteria
    """
    r_p, _, _ = aimm.shellVolume()
    n_pairs_t_mm_all = np.zeros( (0, len(r_p) ) )
    
    for dt_m in dt_m_arr:
        n_pairs_t_mm = np.load(data_dir+'all_pairs_t_mm%.1f_r.npy'%(dt_m), allow_pickle=True)
        
        # save the counts for all radius bins for a given time since merger
        n_pairs_t_mm_all = np.append(n_pairs_t_mm_all, n_pairs_t_mm, axis=0)
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
    Function to convert the number of pairs into a fraction
    @dt_m_idx :: the time since merger array index 
                --> dt_m_idx = 1 corresponds to 1 Gyr; chosen from [0.5, 1, 2, 3, 4]
    @redshift_limit :: the initial redshift limit set on the sample (needed for opening dir)
    """
    num_pairs = all_pairs_vs_rp[1:] - all_pairs_vs_rp[:-1]
    
    # get shell volume and projected radius bins
    r_p, _, shell_volume = aimm.shellVolume()
    
    # normalization
    total_num_pairs = len(hd_obj)    
    N = total_num_pairs*(total_num_pairs - 1)
    
    f_pairs = num_pairs/(N*shell_volume)
    return f_pairs, error(num_pairs)

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