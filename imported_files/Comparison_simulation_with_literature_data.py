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

def openPairsFiles(key, data_dir = 'Data/pairs_z2.0/', redshift_limit = 2, mass_max = 3, dv_cut = 500):
    """
    Function to open all the files with 
    """
    # get shell volume and projected radius bins
    r_p, dr_p, shell_volume = aimm.shellVolume()
    pairs_idx_all, n_pairs_arr = [], []
    
    for i in range(len(r_p)):
        if key == 'mm and dv':
            filename = 'Major_pairs/pairs_idx_r%.1f_mm%d_dv%d.npy'%(i, mass_max, dv_cut)
        if key == 'dv':
            filename = 'dv_pairs/pairs_idx_r%.1f_dv%d.npy'%(i, dv_cut)
        if key == 'all':
            filename = 'pairs_idx_r%d.npy'%(i)
            
        pairs_idx = np.load(data_dir+filename, allow_pickle=True)
        n_pairs = countSelectedPairs(pairs_idx, print_msg = False)
        
        n_pairs_arr.append(n_pairs)
        pairs_idx_all.append(pairs_idx)        
    return np.array([pairs_idx_all, n_pairs_arr], dtype=object)

def saveTmmFiles(key, dt, arr , redshift_limit = 2):
    """
    Function decides where to save the tmm evaluated files
    """
    if key == 'mm and dv':
        np.save('Data/pairs_z%.1f/Major_pairs/all_pairs_t_mm%.1f.npy'%(redshift_limit, dt), arr, allow_pickle=True)
        
    if key == 'dv':
        np.save('Data/pairs_z%.1f/dv_pairs/all_pairs_t_mm%.1f.npy'%(redshift_limit, dt), arr, allow_pickle=True)

    # if you want to save all the pairs
    if key == 'all':
        np.save('Data/pairs_z%.1f/all_pairs_t_mm%.1f.npy'%(redshift_limit, dt), arr, allow_pickle=True)
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

def defineTimeSinceMergeCut2(hd_obj, pairs_idx, cosmo, diff_t_mm_arr, time_since_merger = 1, redshift_limit = 2):
    """
    Define the time since merger cut 
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
                diff_time_pair = diff_t_mm_arr[p_idx]
                # only consider pairs that pass this time since merger-scale criterion
                if (diff_time_pair <= time_since_merger) or  (diff_time <= time_since_merger):
                    t_mm_idx.append(p_idx)
                    
        # save this info for the given halo in the object array
        all_t_mm_idx.append(t_mm_idx)
    
    count_t_mm = countSelectedPairs(all_t_mm_idx, print_msg = False, string = 'T_mm = %d Gyr: '%time_since_merger)
    return all_t_mm_idx, count_t_mm


def concatAllTmmFiles(dt_m_arr, key, redshift_limit=2):
    """
    Function to concatenate all the files containing pairs for different T_mm criteria
    """
    r_p, _, _ = aimm.shellVolume()
    n_pairs_t_mm_all = np.zeros( (0, len(r_p) ) )
    
    if key == 'mm and dv':
        data_dir = 'Data/pairs_z%.1f/Major_pairs/'%redshift_limit
    if key == 'dv':
        data_dir = 'Data/pairs_z%.1f/dv_pairs/'%redshift_limit
    if key == 'all':
        data_dir = 'Data/pairs_z%.1f/'%redshift_limit    
        
    for dt_m in dt_m_arr:
        n_pairs_t_mm = np.load(data_dir+'all_pairs_t_mm%.1f.npy'%(dt_m), allow_pickle=True)
        
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
    N = 2*total_num_pairs*(total_num_pairs - 1)
    
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

def getPairIndicies(pairs_idx, r, keyword = ''):
    "Function generates an array that holds all the pair indicies"
    pairs_arr = np.zeros((0, 2))
    two_idx = []
    
    # get the indicies of the pair
    for j in range(pairs_idx[r].shape[0]):
        if keyword == 'all':
            if len(pairs_idx[r][j]) > 1:
                for p in pairs_idx[r][j]:
                    two_idx = [j, p]
                    pairs_arr = np.append(pairs_arr, [two_idx], axis=0)
        else:
            if len(pairs_idx[r][j]) >= 1:
                for p in pairs_idx[r][j]:
                    two_idx = [j, p]
                    pairs_arr = np.append(pairs_arr, [two_idx], axis=0)
    return pairs_arr

def getHalosWithMaxRcompanion(pairs_mm_all, keyword = 'mm and dv', index_of_max_companion = 7):
    """
    Function to get all the halos within the defined radius r_p[index_of_max_companion] (usually 80 kpc)
    @pairs_mm_all[0] :: list of lists holding indicies of all halos with a companion <r_p[r]
    @keyword :: can be 'dv' --> redshift selection only
                'mm and dv' --> major merger and redshift selection
                'all' --> no selection
    @index_of_max_companion :: this is the index of the r_p arr 
                            -->  decides to save companions within r_p[idx]
    @returns :: unique elements of an (N, 2) array containing idx of all halos with a pairs within r_p[idx]
    """
    halos_w_pair_lessthan_r = np.zeros((0,2))
    
    for r in range(index_of_max_companion):
        pairs_arr = getPairIndicies(pairs_mm_all[0], r, keyword = keyword)
        halos_w_pair_lessthan_r = np.append(halos_w_pair_lessthan_r, pairs_arr, axis=0)

    halos_w_pair_lessthan_r = np.unique(np.concatenate(halos_w_pair_lessthan_r, axis=None))
    return halos_w_pair_lessthan_r

def getRedshiftMatchedControl(hd_halo, pairs_all, r, keyword='dv', step_z=0.01, redshift_limit=2):
    """
    Function gets the redshift matched control sample for all the pairs at a given separation
    @hd_halo :: astropy Table file containing all the halos < redshift_limit
    @pairs_all[0] :: contains all the indicies of the halos in a pair  
    @returns :: np object containing the indicies of the halos chosen in the 'control sample' for every pair & the corresponding count of the total halos in the control 
    """
    # depricated initialition of a list of lists arr ;)
    z_matched_arr = []
    
    # get pair indicies for the given r
    pairs_arr = getPairIndicies(pairs_all[0], r, keyword = keyword)
    
    # get indicies of all halos in a pair with r <~ 80 kpc
    halos_w_pair_lessthan_r = getHalosWithMaxRcompanion(pairs_all)
    
    count_z_matched_bkgs = 0
    z_arr = hd_halo['redshift_R']
    
    # loop over all the pairs at the given separation
    for pairs in pairs_arr:
        # match redshifts
        z1, z2 = z_arr[int(pairs[0])], z_arr[int(pairs[1])]
        mean_z = (z1 + z2)/2
        
        # count all objects in the same redshift bin
        z_matched = [i for i in range(len(z_arr)) if (mean_z - step_z) < z_arr[i] < (mean_z + step_z) ]
        
        # keep all halos in the bin that DO NOT have companions with r <~ 80 kpc
        keeping_idx = [keep for keep in range(len(z_matched)) if z_matched[keep] not in halos_w_pair_lessthan_r]
        z_matched = np.array(z_matched)[keeping_idx]
        
        # save this useful info that is just found out
        z_matched_arr.append(z_matched)
        count_z_matched_bkgs += len(z_matched)
        
    z_matched_control = np.array([z_matched_arr, count_z_matched_bkgs], dtype=object)
    np.save('Data/pairs_z%.1f/control_idx_r%.1f.npy'%(redshift_limit, r), z_matched_control, allow_pickle=True)
    return

def getMassRatios(pairs_arr, hd_halo):
    """
    Function calculates the mass ratio between the pairs
    """
    mass_ratios = []
    m_arr = hd_halo['galaxy_SMHMR_mass']
    
    for pairs in pairs_arr:
        # get masses of the pair
        m1, m2 = m_arr[int(pairs[0])], m_arr[int(pairs[1])]
        ratio = m1/m2        
        mass_ratios.append(ratio)
    return mass_ratios


def getMassMatchedPairs(hd_halo, pairs_all, pairs_selected, r, mr_min = 0.15, mr_max = 6, redshift_limit=2):
    """
    Function matches the mass of pairs 'with selection cuts' with those 'with no selections cuts'
    """
    # depricated initialition of a list of lists arr ;)
    mass_matched_pair_arr = []
    
    # get pair indicies for the given r
    pairs_selected_arr = getPairIndicies(pairs_selected[0], r)
    pairs_all_arr = getPairIndicies(pairs_all[0], r, keyword = 'all')
    
    mass_ratios = getMassRatios(pairs_all_arr, hd_halo)
    m_arr = hd_halo['galaxy_SMHMR_mass']
    
    count_m_matched_pairs = 0
    # loop over all the pairs at the given separation
    for pairs in pairs_selected_arr:
        # match redshifts
        m1_selected, m2_selected = m_arr[int(pairs[0])], m_arr[int(pairs[1])]
        m_ratio = m1_selected/m2_selected
        
        # count all objects in the same mass bin
        m_matched = [i for i in range(len(pairs_all_arr)) if m_ratio - mr_min <= mass_ratios[i] <= m_ratio + mr_max ]
        
        # save this useful info that is just found out
        mass_matched_pair_arr.append(m_matched)       
        count_m_matched_pairs += len(m_matched)
        
    m_matched_control = np.array([mass_matched_pair_arr, count_m_matched_pairs], dtype=object)
    np.save('Data/pairs_z%.1f/control_pairs_idx_r%.1f.npy'%(redshift_limit, r), m_matched_control, allow_pickle=True)
    return

def getMeanZforControlPairs(hd_halo, pairs_all_arr, redshift_limit, r):
    """
    Functino to get the z of the control pairs
    """
    mean_z_arr = []
    z_arr = hd_halo['redshift_R']
    
    # load the chosen mass matched indicies
    m_matched_control = np.load('Data/pairs_z%.1f/control_pairs_idx_r%.1f.npy'%(redshift_limit, r), allow_pickle=True)
    
    for pairs in m_matched_control[0]:
        # get masses of the pair
        z1, z2 = z_arr[int(pairs[0])], z_arr[int(pairs[1])]
        mean_z = (z1 + z2)/2        
        mean_z_arr.append(mean_z)
    return mean_z_arr

def getRedshiftMatchedPairs(hd_halo, pairs_all, pairs_selected, r, mr_min = 0.15, mr_max = 6, redshift_limit=2):
    """
    Function to match the redshifts of the pairs 
    """
    # get pair indicies for the given r
    pairs_selected_arr = getPairIndicies(pairs_selected[0], r)
    pairs_all_arr = getPairIndicies(pairs_all[0], r, keyword = 'all')
    
    count_z_matched_bkgs = 0
    z_arr = hd_halo['redshift_R']
        
    # for every pair
    for pairs in pairs_selected_arr:
        # match redshifts
        z1, z2 = z_arr[int(pairs_selected_arr[0])], z_arr[int(pairs_selected_arr[1])]
        mean_z = (z1 + z2)/2
        
        # redshift of the 'no selection halo pairs'
        mean_z_arr = getMeanZforControlPairs(hd_halo, pairs_all_arr, redshift_limit, r)
        # count all objects in the same redshift bin
        z_matched = [i for i in range(len(z_arr)) if (mean_z - step_z) < z_arr[i] < (mean_z + step_z) ]
    
        # look into the indicies that pass the z criteria
    return