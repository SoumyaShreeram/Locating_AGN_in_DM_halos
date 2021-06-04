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
from scipy.stats import gaussian_kde
import os

# plotting imports
import matplotlib
import matplotlib.pyplot as plt

# personal imports
import Agn_incidence_from_Major_Mergers as aimm
import plotting_cswl05 as pt


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


def openPairsFiles(key = 'mm and dv', data_dir = 'Data/pairs_z2.0/', redshift_limit = 2, mass_max = 3, dz_cut = 0.001, pixel_no='000000'):
    """
    Function to open all the files with 
    """
    # get shell volume and projected radius bins
    r_p, shell_volume = aimm.shellVolume()
    pairs_idx_all, n_pairs_arr = [], []
    
    if key == 'mm and dv':
        filename = 'Major_dv_pairs/p_id_pixel%s_mm%d_dz%.3f.npy'%(pixel_no, mass_max, dz_cut)

    pairs_idx = np.load(data_dir+filename, allow_pickle=True)
    return pairs_idx

def getSnapshotZ(hd_z_halo):
    """
    Function gets the z and a value at the snapshot of the UNIT simulation for all the input halos
    """
    # load the file that has the information about the snapshots
    fname = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS', 'fixedAmp_InvPhase_001', 'snapshot_redshift_list.txt')
    shell_redshifts = np.loadtxt(fname)
    
    # reads of the shell z and a from the file
    shell_z = [i[1] for i in shell_redshifts]
    shell_a = [i[2] for i in shell_redshifts]
    
    # reframe the array as a tuple array with upper and lower limits of the shell
    shell_z_edges = np.array([[shell_z[i+1], shell_z[i]] for i in range(len(shell_z)-1)])
    shell_z_edges = np.transpose(shell_z_edges)
    
    shell_a_edges = np.array([[shell_a[i+1], shell_a[i]] for i in range(len(shell_a)-1)])
    shell_a_edges = np.transpose(shell_a_edges)
    
    # scale of last mm for all the input halos
    a = hd_z_halo['HALO_scale_of_last_MM']
    
    # convert the halo redshifts into snapshot redshifts
    zsnap_halo, asnap_halo = [], []
    for i, z in enumerate(hd_z_halo['redshift_R']):
        # get the upper shell edge
        zsnap_halo.append(shell_z_edges[0][(shell_z_edges[0] <= z) & (z <= shell_z_edges[1])][0])
        asnap_halo.append(shell_a_edges[1][(shell_a_edges[0] >= a[i]) & (a[i] >= shell_a_edges[1])][0])
    return zsnap_halo, asnap_halo

def calTmm(cosmo, asnap_halo, zsnap_halo):
    """
    Function calculates the time since last major merger, mm, for the halos given the halo z and scale, a, of last mm
    """
    # convert the merger scale factor into redshift
    merger_z = (1/np.array([asnap_halo]))-1

    # convert the merger & current redshifts into lookback time
    merger_time = cosmo.lookback_time(merger_z)
    current_time = cosmo.lookback_time(zsnap_halo)

    # difference in lookback time between the merger and AGN redshift
    diff_time = merger_time-current_time
    return diff_time

def selectParameterPairs(pairs_idx, r, halo_param_arr, param):
    """
    Select pairs that pass the parameter cuts
    @pairs_idx :: arr of tuples with the chosen pairs < r 
    @r :: index of the radius bin
    @param_arr_all_halos :: arr with values of the parameter of concern 
    @param :: list with [lower_limit, upper_limit] of the parameter bin
    """
    count_pairs = 0
    
    if r == 0:        
        pairs_selected = pairs_idx[r]
        
    if r > 0:
        pairs_this_bin = pairs_idx[r]
        pairs_previous_bin = pairs_idx[r-1]
        
        # get rid of the pairs that were counted in the previous radius bin
        pairs_selected = [current_pair for current_pair in pairs_this_bin if current_pair not in pairs_previous_bin]
        pairs_selected = np.array(pairs_selected)
    
    # process the pairs in the given radius bin, which passed all the mm and dv criteria 
    for i, p in enumerate(pairs_selected): 
        # get the  param values of the pair 
        param_p1, param_p2 = halo_param_arr[int(p[0])],  halo_param_arr[int(p[1])]
        
        if (param[0] <= param_p1 < param[1]) or  ( param[0] <= param_p2 < param[1]):
            count_pairs += 1
    return count_pairs


def generateDeciles(diff_t_mm_arr, tile = 10):
    "Function generates the deciles for the parameter arr"
    deciles = [int(i*((len(diff_t_mm_arr)-1)/tile)) for i in np.arange(tile+1)]
    dt_m_arr = np.sort(diff_t_mm_arr)[deciles]
    
    dt_m_bins_arr = [[dt_m_arr[i], dt_m_arr[i+1]] for i in np.arange(len(dt_m_arr)-1)]
    return dt_m_bins_arr

def concatAllTmmFiles(dt_m_arr, key, redshift_limit=2, param='t_mm'):
    """
    Function to concatenate all the files containing pairs for different T_mm criteria
    """
    r_p, _ = aimm.shellVolume()
    n_pairs_t_mm_all = np.zeros( (0, len(r_p) ) )
    
    if key == 'mm and dv':
        data_dir = 'Data/pairs_z%.1f/Major_dv_pairs/'%redshift_limit
    if key == 'all':
        data_dir = 'Data/pairs_z%.1f/'%redshift_limit    
        
    for dt_m in dt_m_arr:
        if param == 't_mm':
            n_pairs_t_mm = np.load(data_dir+'all_pairs_%s%.1f.npy'%(param, dt_m), allow_pickle=True)
        if param == 't_mm bins':
            n_pairs_t_mm = np.load(data_dir+'all_pairs_%s%.2f-%.2f.npy'%('t_mm', dt_m[0], dt_m[1]), allow_pickle=True)
        if param == 'x_off':
            n_pairs_t_mm = np.load(data_dir+'all_pairs_%s%.2f-%.2f.npy'%(param, dt_m[0], dt_m[1]), allow_pickle=True)
            
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
    f_pairs = num_pairs/(N*shell_volume[:len(num_pairs)])
    return f_pairs, error(num_pairs)/(N*shell_volume[:len(num_pairs)])

def Gamma(num_pairs, n):
    """
    Function to convert the number of pairs into a fractional number density per shell
    @redshift_limit :: the initial redshift limit set on the sample (needed for opening dir)
    """
    # get shell volume and projected radius bins
    r_p, shell_volume = aimm.shellVolume()
    
    # normalization
    total_pairs = np.sum( n*(n - 1)/2 )
    
    # number density
    Gamma = num_pairs/(total_pairs*shell_volume)
    error_Gamma = error(num_pairs)/(total_pairs*shell_volume)
    return Gamma, error_Gamma 

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

def tuplePairArr(pairs_idx):
    "Function generates an array that holds all the pair indicies"
    pairs_arr = np.zeros((0, 2))
    two_idx = []
    
    # get the indicies of the pair
    for j in range(pairs_idx.shape[0]):
        if len(pairs_idx[j]) >= 1:
            for p in pairs_idx[j]:
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

def decideWhereToSaveControlPairs(count_mz_matched_pairs, r, key = 'pairs', redshift_limit =2, dt_m_bins = [0.5, 1.0]):
    "Function decides where to save the control pairs"
    if key == 'pairs':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Controls/control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, r), count_mz_matched_pairs, allow_pickle=True)
    
    if key == 'selection':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Selection_applied/Controls/control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, r), count_mz_matched_pairs, allow_pickle=True)
    
    if key == 'self_pairs':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Controls/self_control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, r), count_mz_matched_pairs, allow_pickle=True)
        
    if key == 'tmm_pairs':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Tmm_%.2f-%.2fGyr/Controls_mztmm/control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, dt_m_bins[0], dt_m_bins[1],r), count_mz_matched_pairs, allow_pickle=True)
        
    if key == 'tmm_self_pairs':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Tmm_%.2f-%.2fGyr/Controls_mztmm/self_control_pairs_idx_r%.1f_mzTmm.npy'%(redshift_limit, dt_m_bins[0], dt_m_bins[1],r), count_mz_matched_pairs, allow_pickle=True)
    return


def getMZmatchedPairs(hd_halo, pairs_all, pairs_selected, r, mr_min = 0.15, mr_max = 2, redshift_limit=2, step_z = 0.01, param = 't_mm', dt_m_bins = [0.5, 1.0], key = 'pairs'):
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
            if pairs[0] != i[0] and pairs[1] != i[1]:
                mass_condition = (m_ratio == massRatios(i, m_arr))
                z_condition = (mean_z == meanZ(i, z_arr) )
                param_condition =  (mean_param == meanZ(i, param_arr) )

                # count pairs that pass the conditions
                if mass_condition and z_condition and param_condition:
                    count_per_pair += 1
        count_mz_matched_pairs.append(count_per_pair)
    
    decideWhereToSaveControlPairs(count_mz_matched_pairs, r, key = key, redshift_limit = redshift_limit, dt_m_bins = dt_m_bins)
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
    if keyword == 'mm and dv' or keyword == 'selection':
        major_mergers_only, delta_v_cut = True, True
    if keyword == 'all':
        major_mergers_only, delta_v_cut = False, False
    return major_mergers_only, delta_v_cut 

def getRMZ(hd_halo, pairs_mm_dv_all, r):
    """
    Function gets the mass ratio and redshift distribution for the pairs
    """
    m_arr, z_arr = hd_halo['galaxy_SMHMR_mass'], hd_halo['redshift_R']
    
    
    # no need to get rid of pairs from previous bin for r=0
    if r == 0: 
        pairs_selected = getPairIndicies(pairs_mm_dv_all[0], r)

    # for r > 0 we need to get rid of pairs from previous bin
    if r > 0:
        pairs_this_bin = getPairIndicies(pairs_mm_dv_all[0], r)
        pairs_previous_bin = getPairIndicies(pairs_mm_dv_all[0], int(r-1))

        pairs_selected = [current_pair for current_pair in pairs_this_bin if current_pair not in pairs_previous_bin]
        pairs_selected = np.array(pairs_selected)

    mass_ratios_arr, mean_z_arr = [], []

    for pairs in pairs_selected:
        mass_ratios_arr.append(massRatios(pairs, m_arr))
        mean_z_arr.append(meanZ(pairs, z_arr))

    separation_mass_z = np.array([mass_ratios_arr, mean_z_arr], dtype=object)
    return separation_mass_z, np.unique(pairs_selected)

def getNonCumulative(n_pairs_dt_all):
    "Function to get the no cumulative pairs"
    for i in np.arange(1, len(n_pairs_dt_all)):
        n_pairs_dt_all[i, :] = n_pairs_dt_all[i, :] - n_pairs_dt_all[i-1, :]
    return n_pairs_dt_all

def getMassRatioMeanZpairs(hd_z_halo, pairs_all, r_p, generate_mz_mat = True, bins = 25, m_ratio_min = 0.79, m_ratio_max = 1.3, mean_z_min = 0.03,  mean_z_max = 1.9, param='all', redshift_limit=2):
    """
    Function to get the mass and
    """
    if generate_mz_mat:
        m_bin_edges = np.linspace(m_ratio_min, m_ratio_max, bins)
        z_bin_edges = np.linspace(mean_z_min, mean_z_max, bins)

        # create a 2d matrix to do a contour plot
        mass_mat_2d = np.zeros(( bins-1, len(r_p) ) )
        z_mat_2d = np.zeros(( bins-1, len(r_p) ) )
        pairs_idx_all = []
        for r in range(len(r_p)):
            r_m_z_arr, pairs_idx = getRMZ(hd_z_halo, pairs_all, r) 
            pairs_idx_all.append(pairs_idx)
            
            # extract the mass ratio and mean z arrays
            m_arr, z_arr = r_m_z_arr

            counts_m_arr, m_bin_edges = np.histogram(m_arr, bins=m_bin_edges)
            counts_z_arr, z_bin_edges = np.histogram(z_arr, bins=z_bin_edges)

            mass_mat_2d[:, r] = counts_m_arr
            z_mat_2d[:, r] = counts_z_arr

        np.save('Data/mz_mat_%s.npy'%param, np.array([mass_mat_2d, z_mat_2d], dtype=object),allow_pickle=True)
        np.save('Data/pairs_z%.1f/chosen_idx_%s.npy'%(redshift_limit, param), pairs_idx_all, allow_pickle=True)
        mz_mat = np.array([mass_mat_2d, z_mat_2d], dtype=object)
    else:
        mz_mat = np.load('Data/mz_mat.npy',allow_pickle=True)
        pairs_idx_all = np.load('Data/pairs_z%.1f/chosen_idx_%s.npy'%(redshift_limit, param), allow_pickle=True)
    return mz_mat, pairs_idx_all

def convertPairIdxIntoHaloIdx(pairs):
    unique_halo_indicies = []
    for p in pairs:
        halo_idx = []
        # concatenate the halo indicies accross all radius
        halo_idx = np.concatenate(p, axis=None)

        # choose the unique indices alone
        halo_idx = np.unique(halo_idx)

        # save these unique indicies for each case of selection criteria
        unique_halo_indicies.append(halo_idx)

    unique_halo_indicies = np.concatenate(unique_halo_indicies, axis=None)
    unique_halo_indicies = np.unique(unique_halo_indicies)
    return unique_halo_indicies

def gaussianKde2D(a, b):
    "Generates the Gaussian kde"
    X, Y = np.mgrid[np.min(a):np.max(a):100j, np.min(b):np.max(b):100j]
    
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    values = np.vstack([a, b])
    
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return Z


def selectionHalos(hd_z_halo, diff_t_mm_arr,  xoff_min=0.2, xoff_max=0.3, tmm_min=1, tmm_max=3):
    xoff_condition = (hd_z_halo['HALO_Xoff']/hd_z_halo['HALO_Rvir'] > xoff_min) & (hd_z_halo['HALO_Xoff']/hd_z_halo['HALO_Rvir'] < xoff_max)
    tmm_condition = ( diff_t_mm_arr >  tmm_min ) & (diff_t_mm_arr <  tmm_max )

    total_conditions = xoff_condition & tmm_condition
    return total_conditions

def saveSeparationIndicies(all_idx, r, keyword='all', redshift_limit=2, mass_max=3, dz_cut=0.001, pixel_no='000000'):
    """
    Function to decide where to save the separation indicies
    """
    if keyword == 'mm':
            np.save('Data/pairs_z%.1f/Major_pairs/pairs_idx_r%.3f_mm%d.npy'%(redshift_limit, r, mass_max), all_idx, allow_pickle=True)
            print('\n --- Saved mm and dv file --- ')
         
    if keyword == 'mm and dv':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/p_id_pixel%s_mm%d_dz%.3f.npy'%(redshift_limit, pixel_no, mass_max, dz_cut), all_idx, allow_pickle=True)
        print('\n --- Saved mm and dv file --- ')

    if keyword == 'dv':
        np.save('Data/pairs_z%.1f/dv_pairs/pairs_idx_r%0.3f_dz%.3f.npy'%(redshift_limit, r, dz_cut), all_idx, allow_pickle=True)
        print('\n --- Saved dv file --- ')

    # if you want to save all the pairs
    if keyword == 'all':
        np.save('Data/pairs_z%.1f/pairs_idx_r%0.3f.npy'%(redshift_limit, r), all_idx, allow_pickle=True)
        print('\n --- Saved no cuts file --- ')
        
    if keyword == 'selection':
        np.save('Data/pairs_z%.1f/Major_dv_pairs/Selection_applied/pairs_idx_r%.3f_mm%d_dz%.3f.npy'%(redshift_limit, r, mass_max, dz_cut), all_idx, allow_pickle=True)
        print('\n --- Saved mm and dv selected files --- ')
    return

def getPlotModel(pairs_all, hd_z_halo, diff_t_mm_arr, vol, xoff_min=0.17, xoff_max=0.54, tmm_min=0.6, tmm_max=1.2, redshift_limit=2):
    """
    Function generates plots and saves the model for number of halos as a function of separations
    """
    total_conditions = selectionHalos(hd_z_halo, diff_t_mm_arr, xoff_min=xoff_min, xoff_max=xoff_max, tmm_min=tmm_min, tmm_max=tmm_max)
    hd_agn_halo = hd_z_halo[total_conditions]
    print("AGNs: %d"%(len(hd_agn_halo)) )

    pairs_selected = openPairsFiles(key = 'selection', param_bins = [xoff_min, xoff_max, tmm_min, tmm_max])

    fig, ax = plt.subplots(3,1,figsize=(5,14))
    model = pt.plotModelResults(ax, hd_z_halo, pairs_all, pairs_selected, vol)

    np.save('Data/pairs_z%.1f/prediction_xoff%.2f-%.2f_tmm%.1f-%.1fGyr.npy'%(redshift_limit, xoff_min, xoff_max, tmm_min, tmm_max), np.array(model), allow_pickle=True)
    return 
