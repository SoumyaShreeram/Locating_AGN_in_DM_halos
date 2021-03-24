"""
01. Exploring parameters in DM halos and sub-halos

This python file contains the function of the corresponding notebook '01_Exploring_DM_Haloes.ipynb'.

The notebook is divided into the following sections:
1. Loading data and defining input parameters
2. Studying sizes of haloes and sub-haloes

**Script written by**: Soumya Shreeram <br>
**Project supervised by**: Johan Comparat <br>
**Date**: 23rd February 2021
"""

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# system imports
import os
import sys

# scipy modules
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
"""

1. Functions for loading data

"""

def getFilename(look_dir, pixel_no):
    """
    Function sets the path and generates filename for opening the files
    @look_dir :: keyword to decide which files to select (galaxy, agn, or halos)
    
    Returns @filename :: the fullname of the chosen file directory
    """
    agn_dir = 'cat_AGN_all'
    hp_dir = 'cat_eRO_CLU_b8_CM_0_pixS_20.0_galNH'
    gal_dir = 'cat_GALAXY_all'
    
    # choose the directory of interest
    if look_dir == 'agn':
        dir_name = agn_dir
        pixel_file = pixel_no + '.fit'
        
    if look_dir == 'galaxy':
        dir_name = gal_dir
        pixel_file = pixel_no + '.fit'
        
    if look_dir == 'halo':
        dir_name = hp_dir
        pixel_file = 'c_'+ pixel_no[0] + '_N_%d.fit'%pixel_no[1]
    
    # set path
    filename = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS', 'fixedAmp_InvPhase_001', dir_name, pixel_file)
    return filename

def getAgnData(hd_agn, agn_FX_soft, redshift_limit):
    """
    Function to get the relavant data for AGNs
    @hd_agn :: table file with all relevant info on AGNs
    @AGN_FX_soft, AGN_SDSS_r_magnitude :: limits on flux, and brightness to be classified as an AGN
    @redshift_limit :: decides until which AGNs to consider in the sample
    --> typically, we are interested in the low-z universe for this project
    
    Returns:: @pos_z :: positions (ra, dec) and redshifts of the downsampled AGNs
    """
    # criteria on flux and brightness for selection of AGNs 
    downsample_agn = (hd_agn['FX_soft']>agn_FX_soft) & (hd_agn['redshift_R']<redshift_limit)
    
    # get the ra, dec and z for the AGNs
    ra_AGN = hd_agn['RA'][downsample_agn]
    dec_AGN = hd_agn['DEC'][downsample_agn]
    z_AGN = hd_agn['redshift_R'][downsample_agn]
    pos_z = [ra_AGN, dec_AGN, z_AGN]
    
    # scale factor of last major merger
    scale_merger = hd_agn['HALO_scale_of_last_MM'][downsample_agn]
    
    return pos_z, scale_merger, downsample_agn

def getGalaxyData(hd_gal, galaxy_SMHMR_mass, redshift_limit):
    """
    Function to get relavant data for galaxies
    
    """
    downsample_gal = (hd_gal['galaxy_SMHMR_mass']>galaxy_SMHMR_mass) & (hd_gal['redshift_R']<redshift_limit)
    
    # get the ra, dec and z for the AGNs
    ra_gal = hd_gal['RA'][downsample_gal]
    dec_gal = hd_gal['DEC'][downsample_gal]
    z_gal = hd_gal['redshift_R'][downsample_gal]
    pos_z = [ra_gal, dec_gal, z_gal]
    
    # scale factor of last major merger
    scale_merger = hd_gal['HALO_scale_of_last_MM'][downsample_gal]  
    
    return pos_z, scale_merger, downsample_gal

def getHaloData(hd_halo, cluster_params, redshift_limit):
    """
    Function to get relavant data for halos that can be classified as clusters
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them
    @cluster_params :: contains the minimum mass for a central halo to be classified as a cluster (solar masses)
    @redshift_limit :: limit upto which the objects are chosen
    """
    min_cluster_mass = cluster_params[0]
    
    # selection conditions for halos
    redshift_condition = (hd_halo['redshift_R']<redshift_limit) 
    cen_halo_condition = (hd_halo['HALO_M500c']>min_cluster_mass) & (hd_halo['HALO_pid']==-1)
    sat_halo_condition = (hd_halo['HALO_pid']!=-1)
    
    # downsample based on central/sattelite-halo information
    cen = redshift_condition & cen_halo_condition
    sat = redshift_condition & sat_halo_condition
      
    # ra for cen and sat halos
    ra_cen = hd_halo['RA'][cen]
    ra_sat = hd_halo['RA'][sat]
    
    # dec for cen and sat halos
    dec_cen = hd_halo['DEC'][cen]
    dec_sat = hd_halo['DEC'][sat]
    
    # redshift for cen and sat halos
    z_cen = hd_halo['redshift_R'][cen]
    z_sat = hd_halo['redshift_R'][sat]
        
    pos_z_cen = [ra_cen, dec_cen, z_cen]
    pos_z_sat = [ra_sat, dec_sat, z_sat]
    return pos_z_cen, pos_z_sat

def concatenateObjs(arr0, arr1, arr2):
    "Function to concatenate 3 arrays"
    return np.concatenate((np.array(arr0), np.array(arr1), np.array(arr2)), axis=None)

def concatMultipleArrays(arr, num_arrays):
    "Function to concatenate multiple 3-arrays' simultaneously"
    if num_arrays == 3:
        arr0, arr1, arr2 = arr[0], arr[1], arr[2]
        
        concat_arr0 = concatenateObjs(arr0[0], arr1[0], arr2[0])
        concat_arr1 = concatenateObjs(arr0[1], arr1[1], arr2[1])
        concat_arr2 = concatenateObjs(arr0[2], arr1[2], arr2[2]) 
    else:
        "Concatenation not yet developed: look up or do them separately"
    return concat_arr0, concat_arr1, concat_arr2

def getClusterPositionsRedshift(hd_halo0, hd_halo1, hd_halo2, cluster_params, redshift_limit):
    """
    Function to get the positions and redshifts of the clusters that pass the required criterion
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them 
    --> divided into 3 because each hd_halo holds info on 1000 halos alone
    @cluster_params :: contains clu_FX_soft, galaxy_mag_r, min_cluster_mass where
        @min_cluster_mass :: min mass for halo to be called a cluster
    @redshift_limit :: upper limit on redshift
    @Returns :: pos_z_clu :: [ra, dec, redshift] of all the clusters in the 3 files
    """
    # get positions and redshift of all the selected clusters
    pos_z_cen0, pos_z_sat0 = getHaloData(hd_halo0, cluster_params, redshift_limit)
    pos_z_cen1, pos_z_sat1 = getHaloData(hd_halo1, cluster_params, redshift_limit)
    pos_z_cen2, pos_z_sat2 = getHaloData(hd_halo2, cluster_params, redshift_limit)
    
    # concatenates the ra, dec, and z for the central clusters
    ra_cen_clu, dec_cen_clu, z_cen_clu = concatMultipleArrays([pos_z_cen0, pos_z_cen1, pos_z_cen2], num_arrays = 3)
    
    # concatenates the ra, dec, and z for the sub clusters
    ra_sat_clu, dec_sat_clu, z_sat_clu = concatMultipleArrays([pos_z_sat0, pos_z_sat1, pos_z_sat2], num_arrays = 3)

    pos_z_cen = [ra_cen_clu, dec_cen_clu, z_cen_clu]
    pos_z_sat = [ra_sat_clu, dec_sat_clu, z_sat_clu ]
    return pos_z_cen, pos_z_sat

def getMergerTimeDifference(merger_val, redshifts, cosmo):
    """
    Function to calculate the time difference between the merger time and current time
    @merger_val :: scale factor of the merged object
    @cosmo :: cosmology used for the calculation
    """
    # convert the merger scale factor into redshift
    merger_z = [z_at_value(cosmo.scale_factor, a) for a in merger_val]
    
    # convert the merger & current redshifts into lookback time
    merger_time = cosmo.lookback_time(merger_z)
    current_time = cosmo.lookback_time(redshifts)
    
    # difference in lookback time between the merger and AGN redshift
    diff_time = merger_time-current_time
    return diff_time

def cenSatObjects(conditions_obj, hd_obj, min_cluster_mass):
    "Function to obtain the merger rates of central and sattelite objects"
    # conditions
    cen = conditions_obj & (hd_obj['HALO_pid'] == -1) & (hd_obj['HALO_M500c']>min_cluster_mass)
    sat = conditions_obj & (hd_obj['HALO_pid'] != -1)
    
    # arrays
    cen_obj = hd_obj[cen]
    sat_obj = hd_obj[sat]
    return [cen_obj, sat_obj]