"""
01. Exploring parameters in DM halos and sub-halos

This python file contains the function of the corresponding notebook '01_Exploring_DM_Haloes.ipynb'.

The script is divided into the following sections:
1. Functions for loading data
2. Functions to study mergers in the central and satellite object distributions

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 23rd February 2021
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
    hp_dir = 'cat_GALAXY_all'
    clu_dir = 'cat_eRO_CLU_b8_CM_0_pixS_20.0_galNH'
    
    # choose the directory of interest
    if look_dir == 'agn':
        dir_name = agn_dir
        pixel_file = pixel_no + '.fit'
        
    if look_dir == 'galaxy' or look_dir == 'halo':
        dir_name = hp_dir
        pixel_file = pixel_no + '.fit'
        
    if look_dir == 'cluster':
        dir_name = clu_dir
        pixel_file = 'c_'+ pixel_no[0] + '_N_%d.fit'%pixel_no[1]
    
    # set path
    filename = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS', 'fixedAmp_InvPhase_001', dir_name, pixel_file)
    return filename


def getHeaders(pixel_no, obj, clu_files_no):
    """
    Function to get all the required headers to access simulation data
    @pixel_no :: pixel number chosen to open simulation file
    @obj :: objects whose headers you wish to open
    @clu_files_no :: number of files into which the cluster data is divided into
    
    @Returns :: hd_all :: list containing all the header files
    """
    hd_agn, hd_halo, hd_clu = [], [], []
    
    if np.any(obj=='agn'):
        agn_filename = getFilename('agn', pixel_no = pixel_no)
        hd_agn = Table.read(agn_filename, format='fits') 
                
    if np.any(obj=='halo'):
        halo_filename = getFilename('halo', pixel_no = pixel_no)
        hd_halo = Table.read(halo_filename, format='fits')
        
    if np.any(obj=='cluster'):
        hd_clu = []
        for i in range(clu_files_no):
            clu_filename = getFilename('cluster', pixel_no = [pixel_no, i])
            hd_c = Table.read(clu_filename, format='fits')
            hd_clu.append(hd_c)       
    return hd_agn, hd_halo, hd_clu

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
    if isinstance(galaxy_SMHMR_mass, (int, float)):
        downsample_gal = (hd_gal['galaxy_SMHMR_mass']>galaxy_SMHMR_mass) & (hd_gal['redshift_R']<redshift_limit)
    else:
        downsample_gal = (hd_gal['redshift_R']<redshift_limit)
        
    # get the ra, dec and z for the AGNs
    ra_gal = hd_gal['RA'][downsample_gal]
    dec_gal = hd_gal['DEC'][downsample_gal]
    z_gal = hd_gal['redshift_R'][downsample_gal]
    pos_z = [ra_gal, dec_gal, z_gal]
    
    # scale factor of last major merger
    scale_merger = hd_gal['HALO_scale_of_last_MM'][downsample_gal]  
    
    return pos_z, scale_merger, downsample_gal

def getClusterData(hd_halo, cluster_params, redshift_limit):
    """
    Function to get relavant data for classified clusters 
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

def concatMultipleArrays(pos_z_all, num_cluster_files):
    """
    Function to concatenate multiple 3-arrays' simultaneously
    @pos_z_all :: positions and redshifts from all the cluster files
    @num_cluster_files :: number of cluster files
    """
    ra, dec, z = [], [], []
    for num_clu in range(num_cluster_files):
        ra.append(pos_z_all[num_clu][0])
        dec.append(pos_z_all[num_clu][1])
        z.append(pos_z_all[num_clu][2])
    
    ra_all = np.concatenate(ra, axis=None)
    dec_all = np.concatenate(dec, axis=None)
    z_all = np.concatenate(z, axis=None)
    return ra_all, dec_all, z_all


def getClusterPositionsRedshift(hd_clu, cluster_params, redshift_limit):
    """
    Function to get the positions and redshifts of the clusters that pass the required criterion
    @hd_clu :: list of cluster headers (each header ahs info on 1000 clusters alone)
    @cluster_params :: contains halo_mass_500c and central_Mvir
    @redshift_limit :: upper limit on redshift
    @Returns :: pos_z_clu :: [ra, dec, redshift] of all the clusters in the 3 files
    """
    pos_z_cen_all, pos_z_sat_all = [], []
    
    for i in range(len(hd_clu)):
        # get positions and redshift of all the selected clusters
        pos_z_cen, pos_z_sat = getClusterData(hd_clu[i], cluster_params, redshift_limit)
        pos_z_cen_all.append(pos_z_cen)
        pos_z_sat_all.append(pos_z_sat)
        
    # concatenates the ra, dec, and z for the central clusters
    pos_z_cen_clu = concatMultipleArrays(pos_z_cen_all, len(hd_clu))
    pos_z_sat_clu = concatMultipleArrays(pos_z_sat_all, len(hd_clu))
    return pos_z_cen_clu, pos_z_sat_clu


def printNumberOfObjects(pos_z_AGN, pos_z_cen_clu, pos_z_sat_clu, pos_z_gal, pos_z_halo):
    """
    Function to print total number of objects in the chosen pixel file
    """
    print('AGNs/haloes in the pixel are in z = %.2f to %.2f'%(np.min(pos_z_AGN[2]), np.max(pos_z_AGN[2])))
    num_total_clusters = len(pos_z_cen_clu[2]) + len(pos_z_sat_clu[2])
    
    print('%d DM halos, %d clusters, %d AGNs, %d galaxies'%(len(pos_z_halo[0]), num_total_clusters, len(pos_z_AGN[2]), len(pos_z_gal[0])))
    return

"""

Functions to study mergers in the central and satellite object distributions

"""

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
    cen = conditions_obj & (hd_obj['HALO_pid'] == -1) & (hd_obj['HALO_Mvir']>min_cluster_mass)
    sat = conditions_obj & (hd_obj['HALO_pid'] != -1)
    
    # arrays
    cen_obj = hd_obj[cen]
    sat_obj = hd_obj[sat]
    return [cen_obj, sat_obj]