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

# import required packages

import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
import os
import numpy as np
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
    return pos_z

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
    return pos_z

def getHaloData(hd_halo, cluster_params, redshift_limit):
    """
    Function to get relavant data for halos that can be classified as clusters
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them
    @min_cluster_mass :: minimum mass for a halo to be classified as a cluster (solar masses)
    """
    min_cluster_mass = cluster_params[0]
    
    # select halos with cuts in Fx, BCG mag_r, M_vir, and z
    mass_z_condition = (hd_halo['HALO_M500c']>min_cluster_mass) & (hd_halo['redshift_R']<redshift_limit)
    
    downsample_halo = mass_z_condition
    
    # get the ra, dec and z for all the halos
    ra_halo = hd_halo['RA'][downsample_halo]
    dec_halo = hd_halo['DEC'][downsample_halo]
    z_halo = hd_halo['redshift_R'][downsample_halo]
    pos_z = [ra_halo, dec_halo, z_halo]  
    
    return pos_z, downsample_halo

def concatenateObjs(arr1, arr2, arr3):
    return np.concatenate((np.array(arr1), np.array(arr2), np.array(arr3)), axis=None)

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
    pos_z_clu0, _ = getHaloData(hd_halo0, cluster_params, redshift_limit)
    pos_z_clu1, _ = getHaloData(hd_halo1, cluster_params, redshift_limit)
    pos_z_clu2, _ = getHaloData(hd_halo1, cluster_params, redshift_limit)
    
    # concatenates the ra, dec, and z for the clusters
    ra_clu = concatenateObjs(pos_z_clu0[0], pos_z_clu1[0], pos_z_clu2[0])
    dec_clu = concatenateObjs(pos_z_clu0[1], pos_z_clu1[1], pos_z_clu2[1])
    z_clu = concatenateObjs(pos_z_clu0[2], pos_z_clu1[2], pos_z_clu2[2]) 

    pos_z_clu = [ra_clu, dec_clu, z_clu]
    return pos_z_clu

def getHostSatHalos(hd_halo, cluster_params, redshift_limit):
    """
    Function to get the position of central and satellite halos
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them
    @cluster_params :: contains clu_FX_soft, galaxy_mag_r, min_cluster_mass where
        @min_cluster_mass :: min mass for halo to be called a cluster
    @redshift_limit :: upper limit on redshift
    """
    # get positions and redshift of all the selected clusters
    _, downsample_halo = getHaloData(hd_halo, cluster_params, redshift_limit)
    
    # get central and sub halo related data
    cen = (hd_halo[downsample_halo]['HALO_pid']==-1)
    sat = (cen == False)
    
    # ra and dec for cen and sat halos
    ra_cen = hd_halo['RA'][cen]
    ra_sat = hd_halo['RA'][sat]
    
    dec_cen = hd_halo['DEC'][cen]
    dec_sat = hd_halo['DEC'][sat]
    
    pos_cen = [ra_cen, dec_cen]
    pos_sat = [ra_sat, dec_sat]
    return pos_cen, pos_sat

def getPositionsHostSatHalos(hd_halo0, hd_halo1, hd_halo2, cluster_params, redshift_limit):
    """
    Function to concatenate the positions of the halo files
    @hd_halo :: table with all relevant info on halos, clusters, and galaxies within them 
    --> divided into 3 because each hd_halo holds info on 1000 halos alone
    @cluster_params :: contains clu_FX_soft, galaxy_mag_r, min_cluster_mass where
        @min_cluster_mass :: min mass for halo to be called a cluster
    @redshift_limit :: upper limit on redshift
    """
    pos_cen0, pos_sat0 = getHostSatHalos(hd_halo0, cluster_params, redshift_limit)
    pos_cen1, pos_sat1 = getHostSatHalos(hd_halo1, cluster_params, redshift_limit)
    pos_cen2, pos_sat2 = getHostSatHalos(hd_halo2, cluster_params, redshift_limit)
    
    
    # concatenates the ra, dec for central halos
    ra_cen = concatenateObjs(pos_cen0[0], pos_cen1[0], pos_cen2[0])
    dec_cen = concatenateObjs(pos_cen0[1], pos_cen1[1], pos_cen2[1])
    
    # concatenates the ra, dec for satellite halos
    ra_sat = concatenateObjs(pos_sat0[0], pos_sat1[0], pos_sat2[0])    
    dec_sat = concatenateObjs(pos_sat0[1], pos_sat1[1], pos_sat2[1])  
    return ra_cen, dec_cen, ra_sat, dec_sat