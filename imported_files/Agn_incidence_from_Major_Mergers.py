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

"""
Function to create MM catalog and to count pairs
"""

def createMMcatalog(hd_obj, obj_conditions, cosmo, mass_range, time_since_merger=1):
    """
    Function to create Major Merger (MM) catalog
    @hd_obj :: header file for the object of interest 
    @obj_conditions :: prior conditions to define the object sample
    @cosmo :: cosmology used in the notebook (Flat Lambda CDM)
    @mass_range :: [lower, upper] limits on range on galaxy stellar masses to create pair sample
    @time_since_merger :: int to decide the objects with mergers < x Gyr
    """
    # condition for stellar mass
    mass_condition = (hd_obj['galaxy_SMHMR_mass'] > mass_range[0]) & (hd_obj['galaxy_SMHMR_mass'] < mass_range[1])
    
    # converting the time since merger into scale factor
    merger_z = z_at_value(cosmo.lookback_time, time_since_merger*u.Gyr)
    merger_scale = 1/(1+merger_z)
    
    # defining the merger condition
    merger_condition = (hd_obj['HALO_scale_of_last_MM']>merger_scale)
    
    downsample = obj_conditions + mass_condition + merger_condition
    return hd_obj[downsample]

def shellVolume(r_p_min=10, r_p_max=100):
    "Function to get the shell volume"
    bin_r_p = 
    
    x_bin_xi3D_log = bin_xi3D_log[:-1]+ log_dR/2.
    x_bin_xi3D = 10**x_bin_xi3D_log
    
    shell_volume = (bin_xi3D[1:]**3. - bin_xi3D[:-1]**3.)*4*np.pi/3.
    return shell_volume, bin_xi3D, x_bin_xi3D


def convertToSphericalCoord(pos_z, f_z_to_comoving_dist):
    """
    Function to convert (x, y, z) == (ra, dec, z)  into sperical coordinates  
    """    
    ra, dec, redshift = pos_z[0], pos_z[1], pos_z[2]
    
    # defining sperical coordinates
    dC = f_z_to_comoving_dist(redshift) # Mpc
    theta = ra*np.pi/180.
    phi   = dec*np.pi/180.
    
    # new x, y, z coordinates (Mpc)
    x = dC *np.sin(theta)*np.cos(phi)
    y = dC *np.sin(theta)*np.sin(phi)
    z = dC *np.cos(theta)
    return [x, y, z]

    
def findPairs(pos_z_spherical, bin_xi3D, shell_volume, leafsize=1000.0):
    """
    Find pairs of objects
    @pos_z_spherical :: [x, y, z] expressed in spherical coord
    @radius_of_search :: radius to produce a count for
    @
    """
    # create tree
    tree_data = cKDTree(np.transpose(pos_z_spherical), leafsize=leafsize)
    
    # count neighbours
    pairs = tree_data.count_neighbors(tree_data, r=bin_xi3D)    
    
    # ??
    number_pairs = (pairs[1:] - pairs[:-1])/shell_volume
    return number_pairs