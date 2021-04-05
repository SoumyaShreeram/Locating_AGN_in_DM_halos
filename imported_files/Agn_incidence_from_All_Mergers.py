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

# personal imports
import Agn_incidence_from_Major_Mergers as aimm

"""
Function to create MM catalog and to count pairs
"""

def createMergerCatalog(hd_obj, obj_conditions, cosmo, time_since_merger=1):
    """
    Function to create Major Merger (MM) catalog
    @hd_obj :: header file for the object of interest 
    @obj_conditions :: prior conditions to define the object sample
    @cosmo :: cosmology used in the notebook (Flat Lambda CDM)
    @mass_range :: [lower, upper] limits on range on galaxy stellar masses to create pair sample
    @time_since_merger :: int to decide the objects with mergers < x Gyr
    """
    # converting the time since merger into scale factor
    merger_z = z_at_value(cosmo.lookback_time, time_since_merger*u.Gyr)
    merger_scale = 1/(1+merger_z)
    
    # defining the merger condition
    merger_condition = (hd_obj['HALO_scale_of_last_MM']>merger_scale)
    
    downsample = obj_conditions & merger_condition
    return hd_obj[downsample], downsample


def getNumberDensityOfPairs(hd_mm_all):
    """
    Function to get the number density of pairs found for the range of projected radii for different mass bins
    """
    # get shell volume and projected radius bins
    r_p, _, shell_volume = aimm.shellVolume()

    # define empty array to same number of pairs detected as a function of radius
    num_pairs_all = []

    num_pairs = aimm.findPairs(hd_mm_all, leafsize=1000.0)
    num_pairs_all.append(np.array(num_pairs))
    return num_pairs_all, r_p, shell_volume

def studyTimeSinceMergerEffects(hd_obj, conditions_obj, cosmo, dt_m_arr):
    """
    Function to study the effect of time since merger of counting MM pairs
    """
    # get shell volume and projected radius bins
    r_p, _, _ = aimm.shellVolume()
    
    num_pairs_obj_dt_m, num_pairs_obj__mass_dt_m = [], []
    all_pairs = np.zeros((0, len(r_p[1:]) ))
    
    for dt_m in dt_m_arr:
        hd_mm_all, _ = createMergerCatalog(hd_obj, conditions_obj, cosmo, dt_m)

        # get pairs for range of different time since merger samples
        all_pairs_dt, _, _ = getNumberDensityOfPairs(hd_mm_all)
    
        # get rid of 0s
        all_pairs = np.append(all_pairs, [all_pairs_dt[0]], axis=0)        
    return all_pairs