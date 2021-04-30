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
def mergerBins(hd_halo, galaxy_SMHMR_mass, mass_ratio=4):
    """
    Function to get the major merger bins
    @hd_halo :: used to get the max mass of stellar mass halo
    @galaxy_SMHMR_mass :: galaxy stellar mass to the log10
    @mass_ratio :: M_1/M_2 to be classified as MM 
    --> (Ellison et. al 2013 say M_1/M_2 == 4)
    
    @Returns :: mass bins array
    """
    # finding the limits on stellar mass of galaxies in halos
    m_min, m_max = galaxy_SMHMR_mass, np.max(hd_halo['galaxy_SMHMR_mass']) 
    mass_range_log = [ m_min, m_max ]
    mass_range = [10**m for m in mass_range_log]
    
    # array of stellar mass bins
    mass_bins = [mass_range[0]]
    input_mass = mass_range[0]
    
    limit_reached = False
    while not limit_reached:
        # major merger == True if mass_ratio = ...
        input_mass = mass_ratio*input_mass
        mass_bins.append(input_mass)
        
        # limit is achieved when
        if input_mass > mass_range[1]:
            limit_reached = True
    return np.log10(mass_bins)

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
    
    downsample = obj_conditions & mass_condition & merger_condition
    return hd_obj[downsample], downsample

def norm(n):
    if n != 0:
        out = n
    else:
        out = 1
    return out

def shellVolume(r_p_min=1e-2, r_p_max=1.5e-1, num_bins=15):
    """
    Function to create projected radius array and to get the shell volume at every increment
    """
    # projected radius of separation r_p (kpc)
    r_p = np.linspace(r_p_min, r_p_max, num=num_bins+1)
    
    # increment in radius (kpc)
    dr_p = [r_p[i+1] - r_p[i] for i in range(len(r_p)-1)]
    
    # shell vol in kpc^3
    shell_volume = 4*np.pi*( ((r_p[1:])*u.pc)**3 - ((r_p[:-1])*u.pc)**3)
    return r_p, dr_p, shell_volume


def getSphericalCoord(hd_halo):
    """
    Function to get (x, y, z) OR (ra, dec, z)  in sperical coordinates  
    """
    pos_spherical = [hd_halo['HALO_x'], hd_halo['HALO_y'], hd_halo['HALO_z']]
    return pos_spherical


def findPairs(hd_obj, leafsize=1000.0):
    """
    Find pairs of objects
    @hd_obj :: Table <object> for an AGN/halo ...
    """
    pos_spherical = getSphericalCoord(hd_obj)
    
    # get shell volume and projected radius bins
    r_p, dr_p, shell_volume = shellVolume()

    # create tree
    tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=leafsize)
    
    # count neighbours
    pairs = tree_data.count_neighbors(tree_data, r=r_p) 
    
    # number of pairs / volume times total number of objects 
    N = norm( len(hd_obj) )
    num_pairs = (pairs[1:]-pairs[:-1])/(N*(N-1)*shell_volume)
    return num_pairs

def findPairIndexes(hd_obj, r_p, leafsize=1000.0):
    """
    Find the indicies of pairs of objects
    @hd_obj :: Table <object> for an AGN/halo ...
    """
    pos_spherical = getSphericalCoord(hd_obj)
    
    # get shell volume and projected radius bins
    r_p, dr_p, shell_volume = shellVolume()

    # create tree
    tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=leafsize)
    
    # list of lists of all neighbours' idxs
    pairs_idx = tree_data.query_ball_tree(tree_data, r=r_p) 
    return pairs_idx


def majorMergerSampleForAllMassBins(hd_obj, conditions_obj, cosmo, time_since_merger, galaxy_SMHMR_mass=9, mass_ratio_for_MM=4):
    """
    Function gets the major merger sample for all mass bins, but for a given 'time since merger'
    """
    # generate mass bins for halos with M_1/M_2 < mass_ratio
    mass_bins = mergerBins(hd_obj, galaxy_SMHMR_mass, mass_ratio=mass_ratio_for_MM)
    
    # defining new lists to save info
    hd_mm_all, num_mm = [], []

    # MM headers for objects ==> (1) given mass range and (2) given time since merger
    for i in range(len(mass_bins)-1):
        hd_mm, mm_cond = createMMcatalog(hd_obj, conditions_obj, cosmo, mass_range=[mass_bins[i], mass_bins[i+1]], time_since_merger=time_since_merger)
        
        # append the headers for each mass bin
        hd_mm_all.append(hd_mm)
        
        # append the number of major mergers found for each mass bin
        num_mm.append(len(hd_mm))
    return hd_mm_all, np.array([num_mm, mass_bins], dtype=object)

def getNumberDensityOfPairs(hd_mm_all):
    """
    Function to get the number density of pairs found for the range of projected radii for different mass bins
    """
    # get shell volume and projected radius bins
    r_p, _, shell_volume = shellVolume()

    # define empty array to same number of pairs detected as a function of radius
    num_pairs_all = []

    for i in range(len(hd_mm_all)):
        num_pairs = findPairs(hd_mm_all[i], leafsize=1000.0)
        num_pairs_all.append(np.array(num_pairs))
    return num_pairs_all, r_p, shell_volume

def studyTimeSinceMergerEffects(hd_obj, conditions_obj, cosmo, dt_m_arr, galaxy_SMHMR_mass=9, mass_ratio_for_MM=4):
    """
    Function to study the effect of time since merger of counting MM pairs
    """
    num_pairs_obj_dt_m, num_pairs_obj__mass_dt_m = [], []
    
    for dt_m in dt_m_arr:
        hd_mm_all, _ = majorMergerSampleForAllMassBins(hd_obj, conditions_obj, cosmo, dt_m, galaxy_SMHMR_mass, mass_ratio_for_MM)

        # get pairs for range of different time since merger samples
        all_pairs_mass, _, _ = getNumberDensityOfPairs(hd_mm_all)
    
        # lose mass bin information
        all_pairs = np.concatenate(all_pairs_mass, axis=0)
        all_pairs = all_pairs[all_pairs != 0]
        
        # append information for the given dt_m (time since merger)
        num_pairs_obj_dt_m.append(all_pairs)
        num_pairs_obj__mass_dt_m.append(all_pairs_mass)
    return num_pairs_obj_dt_m, num_pairs_obj__mass_dt_m