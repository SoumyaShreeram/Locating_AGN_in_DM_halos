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
    mass_range = [ 10**galaxy_SMHMR_mass, 10**m_max ]
    
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

def shellVolume(r_p_min=10, r_p_max=100, num_bins=10):
    """
    Function to create projected radius array and to get the shell volume at every increment
    """
    # projected radius of separation r_p (kpc)
    r_p = np.linspace(r_p_min, r_p_max, num=num_bins+1)
    
    # increment in radius (kpc)
    dr_p = [r_p[i+1] - r_p[i] for i in range(len(r_p)-1)]
    
    # multiply by 10e6 to have the shell vol in kpc
    shell_volume = 4*np.pi*(((r_p[:-1])*u.pc)**2)*(dr_p*u.pc)
    return r_p, dr_p, shell_volume


def getSphericalCoord(hd_halo):
    """
    Function to get (x, y, z) OR (ra, dec, z)  in sperical coordinates  
    """
    pos_spherical = [hd_halo['HALO_x'], hd_halo['HALO_y'], hd_halo['HALO_z']]
    return pos_spherical

    
def findPairs(hd_obj, leafsize=1000.0, h=0.6777):
    """
    Find pairs of objects
    @pos_z_spherical :: [x, y, z] expressed in spherical coord
    @radius_of_search :: radius to produce a count for
    @
    """
    pos_spherical = getSphericalCoord(hd_obj)
    
    # get shell volume and projected radius bins
    r_p, dr_p, shell_volume = shellVolume()

    # create tree
    tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=leafsize)
    
    # count neighbours
    pairs = tree_data.count_neighbors(tree_data, r=r_p*1e-2/h)
    
    # number of pairs as a function of distance
    num_pairs = (pairs[1:]-pairs[:-1])/shell_volume
    return num_pairs

def majorMergerSampleForAllMassBins(hd_obj, conditions_obj, galaxy_SMHMR_mass, mass_ratio_for_MM, cosmo, time_since_merger):
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