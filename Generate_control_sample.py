"""
Script to calculate the control sample for pairs 

The script calculates the redshift and mass matched control sample, which comprises of paired or unpaired halos. The script is divided into the following sections:
1. Defining input parameters
2. Open files and get relevant data

Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 5th May 2021
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
import importlib as ib

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl

"""
1. Defining input parameters
"""
# look back into redshifts until...
redshift_limit = 2

# pixel number from the simulation file
pixel_no = '000000'

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

# get shell volume and projected radius bins [Mpc]
r_p, shell_volume = aimm.shellVolume()

# time since merger array [Gyr]
dt_m_arr = [0, 0.5, 1, 2, 3, 4]
key = 'tmm_pairs'  # 'tmm_self_pairs', 'self_pairs', 'pairs'


"""
2. Open files and get relevant data
"""
_, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'halo']))

# Extracting positions and redshifts of the halos
_, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
hd_z_halo = hd_halo[conditions_halo]

print("Number of halos: %d"%(len(hd_z_halo) ))

"""
3. Generate control samples
"""
if key == 'tmm_pairs':
    # load all the priorly calculated tmm, generate the bins
    diff_t_mm_arr = np.load('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), allow_pickle=True)
    dt_m_bins_arr = cswl.decideBins(dt_m_arr, np.max(diff_t_mm_arr))
    
    # load pairs for the chosen bin
    dt_m_bins = dt_m_bins_arr[0]
    pairs_all = cswl.openPairsFiles(key = 'all')
    pairs_selected = cswl.openPairsFiles(key = 'mm and dv and tmm')
    
    # generate control sample for every case of radii
    for r in [10, 11, 12, 13, 14, 15]:
        print('-- Control for MM pairs with r_p = %.3f Mpc --'%r_p[r])
        cswl.getMZmatchedPairs(hd_z_halo, pairs_all, pairs_selected, r=r, dt_m_bins=dt_m_bins, key=key)

        
        
if key == 'tmm_self_pairs':
    # load all the priorly calculated tmm, generate the bins
    diff_t_mm_arr = np.load('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), allow_pickle=True)
    dt_m_bins_arr = cswl.decideBins(dt_m_arr, np.max(diff_t_mm_arr))
    
    # load pairs for the chosen bin
    dt_m_bins = dt_m_bins_arr[0]
    pairs_selected = cswl.openPairsFiles(key = 'mm and dv and tmm')
    
    # generate control sample for every case of radii
    for r in [0]:
        print('-- Control for MM pairs with r_p = %.3f Mpc --'%r_p[r])
        cswl.getMZmatchedPairs(hd_z_halo, pairs_selected, pairs_selected, r=r, dt_m_bins=dt_m_bins, key=key)
        

        
if key == 'self_pairs':
    pairs_mm_all = cswl.openPairsFiles(key = 'mm and dv')

    for r in [0, 1, 2, 3]:
        print('-- Control for MM pairs with r_p = %.3f Mpc --'%r_p[r])
        cswl.getMZmatchedPairs(hd_z_halo, pairs_mm_all, pairs_mm_all, r=r, key=key)

        
        
if key == 'pairs':
    pairs_all = cswl.openPairsFiles(key = 'all')
    pairs_mm_all = cswl.openPairsFiles(key = 'mm and dv')

    for r in [0, 1, 2, 3]:
        print('-- Control for MM pairs with r_p = %.3f Mpc --'%r_p[r])
        cswl.getMZmatchedPairs(hd_z_halo, pairs_all, pairs_mm_all, r=r, key=key)


