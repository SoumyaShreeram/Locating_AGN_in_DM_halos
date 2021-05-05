"""
05. Preliminary comparison of the 𝑓𝑀𝑀 between simulation and data

The notebook is similar to the notebook 02, which builds a major merger catalog. However, here things are done slightly differently: (1) halo pairs are generated, (2) the criteria are applied, rather than the other way around (as shown in notebook 02).

1. Loading data and defining input parameters
2. Finding pairs and creating a major/minor sample
2. Studying merger fraction 𝑓𝑀𝑀 as a function of redshift

Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 23rd April 2021
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

# system imports
import os
import sys
import importlib as ib

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import plotting_cswl05 as pt

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

"""
2. Parameters for creating the Major Merger catalogue
"""
# get shell volume and projected radius bins [Mpc]
r_p, dr_p, shell_volume = aimm.shellVolume()

# time since merger [Gyr]
time_since_merger = 5

# time since merger array [Gyr]
dt_m_arr = [0.5, 1, 2, 3, 4]

# max mass ratio to classify as a major merger [dimensionless]
mass_max = 3

# defining the redshift bin for a merger in terms of dv = c*z [km/s]
dv_cut =  1000

# BOOLEAN: if the pairs have already been computed before
run_find_pairs = False
run_merger_pairs = True

# BOOLEAN: if we want to find all pairs (not just major pairs)
major_mergers_only = False
delta_v_cut = True
keyword = 'dv' # 'mm and dv' or 'all'
"""
3. Open files and get relevant data
"""
_, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'halo']))

# Extracting positions and redshifts of the halos
_, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
hd_halo_z = hd_halo[conditions_halo]

print("Halos: %d"%(len(hd_halo_z) ))

"""
3. Finding halo pairs
"""
if run_find_pairs:
    for r in [0, 1, 2, 3, 4]:
        print('\n ---- pairs within radius %.2f Mpc ---'%r_p[r])
        pos_spherical = aimm.getSphericalCoord(hd_halo_z)

        # create tree
        tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=1000.0)

        # list of lists of all pair indicies per DM halo
        all_mm_idx = tree_data.query_ball_tree(tree_data, r=r_p[r], p=2) 
        count_pairs, _ = cswl.countPairs(all_mm_idx) 
        
        if major_mergers_only:
            # (1) choose only major pairs (major mergers)
            all_mm_idx, count_major_pairs = cswl.majorMergerSelection(hd_halo[conditions_halo], all_mm_idx)
           
        if delta_v_cut:            
            # (2) choose only major pairs that satisfy the delta v criteria
            all_dv_idx, count_dv_major_pairs = cswl.deltaVelSelection(hd_halo_z, all_mm_idx, mm = major_mergers_only, dv_cut=dv_cut)
        
        # save file based on the criteria applied        
        if keyword == 'mm and dv':
            np.save('Data/pairs_z%.1f/Major_pairs/pairs_idx_r%.1f_mm%d_dv%d.npy'%(redshift_limit, r, mass_max, dv_cut), all_dv_idx, allow_pickle=True)
            print('\n --- Saved mm file --- ')
            
        if keyword == 'dv':
            np.save('Data/pairs_z%.1f/dv_pairs/pairs_idx_r%.1f_dv%d.npy'%(redshift_limit, r, dv_cut), all_dv_idx, allow_pickle=True)
            print('\n --- Saved dv file --- ')
            
        # if you want to save all the pairs
        if keyword == 'all':
            np.save('Data/pairs_z%.1f/pairs_idx_r%d.npy'%(redshift_limit, r), all_mm_idx, allow_pickle=True)
            print('\n --- Saved no cuts file --- ')
            
"""
4. Studying the effect of Δ𝑡_merger on MM pairs

Now that all the pairs for the chosen cases of time since major mergers are computed, we can proceed to calculate the fraction of halo pairs for each case.

𝑓_halo_pairs = NP / N(N−1)×Shell volume

where 𝑁𝑃 is the number of pairs and 𝑁 is the total number of objects from which pairs are chosen.
"""

if run_merger_pairs:
    pairs_all = cswl.openPairsFiles(data_dir='Data/pairs_z%.1f/'%redshift_limit, key = keyword, dv_cut= dv_cut)
    
    diff_t_mm_arr = np.load('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), allow_pickle=True)
    
    for dt_m in dt_m_arr[1:2]:
        count_t_mm_arr = []

        for r in range(len(r_p)): 
            print('\n ---- pairs within radius %.2f Mpc ---'%r_p[r])
        
            _, count_t_mm = cswl.defineTimeSinceMergeCut2(hd_halo_z, pairs_all[0][r], cosmo, diff_t_mm_arr, time_since_merger = dt_m, redshift_limit = redshift_limit)
            count_t_mm_arr.append(count_t_mm)
          
        cswl.saveTmmFiles(keyword, dt_m, arr = count_t_mm_arr, redshift_limit = redshift_limit)