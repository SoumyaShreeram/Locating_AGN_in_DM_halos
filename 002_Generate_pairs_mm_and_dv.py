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
number = np.arange(131, 140)
pixel_no_arr = ['000'+ str(n) for n in number]

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

"""
2. Parameters for creating the Major Merger catalogue
"""
# get shell volume and projected radius bins [Mpc]
r_p, shell_volume = aimm.shellVolume()

dt_m_arr = np.load('Data/pairs_z%.1f/t_mm_deciles.npy'%redshift_limit, allow_pickle=True)
dt_m_bins_arr = [[dt_m_arr[i], dt_m_arr[i+1]] for i in np.arange(len(dt_m_arr)-1)]

# max mass ratio to classify as a major merger [dimensionless]
mass_max = 3

# defining the redshift bin for a merger in terms of dv = c*z [km/s]
dz_cut =  0.001

# BOOLEAN: if the pairs have already been computed before
run_find_pairs = True

run_merger_pairs = False
save_merger_indicies = False

# keywords can be: 'mm and dv', 'dv' or 'all' 
keyword = 'mm and dv'
major_mergers_only, delta_v_cut = cswl.decideBools(keyword = keyword)

# if you wish to apply tmm and xoff selection cuts
apply_selection = False
if apply_selection:
    new_key = 'selection'
    xoff_min, xoff_max= 0.17, 0.54
    tmm_min, tmm_max= 0.6, 1.2
else:
    new_key = keyword
    
# do you want to store the indicies or just the counts?
save_indicies, save_counts = True, True
"""
3. Open files and get relevant data
"""
count_pairs_arr2D = np.zeros((0, len(r_p)))

# iterate over the pixels of the simulation
for pixel_no in pixel_no_arr:
    all_mm_dv_idx_all = []
    count_pairs_arr = []

    # open halo file
    _, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'halo']))

    # extracting the conditions for choosing halos
    _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
    hd_halo_z = hd_halo[conditions_halo]
    print("Halos: %d"%(len(hd_halo_z) ))

    if apply_selection:
        # after tmm and xoff conditions
        diff_t_mm_arr = np.load('Data/pairs_z%.1f/t_mm/pixel_%s.npy'%(redshift_limit, pixel_no), allow_pickle=True)
        diff_t_mm_arr = np.abs(diff_t_mm_arr)
        
        # applying the cuts
        total_conditions = cswl.selectionHalos(hd_halo_z, diff_t_mm_arr, xoff_min=xoff_min, xoff_max=xoff_max, tmm_min=tmm_min, tmm_max=tmm_max)
        hd_halo_z = hd_halo_z[total_conditions]
        print("AGNs: %d"%(len(hd_halo_z)) )


    """
    3. Finding halo pairs
    """
    if run_find_pairs:
        for r in range(len(r_p)):
            print('\n ---- pairs within radius %.3f Mpc ---'%r_p[r])
            pos_spherical = aimm.getSphericalCoord(hd_halo_z)

            # create tree
            tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=1000.0)

            # list of lists of all pair indicies per DM halo
            all_idx = tree_data.query_ball_tree(tree_data, r=r_p[r], p=2) 
            count_pairs = cswl.countSelectedPairs(all_idx, string = 'All pairs: ') 

            if delta_v_cut:            
                # (1) choose  pairs that satisfy the delta v criteria
                all_dv_idx, count_pairs = cswl.deltaVelSelection(hd_halo_z, all_idx, dz_cut=dz_cut)

            if major_mergers_only and delta_v_cut:
                # (2) choose major pairs (major mergers) with the delta v criteria
                all_mm_dv_idx, count_pairs = cswl.majorMergerSelection(hd_halo_z, all_dv_idx)

            if major_mergers_only and not delta_v_cut:
                # (3) choose only major pairs (major mergers)
                all_mm_idx, count_pairs = cswl.majorMergerSelection(hd_halo_z, all_idx, keyword=keyword)
            
            if save_indicies:
                # save file based on the criteria applied
                pairs_selected = cswl.tuplePairArr(np.array(all_mm_dv_idx))
                all_mm_dv_idx_all.append(pairs_selected)           
            
            if save_counts:
                count_pairs_arr.append(count_pairs)
        
        # decides wether to save the counts of pairs or/and the indicies
        if save_indicies:
            cswl.saveSeparationIndicies(all_mm_dv_idx_all, r_p[r], keyword=new_key, redshift_limit=redshift_limit, pixel_no=pixel_no)
    
    
        if save_counts:
            count_pairs_arr2D = np.append(count_pairs_arr2D, [count_pairs_arr], axis=0)

    
np.save('Data/pairs_z%.1f/Major_dv_pairs/Selection_applied/num_pairs_pixels_%s-%s.npy'%(redshift_limit, pixel_no_arr[0], pixel_no_arr[-1]), count_pairs_arr2D, allow_pickle=True)

"""
4. Studying the effect of Δ𝑡_merger on MM pairs

Now that all the pairs for the chosen cases of time since major mergers are computed, we can proceed to calculate the fraction of halo pairs for each case.

𝑓_halo_pairs = NP / N(N−1)×Shell volume

where 𝑁𝑃 is the number of pairs and 𝑁 is the total number of objects from which pairs are chosen.
"""

if run_merger_pairs:
    pairs_all = cswl.openPairsFiles(data_dir='Data/pairs_z%.1f/'%redshift_limit, key = keyword, dz_cut= dz_cut)
    
    diff_t_mm_arr = np.load('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), allow_pickle=True)
    
    for i, dt_m in enumerate(dt_m_bins_arr):
        count_t_mm_arr = []
        dt_m_bins =  dt_m_bins_arr[i]
        
        # get index number and deicde the lower- upper- limit of Tmm
                
        for r in range(len(r_p)): 
            print('\n ---- Merger pairs within radius %.2f Mpc, %.1f - %.1f Gyr ---'%(r_p[r], dt_m_bins[0], dt_m_bins[1]))
        
            all_t_mm_idx, count_t_mm = cswl.selectParameterPairs(hd_halo_z, pairs_all[0][r], cosmo, diff_t_mm_arr, param = dt_m_bins, redshift_limit = redshift_limit)
            count_t_mm_arr.append(count_t_mm)
        
            if save_merger_indicies and keyword == 'mm and dv':
                np.save('Data/pairs_z%.1f/Major_dv_pairs/Tmm_%.2f-%.2fGyr/pairs_idx_r%.3f_mm%d_dz%.3f.npy'%(redshift_limit, dt_m_bins[0], dt_m_bins[1], r_p[r], mass_max, dz_cut), all_t_mm_idx, allow_pickle=True)
                print('\n --- Saved mm and dv file --- ')
        
        # saves the counts (of halo pairs for a given bin, for all radius)
        cswl.saveTmmFiles(keyword, dt_m_bins, arr = count_t_mm_arr, redshift_limit = redshift_limit)