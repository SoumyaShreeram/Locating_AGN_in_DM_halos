"""
007. Generated pairs for the halos, given criteria for major merger, redshift bin, xoff, and tmm

Here the pairs are computed using the query ball tree cKDTree algorithm. These pairs are chosen only if the mass ratio of the objects forming the pairs in 0.33<m1/m2<3 and they have a redshift difference of < 0.001. This ensures that if the pairs underwent a merger, it must be a major merger. The script is divided into the following sections:

1. Defining input parameters
2. Parameters for creating the Major Merger catalogue
3. Find pairs for chosen pixels
    3.1. Iteration: finding pairs for every radius bin
    
    
Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 15th June 2021
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
import All_sky as sky


"""
1. Defining input parameters
"""


# look back into redshifts until...
redshift_limit = 2

# pixel number from the simulation file
ll, ul = int(sys.argv[1]), int(sys.argv[2])
pixel_no_cont_arr = sky.allPixelNames()
pixel_no_arr = pixel_no_cont_arr[ll:ul]

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777



"""
2. Parameters for creating the Major Merger catalogue
"""


# get shell volume and projected radius bins [Mpc]
r_p, shell_volume = aimm.shellVolume()

# max mass ratio to classify as a major merger [dimensionless]
mass_max = 3

# defining the redshift bin for a merger in terms of dv = c*z [km/s]
dz_cut =  0.001

# keywords can be: 'mm and dv', 'dv' or 'all' 
keyword = 'mm and dv'
major_mergers_only, delta_v_cut = cswl.decideBools(keyword = keyword)

# if you wish to apply tmm and xoff selection cuts
apply_selection = True
if apply_selection:
    new_key = 'selection'
    xoff_min, xoff_max= sys.argv[3], sys.argv[4]
    tmm_min, tmm_max= None, None
else:
    new_key = keyword
    
# do you want to store the indicies or just the counts?
save_indicies = False


"""
3. Find pairs for chosen pixels
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
        diff_t_mm_arr = np.load('../Data/pairs_z%.1f/t_mm/pixel_%s.npy'%(redshift_limit, pixel_no), allow_pickle=True)
        diff_t_mm_arr = np.abs(diff_t_mm_arr)

        # applying the cuts
        total_conditions = cswl.selectionHalos(hd_halo_z, diff_t_mm_arr, xoff_min=xoff_min, xoff_max=xoff_max, tmm_min=tmm_min, tmm_max=tmm_max)
        hd_halo_z = hd_halo_z[total_conditions]
        print("AGNs: %d"%(len(hd_halo_z)) )


    """
    3.1 Iteration: finding pairs for every radius bin
    """
    
    
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


        count_pairs_arr.append(count_pairs)

        # decides wether to save the counts of pairs or/and the indicies
        if save_indicies:
            cswl.saveSeparationIndicies(all_mm_dv_idx_all, r_p[r], keyword=new_key, redshift_limit=redshift_limit, pixel_no=pixel_no)


    count_pairs_arr2D = np.append(count_pairs_arr2D, [count_pairs_arr], axis=0)

    
np.save('../Data/pairs_z%.1f/Major_dv_pairs/Selection_applied/num_pairs_pixels_%s-%s_xoff_%.2f-%.2f_tmm_%.1f-%.1f.npy'%(redshift_limit, pixel_no_arr[0], pixel_no_arr[-1], xoff_min, xoff_max, ), count_pairs_arr2D, allow_pickle=True)