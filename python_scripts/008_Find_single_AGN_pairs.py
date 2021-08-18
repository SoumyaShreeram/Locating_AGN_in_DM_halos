"""
008. Generated AGN halo pairs

This script counts single AGN pairs. This is calculated with the AGN catalogue that
accounts for close pairs, and also for the one that doesn't.

1. Defining input parameters
2. Parameters for creating the Major Merger catalogue
3. Find pairs for chosen pixels
    3.1. Iteration: finding pairs for every radius bin
    
    
Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 22nd June 2021
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
import glob

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('../imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import plotting_cswl05 as pt
import All_sky as sky
import Scaling_relations as sr

"""
1. Defining input parameters
"""


# pixel number from the simulation file
ll, ul = int(sys.argv[1]), int(sys.argv[2])
pixel_no_cont_arr = sky.allPixelNames()
pixel_no_arr = pixel_no_cont_arr[ll:ul]

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

# look back into redshifts until...
redshift_limit = 2

data_dir = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS',\
         'fixedAmp_InvPhase_001')

"""
2. Parameters for creating the Major Merger catalogue

These criteria were also applied when building the model. 
So for fair compairson the validation is also applied with the same criteria.

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

cp_agn_cat = False

# fraction of close pair agns added to the cat_AGN_all
frac_cp_agn = float(sys.argv[3])
if frac_cp_agn == 0.15:
    frac_str = '%.2f'%frac_cp_agn
else:
    frac_str = '%.1f'%frac_cp_agn

tmm_min , tmm_max = float(sys.argv[4]), float(sys.argv[5])
xoff_min, xoff_max = float(sys.argv[6]), float(sys.argv[7])
param_names = '_%.1f_tmm_%.1f_%.2f_xoff_%.2f'%(tmm_min, tmm_max ,xoff_min, xoff_max)
    
"""
3. Find pairs for chosen pixels
"""

count_pairs_arr2D = np.zeros((0, len(r_p)))

# iterate over the pixels of the simulation
for pixel_no in pixel_no_arr:
    all_mm_dv_idx_all = []
    count_pairs_arr = []
    # open halo file
    hd_agn, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'agn', 'halo']))

    # extracting the conditions for choosing halos
    _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
    hd_halo_z = hd_halo[conditions_halo]
    print("Halos: %d"%(len(hd_halo_z) ))

    if cp_agn_cat:
        # load the close pairs catAGN
        catAGN_dir_name = 'CP_10_sigma_1.0_frac_' + frac_str + '_%.1f_tmm_*_*_xoff_*'%tmm_min
        catAGN_dir = glob.glob(os.path.join(data_dir, catAGN_dir_name ))[0]
        catAGN_dir_file = os.path.join(data_dir, catAGN_dir, pixel_no+'.fit')
        hd_cp_agn = Table.read(catAGN_dir_file, format='fits')
        
        hd_agn_z = hd_cp_agn[hd_cp_agn['redshift_R']< redshift_limit]

    else:
        hd_agn_z = hd_agn[hd_agn['redshift_R']<redshift_limit]
    
    """
    3.1 Iteration: finding pairs for every radius bin
    """
    
    for r in range(len(r_p)):      
        print('\n ---- pairs within radius %.3f Mpc ---'%r_p[r])
        pos_spherical_agn = aimm.getSphericalCoord(hd_agn_z)
        pos_spherical_halo = aimm.getSphericalCoord(hd_halo_z)

        # create tree
        tree_agn = cKDTree(np.transpose(np.abs(pos_spherical_agn)), leafsize=1000.0)
        tree_halo = cKDTree(np.transpose(np.abs(pos_spherical_halo)), leafsize=1000.0)

        # list of lists of all pair indicies per DM halo
        all_idx = tree_agn.query_ball_tree(tree_halo, r=r_p[r], p=2) 
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

        pairs_selected = cswl.tuplePairArr(np.array(all_mm_dv_idx))
        count_pairs_arr.append(len(pairs_selected))
    count_pairs_arr2D = np.append(count_pairs_arr2D, [count_pairs_arr], axis=0)

local_data_dir = '../Data/pairs_z%.1f/cat_AGN_halo_pairs/'%redshift_limit
if cp_agn_cat:
    np.save(local_data_dir+'np_%s-%s_fracAGN%.2f'%(pixel_no_arr[0], pixel_no_arr[-1], frac_cp_agn)+param_names, count_pairs_arr2D, allow_pickle=True)   
else:
    np.save(local_data_dir+'cat_without_CP/np_pixels_%s-%s.npy'%(pixel_no_arr[0], pixel_no_arr[-1]), count_pairs_arr2D, allow_pickle=True)
