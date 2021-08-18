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

from astropy.table import Table, Column, join
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

frac_cp = float(sys.argv[3])
model_no = int(sys.argv[4])

"""
1. Defining input parameters
"""

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

# look back into redshifts until...
redshift_limit = 2

# arr to save the changed fluxs
frac_r500c_arr = [0, .2, 0.5, 1]
    

"""
2. Set directory path and decides the AGN model to used
"""
# set data dir path
data_dir = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS',\
         'fixedAmp_InvPhase_001')
string = '%.1f'%frac_cp

# arr to get the different cp catAGN directories
list_model_names = np.array(glob.glob(os.path.join(data_dir, 'CP_10_sigma_1.0_frac_'+string+'_*_tmm_*_xoff_*')))
model_dir = list_model_names[model_no]
model_name = 'Model_A%d'%(model_no)
print(model_dir, model_name)

cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
use_rand_agn = False

"""
3. Changes the Lx if agn in frac*R500c
"""
counts_cp_agn_arr, counts_agn_arr = np.zeros((0, len(frac_r500c_arr[1:]))), np.zeros((0, len(frac_r500c_arr[1:])))
for pixel_no in pixel_no_arr:
    # get the catAGN_all header
    hd_agn, _, _ = edh.getHeaders(pixel_no, np.array(['agn']))
    hd_agn = hd_agn[hd_agn['redshift_R']<redshift_limit]

    # load the close pairs catAGN
    catAGN = os.path.join(model_dir, pixel_no+'.fit')
    hd_cp_agn = Table.read(catAGN, format='fits')
    
    # load the cluster file for the same pixel
    hd_clu_params = sr.getCluParams(pixel_no)

    # SkyCoord object that is inputted into astropy's search_around_sky
    agn_cp_coord = SkyCoord(ra=hd_cp_agn['RA'], dec=hd_cp_agn['DEC'], frame='icrs', unit=(u.deg, u.deg))
    cluster_coord = SkyCoord(ra=hd_clu_params['RA'], dec=hd_clu_params['DEC'], frame='icrs', unit=(u.deg, u.deg))
    
    # maximum radius to search around the clusters in the given pixel
    max_r500c = np.max(hd_clu_params['R500c_arcmin'])*u.arcmin.to(u.deg)

    # outputs from the search_around_sky gives the info to mask according to r500c distances 
    idx_cluster_cp, idx_agn_cp, d2d_cp, d3d_cp  = agn_cp_coord.search_around_sky(cluster_coord, max_r500c*u.deg)
    
    if use_rand_agn:
        agn_coord = SkyCoord(ra=hd_agn['RA'], dec=hd_agn['DEC'], frame='icrs', unit=(u.deg, u.deg))
        idx_cluster, idx_agn, d2d, d3d  = agn_coord.search_around_sky(cluster_coord, max_r500c*u.deg)
    
        scaled_LX_soft_RF_agn_all = np.zeros((len(frac_r500c_arr)-1, len(hd_clu_params)))
        c_agn = []
    
    scaled_LX_soft_RF_cp_agn_all = np.zeros((len(frac_r500c_arr)-1, len(hd_clu_params)))
    c_cp_agn = []

    for f, frac_r500c in enumerate(frac_r500c_arr[1:]):
        if use_rand_agn:
            # results that do not account for close pairs
            scaled_LX_soft_RF_agn, count_changes = sr.checkIfAGNfluxinR500c(pixel_no, frac_r500c,\
                                                                        hd_agn, idx_cluster,\
                                                                        idx_agn, d2d)
            scaled_LX_soft_RF_agn_all[f, :] = scaled_LX_soft_RF_agn
            c_agn.append(count_changes)

        # results that account for close pairs
        scaled_LX_soft_RF_agn_cp, count_changes_cp = sr.checkIfAGNfluxinR500c(pixel_no, frac_r500c,\
                                                                          hd_cp_agn,\
                                                                          idx_cluster_cp,\
                                                                          idx_agn_cp, d2d_cp)
        
        scaled_LX_soft_RF_cp_agn_all[f, :] = scaled_LX_soft_RF_agn_cp
        c_cp_agn.append(count_changes_cp)

    # saveing the info into an astropy table
    if use_rand_agn:
        sr.write2Table(frac_r500c_arr, scaled_LX_soft_RF_agn_all, model_name, pixel_no=pixel_no)
        counts_agn_arr = np.append(counts_agn_arr, [c_agn], axis=0)
    
    sr.write2Table(frac_r500c_arr, scaled_LX_soft_RF_cp_agn_all, model_name, pixel_no=pixel_no, frac_cp=frac_cp)
    counts_cp_agn_arr = np.append(counts_cp_agn_arr, [c_cp_agn], axis=0)

"""
4. Decided where to save files that keep track of the counts of changed cluster Lx
"""       
changed_Lx_dir = '../Data/pairs_z%.1f/Scaling_relations/'%(redshift_limit)
chhanged_Lx_filename = 'Changed_Lx_%s_px_%s.npy'%(pixel_no_arr[0], pixel_no_arr[-1])

if use_rand_agn:
    np.save(changed_Lx_dir+chhanged_Lx_filename, counts_agn_arr, allow_pickle=True)

np.save(changed_Lx_dir+model_name+'/frac_cp_%.2f'%frac_cp+chhanged_Lx_filename, counts_cp_agn_arr, allow_pickle=True)