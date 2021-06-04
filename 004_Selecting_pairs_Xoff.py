"""
004. Select pairs for the chosen Xoff/Rvir deciles

Here the pairs are computed using the query ball tree cKDTree algorithm. The chosen pairs have mass ratio of 0.33<m1/m2<3 and redshift difference of < 0.001. This ensures that if the pairs underwent a merger, it must be a major merger. The script selects the pairs that further pass the criteria that at least one of the components of the pairs have an Xoff/Rvir value within the chosen decile.

1. Loading data and defining input parameters
2. Parameters used for creating the Major Merger catalogue
3. Open files and get relevant data
4. Studying the effect of 𝑋̃_off on MM pairs

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

# pixel numbers chosen for computation from the all-sky simulation file
number = np.arange(0, 10)
pixel_no_arr = ['00000'+ str(n) for n in number]

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

"""
2. Parameters used for creating the Major Merger catalogue
"""
# get shell volume and projected radius bins [Mpc]
r_p, shell_volume = aimm.shellVolume()

# keywords can be: 'mm and dv' or 'all' 
keyword = 'mm and dv'

# arr to save counts for every pixel, for every radius-bin
count_pairs_all_r = np.zeros( (0, len(r_p) ) )

# decile index
decile_idx = 3


"""
3. Open files and get relevant data
"""
# iterate over the pixels of the simulation
for pixel_no in pixel_no_arr:
    _, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'halo']))

    # Extracting positions and redshifts of the halos
    _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
    hd_z_halo = hd_halo[conditions_halo]
    
    xoff_all = hd_z_halo['HALO_Xoff']/hd_z_halo['HALO_Rvir']
    xoff_deciles = cswl.generateDeciles(xoff_all)
    xoff_bins = xoff_deciles[decile_idx]
    
    print("Pixel: %s, Halos: %d, xoff decile: %.2f - %.2f"%( pixel_no, len(hd_z_halo), xoff_bins[0], xoff_bins[1] ))

    """
    3.1 Studying the effect of 𝑋̃_off on MM pairs
    """
    
    # load the pair indicies
    pairs_idx = cswl.openPairsFiles(pixel_no=pixel_no)  
    
    # go over every radius bin to choose pairs within the decile bin
    count_pairs_x_off_arr = []
    for i, r in enumerate(r_p):        
        
        # select pairs
        count_pairs_x_off = cswl.selectParameterPairs(pairs_idx, i, xoff_all, param=xoff_bins)
        count_pairs_x_off_arr.append(count_pairs_x_off)
        
        print('\n---- radius %.2f Mpc: %d ---'%(r, count_pairs_x_off))

    # save this for every pixel
    count_pairs_all_r = np.append(count_pairs_all_r, [count_pairs_x_off_arr], axis=0)
    
# saves the counts (for all chosen pixels and all radii)
np.save('Data/pairs_z%.1f/Major_dv_pairs/num_pairs_pixel%s-%s_xoff_decile%d.npy'%(redshift_limit, pixel_no_arr[0], pixel_no_arr[-1], decile_idx)  , count_pairs_all_r, allow_pickle=True)