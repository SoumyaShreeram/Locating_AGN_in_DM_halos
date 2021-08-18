"""
003. Select pairs for the chosen tmm deciles

Here the pairs are computed using the query ball tree cKDTree algorithm. The chosen pairs have mass ratio of 0.33<m1/m2<3 and redshift difference of < 0.001. This ensures that if the pairs underwent a merger, it must be a major merger. The script selects the pairs that further pass the criteria that at least one of the components of the pairs have a tmm value within the chosen decile.

1. Loading data and defining input parameters
2. Parameters used for creating the Major Merger catalogue
3. Studying the effect of tmm on MM pairs

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
sys.path.append('../imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import plotting_cswl05 as pt
import All_sky as sky

"""
1. Defining input parameters
"""
# look back into redshifts until...
redshift_limit = 1

# pixel number from the simulation file
ll, ul = int(sys.argv[1]), int(sys.argv[2])
pixel_no_cont_arr = sky.allPixelNames()
pixel_no_arr = pixel_no_cont_arr[ll:ul]

# define the cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777


"""
2. Parameters used for creating the Major Merger catalogue
"""

# get shell volume and projected radius bins [Mpc]
r_p, _ = aimm.shellVolume()

# keywords can be: 'mm and dv' or 'all' 
keyword = 'mm and dv'

# arr to save counts for every pixel, for every radius-bin
count_pairs_all_r = np.zeros( (0, len(r_p) ) )

# decile index
decile_idx = int(sys.argv[3])

"""
3. Studying the effect of tmm on MM pairs
"""
# iterate over the pixels of the simulation
for pixel_no in pixel_no_arr:
    
    # get the deciles for the given pixel
    t_mm_arr = np.load('../Data/pairs_z%.1f/t_mm/pixel_%s.npy'%(redshift_limit, pixel_no), allow_pickle=True)
    tmm_deciles = cswl.generateDeciles(t_mm_arr[0])
    t_mm_bins = tmm_deciles[decile_idx]
    
    print('Pixel: %s, tmm decile: %.1f-%.1f Gyr'%(pixel_no, t_mm_bins[0], t_mm_bins[1]))
    
    # load the pair indicies
    pairs_idx = cswl.openPairsFiles(pixel_no=pixel_no, redshift_limit = redshift_limit)  
    
    # go over every radius bin to choose pairs within the decile bin
    count_pairs_t_mm_arr = []
    for i, r in enumerate(r_p):        
        
        # select pairs
        count_pairs_t_mm = cswl.selectParameterPairs(pairs_idx, i, t_mm_arr[0], param=t_mm_bins)
        count_pairs_t_mm_arr.append(count_pairs_t_mm)
        
        print('\n---- radius %.2f Mpc: %d ---'%(r, count_pairs_t_mm))

    # save this for every pixel
    count_pairs_all_r = np.append(count_pairs_all_r, [count_pairs_t_mm_arr], axis=0)
    
# saves the counts (for all chosen pixels and all radii)
np.save('../Data/pairs_z%.1f/Major_dv_pairs/num_pairs_pixel%s-%s_tmm_decile%d.npy'%(redshift_limit, pixel_no_arr[0], pixel_no_arr[-1], decile_idx) , count_pairs_all_r, allow_pickle=True)