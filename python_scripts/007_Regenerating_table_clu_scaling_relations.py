"""
007. Generate cluster table
    
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

# look back into redshifts until...
redshift_limit = .2

# pixel number from the simulation file
pixel_no_cont_arr = sky.allPixelNames()

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

# get all sky files for both agn and clu?
generate_cluster_file = False
generate_agn_file = True

"""
2. Constructing cluster table
"""
if generate_cluster_file:
    # initializing the table
    clu_params = sr.getCluParams(pixel_no=pixel_no_cont_arr[0])

    for pixel_no in pixel_no_cont_arr[1:]:
        print(pixel_no)
        # get the new cluster table for chosen params
        clu_params_new = sr.getCluParams(pixel_no)
        
        # joining the tables
        clu_params = join(clu_params, clu_params_new, join_type='outer')

    # once completed for the full sky, save the table
    clu_params.write('../Data/pairs_z%.1f/Scaling_relations/clu_scaling_relations_all_sky.fit'%(redshift_limit), format='fits') 
    print('saved file')

"""
2. Constructing AGN table
"""
if generate_agn_file:
    agn_params = sr.getAGNparams(pixel_no=pixel_no_cont_arr[0])
    
    for pixel_no in pixel_no_cont_arr[1:]:
        agn_params_new = sr.getAGNparams(pixel_no=pixel_no)

        # joining the tables
        agn_params = join(agn_params, agn_params_new, join_type='outer')

    # once completed for the full sky, save the table
    agn_params.write('../Data/pairs_z%.1f/Scaling_relations/agn_scaling_relations_all_sky.fit'%(redshift_limit), format='fits') 
    print('saved file')




