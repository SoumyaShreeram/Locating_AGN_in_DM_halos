"""
010. Concatenates the cluster files with affected Lx due to AGN
    
Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 1st July 2021
"""

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

from scipy.stats import norm
from scipy import interpolate

sys.path.append('../imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl
import Scaling_relations as sr
import plotting_sr_agn_clu as pt
import All_sky as sky

# look back into redshifts until...
redshift_limit = 2

# fraction of close pair agns added to the cat_AGN_all
frac_cp_agn = 0.03

model_name = 'Model_A3'

using_cp_catAGN = False

hd_clu_params_all = sky.makeClusterFile(redshift_limit=redshift_limit,\
 model_name=model_name, using_cp_catAGN=using_cp_catAGN)

if using_cp_catAGN:
    fname = '../Data/pairs_z%.1f/CLU_with_scaled_Lx_all_sky_%s.fit'%(redshift_limit, model_name)
else:
    fname = '../Data/pairs_z%.1f/CLU_with_scaled_Lx_all_sky_ModelNone.fit'%(redshift_limit)

hd_clu_params_all.write(fname, format='fits')
