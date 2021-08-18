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
import pandas as pd

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
frac_cp_agn = float(sys.argv[1]) # 0.2/0.15/0.1/0.03

model_name = str(sys.argv[2]) #'Model_A1'/0/2/3

# make a new table 
hd_clu_params_all = sky.makeClusterFile(redshift_limit=2, model_name=model_name)

# directory to save the information
model_dir = '../Data/pairs_z%.1f/Scaling_relations/%s/'%(redshift_limit, model_name)
fname = model_dir+'CLU_with_scaled_Lx_all_sky_frac_cp_%.2f.fit'%(frac_cp_agn)

# write the new combined cluster file
hd_clu_params_all.write(fname, format='fits')