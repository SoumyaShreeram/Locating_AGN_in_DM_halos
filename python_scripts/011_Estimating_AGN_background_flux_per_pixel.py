"""
010. Concatenates the CP cat AGN
    
Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 1st July 2021
"""

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.io import ascii 
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

# pixel number from the simulation file
pixel_no = '000000'

# pixel number from the simulation file
ll, ul = int(sys.argv[1]), int(sys.argv[2])
pixel_no_cont_arr = sky.allPixelNames()
pixel_no_arr = pixel_no_cont_arr[ll:ul]

# fraction of close pair agns added to the cat_AGN_all
frac_cp_agn = 0.03

# arr to multiply with r500c of the cluster
frac_r500c_arr = [0, .25, 0.5, .75, 1, 1.5]

min_flux_agn = 5e-15

data_dir = '../Data/pairs_z%.1f/Scaling_relations/bkg_agn_flux/'%(redshift_limit)

for pixel_no in pixel_no_arr:
    print(pixel_no)
    # get averate background flux per pixel
    bkg_agn_flux_per_deg2 = sr.getFluxPerDeg2BkgAgn(pixel_no, min_flux_agn=min_flux_agn)
    
    filename = data_dir+'bkg_agn_flux%.E_cpFrac%.2f_px%s.npy'%(min_flux_agn, frac_cp_agn, pixel_no)

    np.save(filename, bkg_agn_flux_per_deg2, allow_pickle=True)
    print('saved file')