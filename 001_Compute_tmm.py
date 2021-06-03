"""
Quick script to calculate t𝑀𝑀 for all the chosen halos

This python file is used to calculate the time since last major merger for all the halos, t𝑀𝑀, given a prior redshift condition.

1. Loading data and defining input parameters
2. Open files and get relevant data
3. Computing t𝑀𝑀 for a given redshift limit

Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 29th April 2021
"""

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# system imports
import os
import sys

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('imported_files/')
import Exploring_DM_Halos as edh
import Agn_incidence_from_Major_Mergers as aimm
import Comparison_simulation_with_literature_data as cswl

"""
1. Defining input parameters
"""
# look back into redshifts until...
redshift_limit = 2

# pixel number from the simulation file
number = np.arange(700, 768)
pixel_no_arr = ['000'+ str(n) for n in number]

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777


"""
2. Open files and get relevant data
"""
for pixel_no in pixel_no_arr:
    print('\nComputing tmm for pixel no: %s (out of %s)'%(pixel_no, pixel_no_arr[-1]) )
    _, hd_halo, _ = edh.getHeaders(pixel_no, np.array(['halo']))

    # Extracting positions and redshifts of the halos
    _, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)

    hd_z_halo = hd_halo[conditions_halo]
    print("Halos: %d"%(len(hd_z_halo) ))

    """
    3. Computing Δ𝑡_𝑀𝑀 for a given redshift limit
    """
    
    
    np.save('Data/pairs_z%.1f/t_mm/pixel_%s.npy'%(redshift_limit, pixel_no), diff_time.value, allow_pickle=True)