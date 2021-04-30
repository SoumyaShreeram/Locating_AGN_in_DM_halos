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
redshift_limit = 1.0

# pixel number from the simulation file
pixel_no = '000000'

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

# get shell volume and projected radius bins [Mpc]
r_p, dr_p, shell_volume = aimm.shellVolume()

# time since merger array [Gyr]
dt_m_arr = [0.5, 1, 2, 3, 4]


"""
2. Open files and get relevant data
"""
_, hd_halo, _ = edh.getHeaders(pixel_no, np.array(['halo']))

# Extracting positions and redshifts of the halos
pos_z_halo, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)

hd_z_halo = hd_halo[conditions_halo]
print("Halos: %d"%(len(hd_z_halo) ))

"""
3. Computing Δ𝑡_𝑀𝑀 for a given redshift limit
"""
diff_t_mm_arr = []

for i in range(len(hd_halo)):
    print('-- ', i, ' --')
    diff_t_mm = cswl.calTmm(cosmo, hd_z_halo[i]['HALO_scale_of_last_MM'], hd_z_halo[i]['redshift_R'])
        
    diff_t_mm_arr.append(diff_t_mm.value)
    
np.save('Data/diff_t_mm_arr_z%.1f.npy'%(redshift_limit), diff_t_mm_arr, allow_pickle=True)