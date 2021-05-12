"""
05. Preliminary comparison of the 𝑓𝑀𝑀 between simulation and data

The notebook is similar to the notebook 02, which builds a major merger catalog. However, here things are done slightly differently: (1) halo pairs are generated, (2) the criteria are applied, rather than the other way around (as shown in notebook 02).

1. Loading data and defining input parameters
2. Finding pairs and creating a major/minor sample
2. Studying merger fraction 𝑓𝑀𝑀 as a function of redshift

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

# pixel number from the simulation file
pixel_no = '000000'

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777

"""
2. Parameters for creating the Major Merger catalogue
"""
# get shell volume and projected radius bins [Mpc]
r_p, shell_volume = aimm.shellVolume()

# normalized xoff array 
xoff_arr = [0.1, 0.2, 0.4, 0.5, 0.7]

# max mass ratio to classify as a major merger [dimensionless]
mass_max = 3

# defining the redshift bin for a merger in terms of dv = c*z [km/s]
dz_cut =  0.001

# keywords can be: 'mm and dv', 'dv' or 'all' 
# look at decideBools(..) function is cswl for more details)
keyword = 'mm and dv'

"""
3. Open files and get relevant data
"""
_, hd_halo, _ = edh.getHeaders(pixel_no, np.array([ 'halo']))

# Extracting positions and redshifts of the halos
_, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)
hd_z_halo = hd_halo[conditions_halo]

print("Halos: %d"%(len(hd_z_halo) ))

"""
4. Studying the effect of 𝑋̃_off on MM pairs

Now that all the pairs for the chosen cases of time since major mergers are computed, we can proceed to calculate the fraction of halo pairs for each case.

𝑓_halo_pairs = NP / N(N−1)×Shell volume

where 𝑁𝑃 is the number of pairs and 𝑁 is the total number of objects from which pairs are chosen.
"""
pairs_all = cswl.openPairsFiles(data_dir='Data/pairs_z%.1f/'%redshift_limit, key = keyword, dz_cut= dz_cut)
    
xoff_all = hd_z_halo['HALO_Xoff']/hd_z_halo['HALO_Rvir']

for xoff in xoff_arr[0:1]:
    count_xoff_arr = []

    for r in range(len(r_p)): 
        print('\n ---- Merger pairs within radius %.1f Mpc, Xoff = %.2f ---'%((1e3*r_p[r]), xoff))

        _, count_xoff = cswl.selectParameterPairs(hd_z_halo, pairs_all[0][r], cosmo, xoff_all, param = xoff, redshift_limit = redshift_limit, string_param = 'x_off')
        count_xoff_arr.append(count_xoff)

    cswl.saveTmmFiles(keyword, xoff, arr = count_xoff_arr, redshift_limit = redshift_limit, param='x_off')
