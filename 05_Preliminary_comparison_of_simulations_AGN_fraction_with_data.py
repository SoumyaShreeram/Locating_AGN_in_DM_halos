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

# agn parameters
agn_FX_soft = 0

# galaxy parameters
galaxy_SMHMR_mass = 9 # unit: log 10, M_solar

# halo parameters
halo_mass_500c = 10**13.7 # solar masses
central_Mvir = 13.7 # unit: log 10, M_solar
cluster_params = [halo_mass_500c, central_Mvir]

# pixel number from the simulation file
pixel_no = '000000'

# Define cosmology used in the notebook
cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)
h = 0.6777
L_box = 1000.0 / h

"""
2. Parameters for creating the Major Merger catalogue
"""
# get shell volume and projected radius bins [Mpc]
r_p, dr_p, shell_volume = aimm.shellVolume()

# time since merger [Gyr]
time_since_merger = 5

# time since merger array [Gyr]
dt_m_arr = [0.5, 1, 2, 3, 4]

# max mass ratio to classify as a major merger [dimensionless]
mass_max = 3

# defining the redshift bin for a merger in terms of dv = c*z [km/s]
dv_cut =  500

# BOOLEAN: if the pairs have already been computed before
run_find_pairs = False

# BOOLEAN: if we want to find all pairs (not just major pairs)
major_mergers_only = True

"""
3. Open files and get relevant data
"""
hd_agn, hd_halo, _ = edh.getHeaders(pixel_no, np.array(['agn', 'halo']))

# Extracting positions and redshifts of the AGNs, galaxies, and halos
# agns
pos_z_AGN, _, conditions_agn = edh.getAgnData(hd_agn, agn_FX_soft, redshift_limit)    

# galaxies and halos
pos_z_gal, _, conditions_gal = edh.getGalaxyData(hd_halo, galaxy_SMHMR_mass, redshift_limit)
pos_z_halo, _, conditions_halo = edh.getGalaxyData(hd_halo, '', redshift_limit)

print("AGNs: %d, Galaxies: %d, Halos: %d"%(len(pos_z_AGN[0]), len(pos_z_gal[0]), len(pos_z_halo[0]) ))

"""
3. Finding halo pairs
"""
if run_find_pairs:
    for r in [14, 15]:
        print('\n ---- pairs within radius %.2f Mpc ---'%r_p[r])
        pos_spherical = aimm.getSphericalCoord(hd_halo[conditions_halo])

        # create tree
        tree_data = cKDTree(np.transpose(np.abs(pos_spherical)), leafsize=1000.0)

        # list of lists of all pair indicies per DM halo
        pairs_idx = tree_data.query_ball_tree(tree_data, r=r_p[r], p=2) 
        count_pairs, _ = cswl.countPairs(pairs_idx) 
        
        if major_mergers_only:
            # (1) choose only major pairs (major mergers)
            all_mm_idx, count_major_pairs = cswl.majorMergerSelection(hd_halo[conditions_halo], pairs_idx)
            pairs_idx = all_mm_idx
            
        # (2) choose only major pairs that satisfy the delta v criteria
        all_dv_idx, count_dv_major_pairs = cswl.deltaVelSelection(hd_halo[conditions_halo], pairs_idx)
        
        # save file
        np.save('Data/pairs_z%d/pairs_idx_r%d_mm%d_dv%d.npy'%(redshift_limit, r, mass_max, dv_cut), all_dv_idx, allow_pickle=True)

"""
4. Studying the effect of Δ𝑡_merger on MM pairs

Now that all the pairs for the chosen cases of time since major mergers are computed, we can proceed to calculate the fraction of halo pairs for each case.

𝑓_halo_pairs = NP / N(N−1)×Shell volume

where 𝑁𝑃 is the number of pairs and 𝑁 is the total number of objects from which pairs are chosen.
"""

pairs_all = cswl.openPairsFiles(data_dir='Data/pairs_z2/Major_pairs/')

count_t_mm_arr_all_radius = np.zeros( (0, len(r_p) ) )

for dt_m in [dt_m_arr[-1]]:
    print('\n Choosing pairs with T_MM ~ %.1f'%dt_m)
    count_t_mm_arr = []
    
    for r in range(len(r_p)): 
        print('r_p = %.2f Mpc ...\n'%r_p[r])
        _, count_t_mm = cswl.defineTimeSinceMergeCut(hd_halo[conditions_halo], pairs_all[0][r], cosmo, time_since_merger = dt_m)
        count_t_mm_arr.append(count_t_mm)
    
    # save the counts for all radius bins for a given time since merger
    count_t_mm_arr_all_radius = np.append(count_t_mm_arr_all_radius, [count_t_mm_arr], axis=0)
    
np.save('Data/pairs_z%d/Major_pairs/all_pairs_t_mm%.1f_r.npy'%(redshift_limit, dt_m), count_t_mm_arr_all_radius, allow_pickle=True)