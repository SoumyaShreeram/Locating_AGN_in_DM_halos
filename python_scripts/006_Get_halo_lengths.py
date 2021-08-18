"""
006. Quick script that counts the number of halos in each pixel of the UNIT simulation

Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 04th June 2021
"""
import numpy as np
import sys

import astropy.io.fits as fits
from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('../imported_files/')
import Exploring_DM_Halos as edh
import All_sky as sky 

get_halo_lengths = False

if get_halo_lengths:
	sky.getHaloLengths(redshift_limit=0.2)
else:
	sky.getAgnLengths(redshift_limit=2, frac_cp_agn=0.04)