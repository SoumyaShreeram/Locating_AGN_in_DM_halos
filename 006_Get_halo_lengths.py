"""
006. Quick script that counts the number of halos in each pixel of the UNIT simulation

Script written by: Soumya Shreeram
Project supervised by: Johan Comparat
Date: 04th June 2021
"""
import numpy as np
import sys

# Load the imported file(s) that contains all the functions used in this notebooks
sys.path.append('imported_files/')
import Exploring_DM_Halos as edh
import All_sky as sky 

sky.getHaloLengths()