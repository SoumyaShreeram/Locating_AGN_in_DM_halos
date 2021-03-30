# -*- coding: utf-8 -*-
"""Plotting.py for notebook 02_AGN_incidence_in_Major_Mergers

This python file contains all the functions used for plotting graphs and maps in the 2nd notebook (.ipynb) of the repository: 02. Creating a Major Merger (MM) catalogue to study AGN incidence due to galaxy mergers

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 30th March 2021
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# scipy modules
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

import os
import importlib

# plotting imports
import matplotlib
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import Exploring_DM_Haloes as edh