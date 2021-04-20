# -*- coding: utf-8 -*-
"""Plotting.py for notebook 04_Modelling_AGN_fractions_from_literature

This python file contains all the functions used for modelling the AGN fraction based on measurements from literature

Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 20th April 2021
"""
# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np

# scipy modules
import scipy.odr as odr

import os
import importlib

# plotting imports
import matplotlib

"""
Functions begin
"""

def getXY(arr):
    "Function to get x and y arr from the parent arr"
    x = [np.vstack(arr)[i][0] for i in range(len(arr))]
    y = [np.vstack(arr)[i][1] for i in range(len(arr))]
    return x, y

def getErrArrays(x):
    "Function to get the error arrays from the parent array"
    x_err_arr = [x[i][:2] for i in range(len(x))]
    y_err_arr = [x[i][2:] for i in range(len(x))]
    return x_err_arr, y_err_arr

def getErr(x_err_arr, x, y_err_arr, y):
    "Function to transform the errors arrays into readable pyplot formats"
    xerr = np.abs(np.transpose(np.vstack(x_err_arr)) - x)
    yerr = np.abs(np.transpose(np.vstack(y_err_arr)) - y)
    return xerr, yerr

def powerLaw(beta, x):
    return  -beta[0]/np.power(x, beta[1])

def performODR(X, Y, xerr_all, yerr_all):
    "Function to fit the empirical data"
    # model object
    power_law_model = odr.Model(powerLaw)

    # data and reverse data object
    data = odr.RealData(X, Y, sx=xerr_all, sy=yerr_all)
    
    # odr with model and data
    myodr = odr.ODR(data, power_law_model, beta0=[0.2, 0.])
    
    out = myodr.run()
    out = myodr.restart(iter=1000)
    return out