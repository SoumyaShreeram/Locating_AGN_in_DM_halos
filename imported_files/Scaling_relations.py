"""
This python file contains all the functions relavant to the scaling relations between galaxy clusters


Script written by: Soumya Shreeram 
Project supervised by Johan Comparat 
Date created: 21st June 2021
"""
# scipy modules
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

# astropy modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, QTable, Column, join
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, z_at_value

import numpy as np
from scipy.stats import gaussian_kde
import os
import glob

# plotting imports
import matplotlib
import matplotlib.pyplot as plt

# personal imports
import Agn_incidence_from_Major_Mergers as aimm
import plotting_cswl05 as pt
import All_sky as sky
import Exploring_DM_Halos as edh

def readCatFile(pixel_file, dir_name='cat_eRO_CLU_b8_CM_0_pixS_20.0_M500c_13.0_FX_-14.5_galNH'):
	# set path
    filename = os.path.join('/data24s', 'comparat', 'simulation', 'UNIT', 'ROCKSTAR_HALOS', 'fixedAmp_InvPhase_001', dir_name, pixel_file)
    return filename

def getCluParams(pixel_no, redshift_limit = 2,\
 cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)):
    """
    Function to get the relavant parameters from the cluster file
    @pixel_no :: the pixel number of the cluster file 
    Note: the cluster file for a given pixel is further divided into multiple files
    """
    pixel_name = 'c_'+pixel_no+'_N_*.fit'

    # get all the cluster files for the given pixel
    for i, f in enumerate(glob.glob(readCatFile(pixel_file=pixel_name))):
        # for case 0, initiate the array 
        if i == 0:
            hd_clu_all =  Table.read(f, format='fits')   
            hd_clu_all.keep_columns(['RA', 'DEC', 'redshift_R', 'dL', 'nH', \
            'HALO_Mvir', 'HALO_Rvir', 'HALO_M500c', 'CLUSTER_LX_soft_RF',\
            'CLUSTER_LX_soft_RF_halfR500', 'CLUSTER_LX_soft_RF_twiceR500',\
            'CLUSTER_kT', 'CLUSTER_FX_soft', 'R500c_kpc', 'R500c_arcmin', \
            'galaxy_SMHMR_mass'])

        else:
            hd_clu_all_new =  Table.read(f, format='fits')   
            hd_clu_all_new.keep_columns(['RA', 'DEC', 'redshift_R', 'dL', 'nH', \
            'HALO_Mvir', 'HALO_Rvir', 'HALO_M500c', 'CLUSTER_LX_soft_RF',\
            'CLUSTER_LX_soft_RF_halfR500', 'CLUSTER_LX_soft_RF_twiceR500',\
            'CLUSTER_kT', 'CLUSTER_FX_soft', 'R500c_kpc', 'R500c_arcmin', \
            'galaxy_SMHMR_mass'])

            # joining the tables
            hd_clu_all = join(hd_clu_all, hd_clu_all_new, join_type='outer')
    hd_clu_all = hd_clu_all[hd_clu_all['redshift_R']<redshift_limit]
    return hd_clu_all 


def getAGNparams(pixel_no, frac_cp_agn):
	catAGN = readCatFile(dir_name='CP_10_sigma_1.0_frac_%.2f', pixel_file=pixel_no+'.fit')
	hd_agn = Table.read(catAGN, format='fits')
	hd_agn.keep_columns(['RA', 'DEC', 'redshift_R', 'dL', 'nH',\
	 'galaxy_SMHMR_mass', 'HALO_Mvir', 'HALO_Rvir', 'LX_soft',\
	  'FX_soft', 'FX_soft_attenuated'])
	return hd_agn

    
def checkIfAGNinR500c(pixel_no, frac_r500c_arr, f, frac_r500c, hd_agn, \
 idx_cluster, idx_agn, d2d):
    """
    Function checks if AGN is within cluster (some fraction)*r500c and adds flux accordingly
    """
    hd_clu_params = getCluParams(pixel_no)
    log_Lx = hd_clu_params['CLUSTER_LX_soft_RF']

    # arr whose Lx values are changed based on wether or not an AGN is within it
    scaled_LX_soft_RF_agn = hd_clu_params['CLUSTER_LX_soft_RF'] 
    
    # get the r500c in degrees for the clusters with agn neighbours
    r500c = hd_clu_params[idx_cluster]['R500c_arcmin'].to(u.degree)
    print('Scaling flux if AGN exist between %.1f and %.1f times R_500c'%(frac_r500c_arr[f], frac_r500c))
    
    # if agn is within frac*R500c bin of the cluster : frac_r500c = 0-0.2, 0-0.5, 0-1, etc 
    cond_for_agn_in_r500c = (frac_r500c_arr[f]*r500c < d2d) & (d2d <= frac_r500c*r500c)
    agn_within_fr500c = np.where( cond_for_agn_in_r500c )

    idx_clu_w_agn = idx_cluster[agn_within_fr500c]
    idx_clu_unique_w_agn = np.unique(idx_clu_w_agn)

    idx_agn_in_clu = idx_agn[agn_within_fr500c]
    agn_flux = hd_agn[idx_agn_in_clu]['FX_soft']
    print('%.1f percent clusters have AGN neighbours'%(100*len(idx_clu_unique_w_agn)/len(hd_clu_params)))

    # get the fraction of agn flux wrt the original cluster flux
    cluster_flux = hd_clu_params[idx_clu_w_agn]['CLUSTER_FX_soft']
    frac_up_flux = (cluster_flux + agn_flux)/cluster_flux

    for idx_unique in idx_clu_unique_w_agn:
        sum_over_idx = np.where(idx_unique == idx_clu_w_agn)
        
        # get the contribution of the brightest AGN
        agns_in_clu = agn_flux[sum_over_idx]
        brightest_agn_idx = np.where(agns_in_clu == np.max(agns_in_clu))
        frac_up = frac_up_flux[brightest_agn_idx]

        # get the contribution of the 2nd brightest AGN
        if len(agns_in_clu) > 1:
            agns_in_clu_2 = agns_in_clu[agns_in_clu !=  np.max(agns_in_clu)] 
            second_brightest_agn_idx = np.where(agns_in_clu_2 == np.max(agns_in_clu_2))
            frac_up += frac_up_flux[second_brightest_agn_idx]

        # scaling the cluster rest frame luminosity by this factor
        f_scale_up = (1+ frac_up) 
        
        scaled_LX_soft_RF_agn[idx_unique] = log_Lx[idx_unique] + np.log10(f_scale_up)
    return scaled_LX_soft_RF_agn, len(agn_within_fr500c[0])

def getAGNbkgFlux(pixel_no, area_pixel = 129600/np.pi/768, min_flux_agn=5e-15):
    hd_agn_all, _, _ = edh.getHeaders(pixel_no, np.array(['agn']))
    bkg_flux = hd_agn_all['FX_soft'][hd_agn_all['FX_soft']<min_flux_agn]
    sum_bkg_agn_flux = np.sum(bkg_flux) #*u.erg/(u.second*u.m*u.m)
    bkg_agn_flux_px = sum_bkg_agn_flux/area_pixel
    return bkg_agn_flux_px


def checkIfAGNfluxinR500c(pixel_no, frac_r500c, hd_agn, idx_cluster, idx_agn, d2d,\
 min_flux_agn= 5e-15, redshift_limit=2, frac_cp_agn=0.03):
    """
    Function checks if AGN is within cluster (some fraction)*r500c and adds flux accordingly
    """
    hd_clu_params = getCluParams(pixel_no)
    log_Lx = hd_clu_params['CLUSTER_LX_soft_RF']
    
    # counts the clusters with changed flux
    count_change = 0

    # get the bkg agn flux 
    bkg_agn_flux_px = getAGNbkgFlux(pixel_no, min_flux_agn=min_flux_agn)
    
    # arr whose Lx values are changed based on wether or not an AGN is within it
    scaled_LX_soft_RF_agn = hd_clu_params['CLUSTER_LX_soft_RF'] 
    
    # get the r500c in degrees for the clusters with agn neighbours
    r500c = hd_clu_params[idx_cluster]['R500c_arcmin'].to(u.degree)
    print('Scaling flux if AGN exist inside %.1f times R_500c'%(frac_r500c))
    
    # if agn is within frac*R500c bin of the cluster : frac_r500c = 0-0.2, .2-0.5, 0.5-1, etc 
    cond_for_agn_in_r500c = (d2d <= frac_r500c*r500c)
    agn_within_fr500c = np.where( cond_for_agn_in_r500c )

    idx_clu_w_agn = idx_cluster[agn_within_fr500c]
    idx_clu_unique_w_agn = np.unique(idx_clu_w_agn)

    idx_agn_in_clu = idx_agn[agn_within_fr500c]
    agn_flux = hd_agn[idx_agn_in_clu]['FX_soft']
    print('%.1f percent clusters have AGN neighbours'%(100*len(idx_clu_unique_w_agn)/len(hd_clu_params)))

    # get the fraction of agn flux wrt the original cluster flux
    cluster_flux = hd_clu_params[idx_clu_w_agn]['CLUSTER_FX_soft']
    
    for idx_unique in idx_clu_unique_w_agn:
        sum_over_idx = np.where(idx_unique == idx_clu_w_agn)
        
        r500x_clu = (hd_clu_params[idx_unique]['R500c_arcmin']*u.arcmin).to(u.degree)
        r500x_clu = r500x_clu/u.deg

        # get the contribution of the AGN with subtracted background AGN flux
        bkg_agn_flux = bkg_agn_flux_px*(np.pi)*(frac_r500c*r500x_clu)**2
        total_agn_flux = np.sum(agn_flux[sum_over_idx])
        frac_up = (total_agn_flux - bkg_agn_flux)/hd_clu_params[idx_unique]['CLUSTER_FX_soft']
        
        # scaling the cluster rest frame luminosity by this factor
        f_Lx_scale_up = (1 + frac_up) 
        if f_Lx_scale_up > 1:
            count_change += 1
            scaled_LX_soft_RF_agn[idx_unique] = log_Lx[idx_unique] + np.log10(f_Lx_scale_up)
        else:
            scaled_LX_soft_RF_agn[idx_unique] = log_Lx[idx_unique]
    return scaled_LX_soft_RF_agn, count_change

def getLogLx(pixel_no, cosmo = FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115)):
    """
    Function to get the log10 of the X-ray rest frame luminosity of the cluster
    """
    hd_clu_params = getCluParams(pixel_no)
    
    # define relavant quantities for the plot
    E_z = cosmo.H(hd_clu_params['redshift_R'])/cosmo.H(0)
    log_M500c = np.log10(hd_clu_params['HALO_M500c'])/u.M_sun + np.log10(E_z)
    log_Lx = hd_clu_params['CLUSTER_LX_soft_RF'] - np.log10(E_z)
    return log_Lx

def createMassLuminosityBins(log_M500c, log_Lx_Ez, dlog_M500c = 0.05):
    # create M500c bins in log10
    log_M500c_min, log_M500c_max = np.round(np.min(log_M500c), 1), np.round(np.max(log_M500c)-0.1, 1)
    log_M500c_bins = np.arange( log_M500c_min, log_M500c_max, dlog_M500c )
    
    # define a mean and std Lx array for each M500c bin created
    log_Lx_mean, log_Lx_std = [], []
    
    for bin_idx, (l_edge, r_edge) in enumerate(zip(log_M500c_bins, log_M500c_bins+dlog_M500c)): 
        # get all the clusters within this mass bin
        clu_Lx_condition = (log_M500c>l_edge) & (log_M500c < r_edge)
        log_Lx_temp_Ez = log_Lx_Ez[clu_Lx_condition]
        
        # fill the mean and std values for the Lx bins under concern
        log_Lx_mean.append(np.mean(log_Lx_temp_Ez))

        sigma_percentile = (np.percentile(log_Lx_temp_Ez, 84.13) - np.percentile(log_Lx_temp_Ez, 15.87))/2
        #log_Lx_std.append( sigma_percentile/2 )
        
        #log_Lx_std.append(np.std(log_Lx_temp)*.6745)
        log_Lx_std.append(sigma_percentile)
    return log_M500c_bins, log_Lx_mean, log_Lx_std



def getBinsLxM500c(log_Lx, pixel_no='000000', dlog_M500c = 0.1, redshift_limit=2, \
 cosmo =  FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115), full_sky=True,\
 model_name='Model_A0', frac_cp=0.2):
    """
    Function bins the scatter into log bins of M500c 
    """
    if full_sky:
        model_dir = '../Data/pairs_z%.1f/Scaling_relations/'%redshift_limit
        # all sky cluster file
        if model_name == 'ModelNone':
            fname = model_dir + 'CLU_with_scaled_Lx_all_sky_%s.fit'%model_name
        else:
            fname = model_dir + '%s/CLU_with_scaled_Lx_all_sky_frac_cp_%.2f.fit'%(model_name, frac_cp)

        hd_clu_params = Table.read(fname, format='fits')
    else:
        hd_clu_params = getCluParams(pixel_no)
    
    # define relavant quantities for the plot
    E_z = cosmo.H(hd_clu_params['redshift_R'])/cosmo.H(0)
    log_M500c = np.log10(hd_clu_params['HALO_M500c'])/u.M_sun + np.log10(E_z)
    log_Lx_Ez = log_Lx - np.log10(E_z)

    log_M500c_bins, log_Lx_mean, log_Lx_std = createMassLuminosityBins(log_M500c, log_Lx_Ez, dlog_M500c=dlog_M500c)
    return log_M500c_bins, dlog_M500c, np.array([log_Lx_mean])[0], np.array([log_Lx_std])[0]


def giveLuminosityEveryMassBin(log_M500c, log_Lx, dlog_M500c = 0.05):
    # create M500c bins in log10
    log_M500c_min, log_M500c_max = np.round(np.min(log_M500c), 1), np.round(np.max(log_M500c)-0.1, 1)
    log_M500c_bins = np.arange( log_M500c_min, log_M500c_max, dlog_M500c )
    
    # define a mean and std Lx array for each M500c bin created
    log_Lx_every_M500c, log_M500c_every_Lx = [], []
    
    for bin_idx, (l_edge, r_edge) in enumerate(zip(log_M500c_bins, log_M500c_bins+dlog_M500c)): 
        # get all the clusters within this mass bin
        clu_Lx_condition = (log_M500c>l_edge) & (log_M500c < r_edge)
        log_Lx_temp = log_Lx[clu_Lx_condition]
        log_Lx_every_M500c.append(log_Lx_temp)
    return log_M500c_bins, log_Lx_every_M500c

def checkLxDist(log_Lx, pixel_no='000000', dlog_M500c = 0.1, redshift_limit=2, \
 cosmo =  FlatLambdaCDM(H0=67.77*u.km/u.s/u.Mpc, Om0=0.307115), full_sky=True, model_name='Model_A0'):
    """
    Function check the scatter in log bins of M500c 
    """
    if full_sky:
        # all sky cluster file
        fname = '../Data/pairs_z%.1f/CLU_with_scaled_Lx_all_sky_%s.fit'%(redshift_limit, model_name)
        hd_clu_params = Table.read(fname, format='fits')
        print('Total clusters all sky:', len(hd_clu_params))
    else:
        hd_clu_params = getCluParams(pixel_no)
    
    # define relavant quantities for the plot
    E_z = cosmo.H(hd_clu_params['redshift_R'])/cosmo.H(0)
    log_M500c = np.log10(hd_clu_params['HALO_M500c'])/u.M_sun + np.log10(E_z)
    log_Lx = log_Lx - np.log10(E_z)

    log_M500c_bins, log_Lx_every_M500c = giveLuminosityEveryMassBin(log_M500c, log_Lx, dlog_M500c=dlog_M500c)
    return log_M500c_bins, dlog_M500c, np.array(log_Lx_every_M500c, dtype=object)

def readLxM500cWithAgnTable(frac_r500c_arr, scaled_LX_soft_RF_agn_all, pixel_no='000000', using_cp_catAGN=False, redshift_limit=2):
    "Function to write the data into a fits table using astropy"
    if using_cp_catAGN:
        t = Table.read('../Data/pairs_z%.1f/Scaling_relations/CLU_LX_RF_scaled_%s_catAGN_all_px.fit'%(redshift_limit, 'cp'), format='fits')
    else:
        t =  Table.read('../Data/pairs_z%.1f/Scaling_relations/CLU_LX_RF_scaled_catAGN_all_px.fit'%(redshift_limit), format='fits')
    return t 

def write2Table(frac_r500c_arr, scaled_LX_soft_RF_agn_all, model_name='Model_A0', pixel_no='000000',\
 using_cp_catAGN=True, redshift_limit=2, frac_cp=0.2):
    "Function to write the data into a fits table using astropy"
    names = ['f_AGN_%.1fR_500c'%f for f in frac_r500c_arr[1:]]
    data = [scaled_LX_soft_RF_agn_all[i] for i in range(len(scaled_LX_soft_RF_agn_all))]
    t = Table(data, names=names)
    
    if using_cp_catAGN:
        main_dir = '../Data/pairs_z%.1f/Scaling_relations/'%redshift_limit + model_name
        t.write(main_dir+'/CLU_LX_RF_scaled_frac_cp_%.2f_all_catAGN_px%s.fit'%(frac_cp, pixel_no), format='fits', overwrite=True)
    else:
        t.write('../Data/pairs_z%.1f/Scaling_relations/CLU_LX_RF_scaled_all_catAGN_px%s.fit'%(redshift_limit, pixel_no), format='fits', overwrite=True)
    return

def readTableLxM500c(pixel_no, using_cp_catAGN=False, redshift_limit=2, frac_cp_agn=0.03, model_name='Model_A0'):
    if using_cp_catAGN:
        cluster_Lx_fr500c_w_agn = Table.read('../Data/pairs_z%.1f/Scaling_relations/'%redshift_limit+model_name+'/CLU_LX_RF_scaled_cp_all_catAGN_px%s.fit'%pixel_no, format='fits')
    else:
        cluster_Lx_fr500c_w_agn = Table.read('../Data/pairs_z%.1f/Scaling_relations/CLU_LX_RF_scaled_all_catAGN_px%s.fit'%(redshift_limit, pixel_no), format='fits')    
    return cluster_Lx_fr500c_w_agn


def getAreaPixel(hd_halo):
    width_ra = np.abs(np.max(hd_halo['RA'])-np.min(hd_halo['RA'])) #*u.deg
    width_dec = np.abs(np.max(hd_halo['DEC'])-np.min(hd_halo['DEC'])) #*u.deg
    return width_ra*width_dec

def pixelEndpoints():
    a = np.arange(0, 800, 50)
    b = np.arange(1, len(a), 2)
    #a[b]=a[b]-1
    a = np.append(a, [767])
    return a

