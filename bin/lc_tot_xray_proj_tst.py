from binascii import b2a_base64
import swiftsimio as sw
import numpy as np  
import numpy.ma as ma
import matplotlib.pyplot as plt 
from joey_interpolate_X_Ray_spectra_supermaster import interpolate_X_Ray as interp_xray
from mpl_toolkits.axes_grid1 import make_axes_locatable


import h5py
import healpy as hp
from unyt import cm, erg, s
from numba import jit

import lightcone_io.particle_reader as pr
from matplotlib.colors import LogNorm
from matplotlib.image import NonUniformImage

from astropy.cosmology import FlatLambdaCDM, z_at_value
import os
from tqdm import tqdm

### create or enter a new directory
import os
path = '/cosma8/data/dp004/dc-chen3/narrow_emi_map/mass_tst/'
dir = 'tot_xray_tst'
path = path+dir
# Create a new directory because it does not exist 
if os.path.exists(path) is False:
  os.makedirs(path)
  print('dir '+"is created!")

### setting basics
halo_id = '10976067'
nside=16384
radius = np.radians(1) #deg
vec = np.array([-0.49919698,-0.03709323,-0.86569421])

##### load lightcone data functions ####
def compute_luminosity_distance(part_lc):
    distances = np.sqrt(part_lc['Coordinates'][:, 0]**2 + part_lc['Coordinates'][:, 1]**2 + part_lc['Coordinates'][:, 2]**2)
    redshifts = (1 / part_lc['ExpansionFactors'][()]) - 1
    luminosity_distances = distances * (1 + redshifts)
    return luminosity_distances 
    
@jit(nopython = True)
def assign_energy_to_bin(flux, bin_numbers_for_flux, lc_spec):
    for i in range(lc_spec.shape[0]):
        for j, bin in enumerate(bin_numbers_for_flux[i, :]):
            lc_spec[i, bin] += flux[i, j]

    return lc_spec

def load_snapshot():
    filename = "/cosma6/data/dp004/dc-kuge1/200LH_bestfit_tests/FLAMFIDL200N360Nearest_x_ray/bahamas_0008.hdf5"

    mask = sw.mask(filename)
    # The full metadata object is available from within the mask
    boxsize = mask.metadata.boxsize
    # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
    # We load a small region
    load_region = [[0.0 * b, 0.01 * b] for b in boxsize]

    # Constrain the mask
    mask.constrain_spatial(load_region)

    # Now load the snapshot with this mask
    data = sw.load(filename, mask=mask)

    return data

def update_data_structure(data, part_lc):
    data.gas.densities = part_lc['Densities'][()]
    data.gas.temperatures = part_lc['Temperatures'][()]
    data.gas.masses = part_lc['Masses'][()]
    data.gas.smoothed_element_mass_fractions.hydrogen = part_lc['SmoothedElementMassFractions'][:, 0]
    data.gas.smoothed_element_mass_fractions.helium = part_lc['SmoothedElementMassFractions'][:, 1]
    data.gas.smoothed_element_mass_fractions.carbon = part_lc['SmoothedElementMassFractions'][:, 2]
    data.gas.smoothed_element_mass_fractions.nitrogen = part_lc['SmoothedElementMassFractions'][:, 3]
    data.gas.smoothed_element_mass_fractions.oxygen = part_lc['SmoothedElementMassFractions'][:, 4]
    data.gas.smoothed_element_mass_fractions.neon = part_lc['SmoothedElementMassFractions'][:, 5]
    data.gas.smoothed_element_mass_fractions.magnesium = part_lc['SmoothedElementMassFractions'][:, 6]
    data.gas.smoothed_element_mass_fractions.silicon = part_lc['SmoothedElementMassFractions'][:, 7]
    data.gas.smoothed_element_mass_fractions.iron = part_lc['SmoothedElementMassFractions'][:, 8]
    data.gas.redshifts = (1 / part_lc['ExpansionFactors'][()]) - 1
    return data

data = load_snapshot()


def load_lightcone(vector,radius):
    print('loading lightcone')
    # Specify one file from the spatially indexed lightcone particle data
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

    # Part of the sky to plot
    vector = vec  # Vector pointing at a spot on the sky
    radius = radius # Angular radius around that spot

    # Redshift range to plot (set to None for all redshifts in the lightcone)
    redshift_range = (0.0, 0.1)

    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename)

    # Read in the particle positions and masses
    property_names = ("Coordinates", "Masses", "SmoothedElementMassFractions", "Densities", "Temperatures", "ExpansionFactors")
    data = lightcone["Gas"].read(property_names=property_names,
                                redshift_range=redshift_range,
                                vector=vector, radius=radius,)
    data['mass'] = data['Masses'].value
    
    ### save ra & dec
    data['RA'], data['DEC'] = hp.vec2ang(data['Coordinates'].value, lonlat = True)   
    ### convert ra,dec directly to xy 
    if os.path.exists('lc_spec_xy.npz'):
        print('loading xy')
        xy = np.load('lc_spec_xy.npz')
        data['x'] = xy['x']
        data['y'] = xy['y']
    else:
        print('converting xy')
        zoom_size=1 #deg
        cartproj = hp.projector.CartesianProj(
                lonra=[-zoom_size, zoom_size], latra=[-zoom_size, zoom_size], rot=hp.vec2ang(np.array(vec),lonlat=True)
            )
        ra,dec =hp.vec2ang(data['Coordinates'].value, lonlat = True)   
        x = np.zeros(ra.shape)
        y = np.zeros(dec.shape)
        for i in tqdm(range(len(ra))):
            x[i],y[i]= cartproj.ang2xy(ra[i],dec[i],lonlat=True)
        data['x']=x
        data['y']=y
        np.savez('lc_spec_xy.npz',x=x,y=y)
    return data

##### define function to interpolate x-ray flux #####
def compute_xrays(line_E,line_interval,z_halo):
    print('loading lightcone data')
    lightcone_data = load_lightcone(vec,radius)

    # Load a fake snapshot
    print('loading swiftsimio struct')
    data = load_snapshot()

    # Put relevant parts into snapshot data structure
    print('putting lightcone data into swiftsimio struct')
    data = update_data_structure(data, lightcone_data)
    z_max = data.gas.redshifts.max() # max redshift in sims
    z_halo_max = z_halo+0.025
    rest_line_bin = [line_E-line_interval,line_E+line_interval]
    obs_bin = [rest_line_bin[0]/(1+z_halo_max),rest_line_bin[1]]    

    for i, tp in enumerate(['pure', 'contaminated']):
        # Interpolate xrays
        print('interpolating xrays')
        if tp == 'pure':
            min_energy_for_interp = rest_line_bin[0]
            max_energy_for_interp = rest_line_bin[1]
        elif tp == 'contaminated':
            min_energy_for_interp = rest_line_bin[0]/ (1 + z_halo_max)**2/(1+z_max)
            max_energy_for_interp = rest_line_bin[1]
        xray_spec_lum, restframe_energies = interp_xray(data.gas.densities, data.gas.temperatures, data.gas.smoothed_element_mass_fractions, data.gas.redshifts, data.gas.masses, fill_value = 0, bin_energy_lims = [min_energy_for_interp, max_energy_for_interp])
        print('computing xray luminosity')
        received_energies = restframe_energies / (1 + data.gas.redshifts[:, np.newaxis])
        # Compute flux for all particles
        lum_distance = compute_luminosity_distance(lightcone_data)
        # For each particle bin see which energy bin x ray luminosity falls into 
        print('selecting luminosity')
        xray_spec_lum = [x if obs_bin[0]<x<obs_bin[1] else 0 for x in xray_spec_lum]
        print('computing xray flux')
        xray_spec_flux = xray_spec_lum / (4 * np.pi * lum_distance[:, np.newaxis]**2)
        np.save('Xray_spectrum_{}'.format(tp),xray_spec_flux)

###### define plot functions #####
def msk_dat(dat):
    dat = ma.masked_invalid(dat)
    dat = ma.masked_where(dat<=0,dat)
    vmin = dat.min()
    vmax = dat.max()
    if vmax==np.nan:
        return dat, 'nan','nan'   
    else:
        return dat,vmin,vmax
        
def plot_xy(dat,ax,prop_names,x,y):
    grid = 0.05
    xbins = np.round((np.max(x)-np.min(x))/grid)+1 # 50 kpc grid
    ybins = np.round((np.max(y)-np.min(y))/grid)+1 # 50 kpc grid
    xedges = np.arange(np.min(x),np.min(x)+xbins*grid,grid)
    yedges = np.arange(np.min(y),np.min(y)+ybins*grid,grid)
    dat_bin, xedges, yedges  = np.histogram2d(x,y, 
        weights = dat,  bins = (xedges, yedges)
        )
    dat_bin,vmin,vmax = msk_dat(dat_bin)
    print('mass binned data ranges %s, %s'%(str(vmin),str(vmax)))
    if vmax>0:
        dat_bin = dat_bin.T
        X, Y = np.meshgrid(xedges, yedges)
        dat_bin[dat_bin<=0]=np.nan
        im = ax.pcolormesh(X,Y,dat_bin,norm = LogNorm(),cmap='binary')
     
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title(prop_names)
        ax.set_xticks(np.linspace(np.min(x),np.min(x)+xbins*grid,10))
        ax.set_yticks(np.linspace(np.min(y),np.min(y)+ybins*grid,10))
    else: 
        print('error: all nan')

###### read & compute data #####
lightcone_data = load_lightcone(vec,radius)

# Add a spectrum to each particle
if os.path.exists('Xray_spectrum_pure.npy'):
    tot = np.load('Xray_spectrum_pure.npy')
else:
    compute_xrays(0.653,0.001,0.05)
dat = tot  
# dat = lightcone_data['mass']

######## plot xy ##########
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
plot_xy(dat,ax,'mass',lightcone_data['x'],lightcone_data['y'])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')   
plt.savefig('halo_tot_xray_tst_xy_'+str(halo_id)+'.png')

####### plot ra dec ########
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
plot_xy(dat,ax,'mass',lightcone_data['RA'],lightcone_data['DEC'])
ax.set_xlabel('RA (deg)')
ax.set_ylabel('DEC (deg)')   
plt.savefig('halo_tot_xray_tst_radec_'+str(halo_id)+'.png')

####### convert ra,dec to pix and cartproj ----!!! value will be different, resolution will be smaller!  ########
pix = hp.ang2pix(nside,lightcone_data['RA'],lightcone_data['DEC'],lonlat=True)
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
### creat new healpix map
npix = hp.pixelfunc.nside2npix(nside)
zoom_size=1 #deg
cartproj = hp.projector.CartesianProj(
        lonra=[-zoom_size, zoom_size], latra=[-zoom_size, zoom_size], rot=hp.vec2ang(np.array(vec),lonlat=True)
    )
mollproj = hp.projector.MollweideProj()
dat = dat
map_data = np.zeros(npix, dtype=float)
np.add.at(map_data, pix, dat)

pR = cartproj.projmap(map_data, lambda x, y, z: hp.vec2pix(nside, x, y, z))
pR[pR<=0]=np.nan

if np.count_nonzero(~np.isnan(pR))>0:
    im = ax.imshow(pR, norm = LogNorm(),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('gas mass')

plt.tight_layout()
plt.savefig('halo_tot_xray_tst_pix_'+str(halo_id)+'.png')
plt.close()