from binascii import b2a_base64
import swiftsimio as sw
import numpy as np  
import numpy.ma as ma
import matplotlib.pyplot as plt 
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

### setting basics
halo_id = '10976067'
halo_z = 0.05
nside=16384
radius = np.radians(1) #deg
vec = np.array([-0.49919698,-0.03709323,-0.86569421])

# At the moment the interpolation code expects swiftsimio objects, so we need to load a snapshot to get that
# Then overwrite the relevant parts
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
    data.gas.masses = part_lc['Masses'][()]
    return data

data = load_snapshot()

def load_lightcone(vector,radius):
    '''
    This func load particle lightcone data
    '''
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
    property_names = ("Coordinates", "Masses")
    data = lightcone["Gas"].read_exact(property_names=property_names,
                                redshift_range=redshift_range,
                                vector=vector, radius=radius,)
    data['mass'] = data['Masses']
    
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
        np.savez('lc_spec_xy',x=x,y=y)
    return data

num_spec_bins = 10
lightcone_data = load_lightcone(vec,radius)
print('loading lightcone...')


# ####### plot xy #########
# def msk_dat(dat):
#     dat = ma.masked_invalid(dat)
#     dat = ma.masked_where(dat<=0,dat)
#     vmin = dat.min()
#     vmax = dat.max()
#     if vmax==np.nan:
#         return dat, 'nan','nan'   
#     else:
#         return dat,vmin,vmax
        
# def plot_xy(dat,ax,prop_names,x,y):
#     grid = 0.01
#     xbins = np.round((np.max(x)-np.min(x))/grid)+1 # 5 kpc grid
#     ybins = np.round((np.max(y)-np.min(y))/grid)+1 # 5 kpc grid
#     xedges = np.arange(np.min(x),np.min(x)+xbins*grid,grid)
#     yedges = np.arange(np.min(y),np.min(y)+ybins*grid,grid)
#     dat_bin, xedges, yedges  = np.histogram2d(x,y, 
#         weights = dat,  bins = (xedges, yedges)
#         )
#     dat_bin,vmin,vmax = msk_dat(dat_bin)
#     print('mass binned data ranges %s, %s'%(str(vmin),str(vmax)))
#     if vmax>0:
#         dat_bin = dat_bin.T
#         X, Y = np.meshgrid(xedges, yedges)
#         dat_bin[dat_bin<=0]=np.nan
#         im = ax.pcolormesh(X,Y,dat_bin,norm = LogNorm(),cmap='binary')
     
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im, cax=cax, orientation='vertical')
#         ax.set_title(prop_names)
#         ax.set_xticks(np.linspace(np.min(x),np.min(x)+xbins*grid,10))
#         ax.set_yticks(np.linspace(np.min(y),np.min(y)+ybins*grid,10))
#     else: 
#         print('error: all nan')

zoom_size=1 #deg
# def lim(x,y,edge):
#     x_mid = np.median(x)
#     y_mid = np.median(y)
#     return [[x_mid-edge,x_mid+edge],[y_mid-edge,y_mid+edge]]

# ####### plot ra dec ########
# fig, ax = plt.subplots(1, 1, figsize = (6, 6))
# plot_xy(lightcone_data['mass'],ax,'mass',lightcone_data['RA'],lightcone_data['DEC'])
# radec_lim = lim(lightcone_data['RA'],lightcone_data['DEC'],zoom_size)
# ax.set_xlim(radec_lim[0][0],radec_lim[0][1])
# ax.set_ylim(radec_lim[1][0],radec_lim[1][1])
# ax.set_xlabel('RA (deg)')
# ax.set_ylabel('DEC (deg)')   
# plt.savefig('halo_mass_tst_radec_'+str(halo_id)+'.png')
# plt.close()

# ####### plot xy ########
# fig, ax = plt.subplots(1, 1, figsize = (6, 6))
# plot_xy(lightcone_data['mass'],ax,'mass',lightcone_data['x'],lightcone_data['y'])
# ax.set_xlabel('x (Mpc)')
# ax.set_ylabel('y (Mpc)')  
# cosmo = FlatLambdaCDM(H0=68,Om0=0.3)
# fov_Mpc = cosmo.kpc_comoving_per_arcmin(halo_z).value/1e3*60
# # xy_lim = lim(lightcone_data['x'],lightcone_data['y'],fov_Mpc)
# # ax.set_xlim(xy_lim[0][0],xy_lim[0][1])
# # ax.set_ylim(xy_lim[1][0],xy_lim[1][1])
# plt.savefig('halo_mass_tst_xy_'+str(halo_id)+'.png')
# plt.close()


####### convert ra,dec to pix and cartproj ----!!! value will be different, resolution will be smaller!  ########
pix = hp.ang2pix(nside,lightcone_data['RA'],lightcone_data['DEC'],lonlat=True)
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
### creat new healpix map
npix = hp.pixelfunc.nside2npix(nside)
# calculate resolution and area for each pixel
res = hp.nside2resol(nside,arcmin=True)
pix_a = res**2
cartproj = hp.projector.CartesianProj(
        lonra=[-zoom_size, zoom_size], latra=[-zoom_size, zoom_size], rot=hp.vec2ang(np.array(vec),lonlat=True)
    )
dat = lightcone_data['mass'].value
map_data = np.zeros(npix, dtype=float)
np.add.at(map_data, pix, dat)

pR = cartproj.projmap(map_data/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
pR[pR<=0]=np.nan

if np.count_nonzero(~np.isnan(pR))>0:
    im = ax.imshow(pR, norm = LogNorm(),cmap='binary',extent=[-60,60,-60,60])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(r'gas mass surface density (10$M_sun/arcmin^2$)')
    ax.set_xlabel('ra (arcmin)')
    ax.set_ylabel('dec (arcmin)')
plt.tight_layout()
plt.savefig('halo_surface_density_tst_pix_'+str(halo_id)+'.png')
plt.close()
