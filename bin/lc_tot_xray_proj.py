from binascii import b2a_base64
from http.cookies import CookieError
from xmlrpc.server import MultiPathXMLRPCServer
import swiftsimio as sw
import numpy as np  
import numpy.ma as ma
import matplotlib.pyplot as plt 
from joey_interpolate_X_Ray_spectra_supermaster import interpolate_X_Ray as interp_xray
from mpl_toolkits.axes_grid1 import make_axes_locatable


import h5py
import healpy as hp
import unyt
from unyt import cm, erg, s, c,Mpc,kpc
from numba import jit

import lightcone_io.particle_reader as pr
from matplotlib.colors import LogNorm
from matplotlib.image import NonUniformImage

from astropy.cosmology import FlatLambdaCDM, z_at_value
import os
from tqdm import tqdm

from velociraptor import load as vl_load
from functools import reduce
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

'''
Compute spectra from the particle lightcone output
We only take part of the sky into account
Loop through the different lightcone output files
Find (based on pixels) which particles in the lightcone fall into that patch of sky
For those particles, interpolate xrays
With the interpolated xrays:
- divide by luminosity distance
- shift energies using redshift
- bin result into output spectral resolution
Add results for this lightcone file to overall spectrum for each lightcone pixel
Go to next file
'''

################################# setting global variables ###############################
# global input_filename,halo_id,nside,radius,vector,work_path,zoom_size,z_halo,z_halo_range,z_range,rest_line_bin,obs_bin,npix,SelectObsBin,PLOT,xsize
global input_filename, work_path
input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"
work_path = "."
# ## halo basics
# halo_id = '10976067'
# radius = np.radians(1) #deg
# vector = np.array([-0.49919698,-0.03709323,-0.86569421])

# ### healpy plotting basics
# nside=16384
# npix = hp.pixelfunc.nside2npix(nside)
# xsize = 300 # determine pixel number of final cartproj plot
# zoom_size=1  # fov [-zoom_size,zoom_size] in deg

# ### redshifts for plot titles
# # redshift in sim box
# z_range = np.array([0,0.1]) # z for tot & contam
# # redshift only for halo
# z_halo_range = np.array([0.0486,0.0495]) # z for pure
# # redshift of halo
# z_halo = np.median(z_halo_range)

# ### interpolate line bins
# line_E = 0.653 
# line_interval_lo = 0.001
# line_interval_hi = 0.001
# rest_line_bin = np.array([line_E-line_interval_lo,line_E+line_interval_hi])
# obs_bin = np.array([(rest_line_bin[0]-2*line_interval_lo)/(1+z_halo),rest_line_bin[1]/(1+z_halo)]) 

# ### bool parameters
# SelectObsBin=True  # whether selected emissions by observed bin
# PLOT=False  # whether plot others (not including xrays)

# ### create or enter a new directory
# work_path = '/cosma8/data/dp004/dc-chen3/narrow_emi_map/xray_tot_halo'+ halo_id

# if SelectObsBin==False:
#     dir = '/all'
#     work_path = work_path+dir
#     # Create a new directory because it does not exist 
#     if os.path.exists(work_path) is False:
#         os.makedirs(work_path)
#         os.makedirs(work_path+'/data')
#         os.makedirs(work_path+'/png')
#         print('directory '+"is created!")  
# else:
#     dir = '/obs_select'
#     work_path = work_path+dir
#     # Create a new directory because it does not exist 
#     if os.path.exists(work_path) is False:
#         os.makedirs(work_path)
#         os.makedirs(work_path+'/data')
#         os.makedirs(work_path+'/png')
#         print('directory '+"is created!")  

################################## define functions ##############################################

def dist(x1,y1,z1,x2,y2,z2):
        return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5

def compute_lc_coords(input_filename,xcoords,ycoords,zcoords):
    '''
    This function convert halo coordinate in snapshots into vector 
    when going to find the halo in lightcones

    Parameters
    -----------------------------
    input_filename: str
        input lightcone file
    
    xcoords, ycoords, zcoords: unyt array (1*halo num)
        cmbp coordinates of halo (comoving)

    Returns
    -----------------------------
    xcoords_w_r_t_LC1,ycoords_w_r_t_LC1,zcoords_w_r_t_LC1: np.array (1*3)
        vector to find halo in lightcone

    '''
    lc_str = input_filename.split('/')[-1].split('_')[0]
    if lc_str == 'lightcone0':
        xcoords_w_r_t_LC1 = (xcoords - 750)
        ycoords_w_r_t_LC1 = (ycoords - 750)
        zcoords_w_r_t_LC1 = (zcoords - 750)
    elif lc_str == 'lightcone1':
        xcoords_w_r_t_LC1 = (xcoords - 250)
        ycoords_w_r_t_LC1 = (ycoords - 250)
        zcoords_w_r_t_LC1 = (zcoords - 250)

    xcoords_w_r_t_LC1[xcoords_w_r_t_LC1 > 500] -= 1000
    ycoords_w_r_t_LC1[ycoords_w_r_t_LC1 > 500] -= 1000
    zcoords_w_r_t_LC1[zcoords_w_r_t_LC1 > 500] -= 1000

    xcoords_w_r_t_LC1[xcoords_w_r_t_LC1 < -500] += 1000
    ycoords_w_r_t_LC1[ycoords_w_r_t_LC1 < -500] += 1000
    zcoords_w_r_t_LC1[zcoords_w_r_t_LC1 < -500] += 1000

    return xcoords_w_r_t_LC1,ycoords_w_r_t_LC1,zcoords_w_r_t_LC1

def select_halo(input_filename,catalog_redshift,m200,RELAX):
    '''
    This function select halos in velociprator catalog at given redshift 

    Parameters
    ------------------------------
    input_filename: str
        input lightcone file

    catalog_redshift: float 
        catalogue's redshift for catalogue selecting

    m200: float, unit: 10^10 Msun
        lower limit of halo mass for selecting halo
    
    RELAX: bool
        if select only relaxed halo (distance btw cmbp and c < 100kpc)
        or select only unrelaxed halo (distance btw cmbp and c > 100kpc)

    Returns
    ------------------------------
    save index as halo_id
    
    '''
    
    print("selecting halo ... ")
    lc_str = input_filename.split('/')[-1].split('_')[0]

    cat_ind = str(int(77-np.round(catalog_redshift/0.05)))
    cata_loc="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/"
    cata_name=cata_loc+'catalogue_00'+cat_ind+'/vr_catalogue_00'+cat_ind+'.properties.0'
    catalogue = vl_load(cata_name)

    ##### select main halo
    hosthaloid=catalogue.ids.hosthaloid
    ind1 = np.where(hosthaloid==-1)

    ##### select high mass halo
    m200_crit=catalogue.masses.mass_200crit
    ind2 = np.where(m200_crit>m200)

    ##### whether select relaxed halo
    # halo center position
    xc = catalogue.positions.xc.to(Mpc).value
    yc = catalogue.positions.yc.to(Mpc).value
    zc = catalogue.positions.zc.to(Mpc).value
    # halo lowest potential center xcmbp
    xcmbp= catalogue.positions.xcmbp.to(Mpc).value
    ycmbp= catalogue.positions.ycmbp.to(Mpc).value
    zcmbp= catalogue.positions.zcmbp.to(Mpc).value
    dist_halo = dist(xc,yc,zc,xcmbp,ycmbp,zcmbp)

    if RELAX==True:
        ind3 = np.where(dist_halo<0.1)
    else:
        ind3 = np.where(dist_halo>0.1)

    r200m=catalogue.radii.r_200mean.to(Mpc).value
    ind_sel1 = reduce(np.intersect1d, (ind1,ind2,ind3))

    ### results
    ####### first selection w/o redshift
    xcmbp = xcmbp[ind_sel1]
    ycmbp = ycmbp[ind_sel1]
    zcmbp = zcmbp[ind_sel1]
    r200m = r200m[ind_sel1]
    ### select halo wholy in lightcone slice
    # read cosmology and derive halo's redshift interval range
    xcmbp_lc,ycmbp_lc,zcmbp_lc = compute_lc_coords(input_filename,xcmbp,ycmbp,zcmbp)
    # distance btw the halo and lightcone center (Mpc)    
    dist_lc = dist(xcmbp_lc,ycmbp_lc,zcmbp_lc,0,0,0)

    DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.306)  ## HYDRO_FIDUCIAL
    z_sh = 0.05+0.05*catalog_redshift
    dL = DESyr3.luminosity_distance(z_sh).value
    ##### combine all index
    ind_fin=ind_sel1[np.where((dist_lc+r200m)<dL)]
    print(ind_fin)
    ###### second selection w/ redshift and save output
    if RELAX==True:
        outfilename='haloid_m200c_'+str(m200)+'_1e10msun_z_'+str(z_sh)+'_'+lc_str+'_relaxed'
    else: 
        outfilename='haloid_m200c_'+str(m200)+'_1e10msun_z_'+str(z_sh)+'_'+lc_str+'_unrelaxed'

    np.save(work_path+'/'+outfilename, [int(i) for i in ind_fin])
    print(work_path+outfilename+" has been saved!")

def load_halo_properties(halo_id,catalog_redshift,SAVETXT):
    '''
    This function load and calculate halo properties 

    Parameters
    --------------------
    halo_id: int
        index of halo in velociprator catalog 
    
    SAVETXT: bool
        whether save halo properties to a txt
    '''

    cat_ind = str(int(77-np.round(catalog_redshift/0.05)))
    cata_loc="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/"
    cata_name=cata_loc+'catalogue_00'+cat_ind+'/vr_catalogue_00'+cat_ind+'.properties.0'
    catalogue = vl_load(cata_name)

    # load properties
    DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.306)  ## HYDRO_FIDUCIAL
    xcmbp= catalogue.positions.xcmbp.to(Mpc).value[halo_id]
    ycmbp= catalogue.positions.ycmbp.to(Mpc).value[halo_id]
    zcmbp= catalogue.positions.zcmbp.to(Mpc).value[halo_id]

    lc_str = input_filename.split('/')[-1].split('_')[0]
    
    def reset(x):
        if lc_str == 'lightcone0':
            x1 = x-750
        elif lc_str == 'lightcone1':
            x1 = x-250

        if x1>500:
            x1 -=1000
        elif x1<-500:
            x1+=1000
        return x1

    xcmbp_lc = reset(xcmbp)
    ycmbp_lc = reset(ycmbp)
    zcmbp_lc = reset(zcmbp)
    m200c=catalogue.masses.mass_200crit[halo_id]
    r200m=catalogue.radii.r_200mean[halo_id]
    r200c=catalogue.radii.r_200crit[halo_id]
    vector = np.array([xcmbp_lc,ycmbp_lc,zcmbp_lc])  

    ## calculate properties
    dist_lc = dist(xcmbp_lc,ycmbp_lc,zcmbp_lc,0,0,0)
    z_halo = z_at_value(DESyr3.comoving_distance,dist_lc*u.Mpc,zmax=0.5)
    z_max = z_at_value(DESyr3.comoving_distance,(dist_lc+r200m.value)*u.Mpc,zmax=0.5)
    z_min = z_at_value(DESyr3.comoving_distance,(dist_lc-r200m.value)*u.Mpc,zmax=0.5)
    z_halo_range = np.array([z_min,z_max])
    r200m_arcmin = DESyr3.arcsec_per_kpc_comoving(z_halo)/60*r200m*1000

    print("z_halo", "z_halo_range", "m200c.to(1e10*Msun)", "r200m.to(Mpc)", 'r200m_arcmin (arcmin)', "r200c.to(Mpc)",  "vector (Mpc)")
    print(z_halo.value, z_halo_range, m200c.to(1e10*unyt.Msun).value, r200m.to(Mpc), r200m_arcmin.value, r200c.to(Mpc),  vector)
    return z_halo, z_halo_range, m200c.to(1e10*unyt.Msun), r200m.to(Mpc).value, r200m_arcmin, r200c.to(Mpc).value,  vector


def compute_los_redshift(part_lc,vector):
    '''
    This func calculate line of sight velocity

    Parameters
    -----------------------------------
    part_lc: particle lightcone, lightcone_data["Velocities"]
        velocities of particles from extracted swift particle lightcone
    
    vector: np array 1x3
        Vector pointing at a spot on the sky (where halo is)
    
    Returns
    -----------------------------------
    z_los: float
        line of sight redshift
    '''
    vx = part_lc[:,0]
    vy = part_lc[:,1]
    vz = part_lc[:,2]
    vector_norm = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
    v_los = (vector[0]*vx+vector[1]*vy+vector[2]*vz)/vector_norm
    z_los = np.sqrt((1+v_los/c)/(1-v_los/c))-1
    return z_los

def compute_flux(part_lc,data):
    '''
    This function compute xray luminosity for particle fall in observed energy bin at z=0

    Parameters
    -----------------------
    part_lc: lightcone_data["Coordinates"]
        Coordinates of particles from extracted swift particle lightcone

    data: swift snapshot data, data

    Returns
    -----------------------
    flux_arr: np.array 1x particle num
        flux for particles fall in observable bin
    
    
    '''

    print('interpolating xrays')
    E_bin_lo = (rest_line_bin[0]/(1+data.gas.z_los)/(1+data.gas.redshifts)).value
    E_bin_hi = (rest_line_bin[1]/(1+data.gas.z_los)/(1+data.gas.redshifts)).value

    lum,__ = interp_xray(data.gas.densities, data.gas.temperatures, data.gas.smoothed_element_mass_fractions, data.gas.redshifts, data.gas.masses, fill_value = 0, bin_energy_lims = rest_line_bin)
    lum_arr = np.transpose(lum)

    if SelectObsBin==True:
        particle_fall_in_obs_bin  = 0
        particle_not_in_obs_bin  = 0
        particle_partly_in_obs_bin = 0
        lum_arr = np.zeros(np.shape(data.gas.redshifts))
        for i in range(len(lum_arr)):
            if E_bin_lo[i]>obs_bin[0] and E_bin_hi[i]<obs_bin[1]:
                particle_fall_in_obs_bin +=1
                lum_arr[i] = lum[i]
            elif E_bin_lo[i]<obs_bin[0] and E_bin_hi[i]>obs_bin[1]:
                particle_not_in_obs_bin +=1
            else:
                particle_partly_in_obs_bin +=1
        print("particle_fall_in_obs_bin is %d, particle_not_in_obs_bin is %d, particle_partly_in_obs_bin is %d"%(particle_fall_in_obs_bin,particle_not_in_obs_bin,particle_partly_in_obs_bin) )
    
    # compute luminosity distance
    distances = np.sqrt(part_lc[:, 0]**2 + part_lc[:, 1]**2 + part_lc[:, 2]**2)
    lum_distances = distances * (1 + data.gas.redshifts)
    flux_arr = lum_arr/(4*np.pi*lum_distances**2)
    print('E_bin_lo is ' + str([np.min(E_bin_lo),np.max(E_bin_lo)]))
    print('E_bin_hi is ' + str([np.min(E_bin_hi),np.max(E_bin_hi)]))

    return flux_arr


def load_snapshot():
    '''
    # At the moment the interpolation code expects swiftsimio objects, so we need to load a snapshot to get that
    # Then overwrite the relevant parts in update_data_structure function
    '''
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
    data.gas.z_los = compute_los_redshift(part_lc["Velocities"],vector)
    return data

def compute_mapdata(part_lc,property_data):
    '''
    This function computes map_data for properties, which is input for healpy cartproj plotting

    Parameters
    ---------------------------
    part_lc: lightcone_data["Coordinates"].value
        Coordinates of particles from extracted swift particle lightcone
     
    property_data: numpy array
        data lst for plotting

    Returns
    ---------------------------
    map_data/pix_a: np.array (npix * 1)
        map data list of properties, input for healpy cartproj plotting, which has already divided by pixel area
        unit: property unit/arcmin^2

    '''

    pix = hp.pixelfunc.vec2pix(nside, part_lc[:,0], part_lc[:,1], part_lc[:,2])
    # calculate resolution and area for each pixel
    pix_a = (2*60/xsize)**2
    dat = property_data
    if len(np.shape(dat))>1:
        dat = property_data.T[0]   
    map_data = np.zeros(npix)
    np.add.at(map_data, np.array(pix), np.array(dat)/pix_a)

    return map_data


def plot_all(map_data,property_name,property_units,redshift_range,plot_settings):
    '''
    This function is for plotting (imshow) surface brightness values

    Parameters
    ------------------------------
    map_data_lst: np.array (npix * properties num)
        map data list of properties, output from compute_mapdata
        unit: property unit/arcmin^2

    property_name: list (str)
        list contain data for plotting

    property_units: list (str)
        units for properties

    redshift_range: np.array (1x2)
        describe data's redshift_range, used in data's title

    plot_settings: dict
        {"data_filter":[1e33,1e35], "cmap":"binary"}
    Returns:
    ------------------------------
        
    plot lots of imshow plots: 

    '''
    cartproj = hp.projector.CartesianProj(
                lonra=[-zoom_size, zoom_size], latra=[-zoom_size, zoom_size], 
                rot=hp.vec2ang(np.array(vector),lonlat=True),xsize=xsize
            )
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    pR = cartproj.projmap(map_data, lambda x, y, z: hp.vec2pix(nside, x, y, z))
    
    cmap = plot_settings['cmap']
    if len(plot_settings['data_filter'])>1:
        vmin = plot_settings['data_filter'][0]
        vmax = plot_settings['data_filter'][1]
        vnorm = LogNorm(vmin,vmax)
    else:
        vnorm = LogNorm()
    
    # plt.hist(pR)
    im = ax.imshow(pR, norm=vnorm,cmap=cmap, extent=[-60,60,-60,60])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(property_name+' ('+property_units+')'+
        '\n rest line '+str(rest_line_bin)+' received line [%4f,%4f](keV)'%(obs_bin[0],obs_bin[1]))

    ax.set_xlabel('RA (arcmin)')
    ax.set_ylabel('DEC (arcmin)')


    plt.tight_layout()
    plt.savefig(work_path+'/png/'+property_name+'_halo_'+halo_id+
            '_z_'+str(redshift_range[0])+'_'+str(redshift_range[1])+'.png'
        )
    plt.close()
    print('finish plotting '+property_name)
    del map_data

def compute_xray(input_filename,vector,radius,redshift_range,PLOT):
    '''
    This function first read lightcone file, then computes xray flux for each particle
    if PLOT==True, plot all parameters' plots except for xrays

    Parameters
    -----------------------------
    input_filename: str
        The input particle lightcone file (has been sorted by lightconeio)

    vector: np.array 1x3
        Vector pointing at a spot on the sky

    radius: float 
        Angular radius around that spot (radian)

    redshift_range: np.array 1x2
        redshift range 

    PLOT: boolen 
        if PLOT==True, plot all parameters' plots except for xrays
    
    Returns
    ----------------------------
    flux_map: np.array (npix * 1)
        xray flux of particles falls in obs bin, binned by pixels and divided by pixel area
        output to calculate contam

    coor: np.array (3*particle num)
        coordinates from lightcone_data, output for plotting xray

    '''
    print("loading lightcone")
    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename)

    # Read in the particle positions and masses
    property_names = ("Coordinates","Masses", "SmoothedElementMassFractions", "Densities", "Temperatures", "ExpansionFactors","Velocities")
    lightcone_data = lightcone["Gas"].read_exact(property_names=property_names,
                                redshift_range=redshift_range,
                                vector=vector, radius=radius,)

    # Put relevant parts into snapshot data structure
    data = load_snapshot()
    data = update_data_structure(data, lightcone_data)
    nr_particles = len(data.gas.redshifts)
    print("Total number of particles added to map = %d" % nr_particles)
    coor = np.array(lightcone_data["Coordinates"].value)
    flux_arr = compute_flux(coor,data)

    if PLOT==True:
        print("plotting others ...")
        # plot mass, density, temperature, metal, z_los 
        properties_data=[data.gas.masses, data.gas.densities,data.gas.temperatures,data.gas.smoothed_element_mass_fractions.oxygen,data.gas.redshifts,data.gas.z_los]
        properties_name = ["mass","density", "temperature", "Oxygen abundance", "particle_redshift","line_of_sight_redshift"]
        properties_units = [r'10$M_{sun}/arcmin^2$',r'6.8e-31$g/cm^3/arcmin^2$',r'$K/arcmin^2$',r'$/arcmin^2$',r'$/arcmin^2$',r'$/arcmin^2$']
        for i in range(len(properties_name)):
            map_data = compute_mapdata(coor,properties_data[i].value)
            plot_settings = {"data_filter":[0], "cmap":"binary"}
            plot_all(map_data,properties_name[i],properties_units[i],redshift_range,plot_settings)
            del map_data
        print("plotting others finishes!")
    return flux_arr,coor

def main(check_filename,redshift_range):
    '''
    This is function for calculating xray map data for one property and plotting

    Parameters
    ---------------------------------------
    check_filename: lst (str)
        lst of pure & tot map_data name

    redshift_range: np.array (1x2)
        describe data's redshift_range, used in data's title

    Returns
    ---------------------------------------
    save xray flux as npz

    plot xray

    '''
    if os.path.exists(work_path+'/data/'+check_filename+'.npz') is False:
        # compute pure xrays
        print("computing " + check_filename)
        flux,coor = compute_xray(input_filename,vector,radius,redshift_range,PLOT)
        np.savez_compressed(work_path+'/data/'+check_filename+'.npz',flux=flux,coor=coor)
        print(work_path+'/data/'+check_filename+'.npz  has been saved!')
    else:
        print("loading " + check_filename)
        data = np.load(work_path+'/data/'+check_filename+'.npz')
        flux = data['flux']
        coor = data['coor']
    print("plotting " + check_filename)
    flux_map = compute_mapdata(coor,flux)
    plot_settings = {"data_filter":[1e30,1e36], "cmap":"binary"}
    plot_all(flux_map,check_filename,r'erg/s/$cm^3/s/arcmin^2$',redshift_range,plot_settings)





# # ##### compute, save and plot pure and total data, if has computed pure and tot before, just load the data
# main('xray_flux_pure',z_halo_range)
# main('xray_flux_tot',z_range)

# ##### compute, plot contam data (saving takes lots of time, so didn't save contam)
# print('computing contam ... ')
# data_pure = np.load(work_path+'/data/'+'xray_flux_pure'+'.npz')
# data_tot = np.load(work_path+'/data/'+'xray_flux_tot'+'.npz')
# flux_pure = data_pure['flux']
# # print(np.shape(data_pure['flux']))
# # print(np.shape(flux_pure))
# # print(np.count_nonzero(flux_pure))

# pure_coor = data_pure['coor']
# flux_tot = data_tot['flux']
# # print(np.shape(data_tot['flux']))
# # print(np.shape(flux_tot))
# # print(np.count_nonzero(flux_tot))
# tot_coor = data_tot['coor']
# # calculate 2 map directly and substract them
# flux_map_tot = compute_mapdata(tot_coor,flux_tot)
# flux_map_pure = compute_mapdata(pure_coor,flux_pure)
# flux_map = flux_map_tot-flux_map_pure
# # print(np.count_nonzero(flux_map))
# plot_settings = {"data_filter":[1e30,1e36], "cmap":"binary"}
# plot_all(flux_map,'xray_contam',r'erg/s/$cm^3/s/arcmin^2$',z_range,plot_settings) 
# del flux_map_pure,flux_map_tot,flux_map,data_pure,data_tot,flux_pure,pure_coor,flux_tot,tot_coor


#### test select halo
# select_halo(input_filename,0,1e4,RELAX=True)
load_halo_properties(10976067,0,SAVETXT=True)

 