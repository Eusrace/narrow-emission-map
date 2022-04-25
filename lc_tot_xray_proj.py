from pyexpat.errors import XML_ERROR_BINARY_ENTITY_REF
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

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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
# Some default styling for the figures; best solution is once at the beginning of the code
# See https://matplotlib.org/3.1.3/tutorials/introductory/customizing.html
# These settings assume that you have used import matplotlib.pyplot as plt 

# Smallest font size is a 10 point font for a 4 inch wide figure. 
# font sizes and figure size are scaled by a factor 2 to have a large figure on the screen

SMALL_SIZE = 10*2                                        
MEDIUM_SIZE = 12*2
BIGGER_SIZE = 14*2

plt.rc('font', size=SMALL_SIZE, family='serif')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE, direction='in')    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE, direction='in')    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)                  # fontsize of the figure title
# plt.rc('figure', figsize='8, 6')                         # size of the figure, used to be '4, 3' in inches

####### setting basic globals #########

global input_filename
# mid res 1Gpc
input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"
# hi res 1Gpc
# input_filename = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/L1000N3600/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"


#################### function provides global variables ######################
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

def select_halo(input_filename,catalog_redshift,m200_lo,m200_hi,RELAX):
    '''
    This function select halos in velociprator catalog at given redshift 

    Parameters
    ------------------------------
    input_filename: str
        input lightcone file

    catalog_redshift: float 
        catalogue's redshift for catalogue selecting

    m200_lo, m200_hi: float, unit: 10^10 Msun
        lower and upper limit of halo mass for selecting halo
    
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
    ind2 = np.where((m200_crit>m200_lo) & (m200_crit<m200_hi))

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
        outfilename='haloid_m200c_'+str(m200_lo)+'_'+str(m200_hi)+'_1e10msun_z_'+str(z_sh)+'_'+lc_str+'_relaxed'
    else: 
        outfilename='haloid_m200c_'+str(m200_lo)+'_'+str(m200_hi)+'_1e10msun_z_'+str(z_sh)+'_'+lc_str+'_unrelaxed'

    np.save(outfilename, [int(i) for i in ind_fin])
    print(outfilename+" has been saved!")

def load_halo_properties(halo_id,catalog_redshift):
    '''
    This function load and calculate halo properties 

    Parameters
    --------------------
    halo_id: int
        index of halo in velociprator catalog 

    Returns 
    --------------------
    z_halo, z_halo_range, m200c.to(1e9*unyt.Msun), r200m.to(Mpc).value, r200m_arcmin, r200c.to(Mpc).value,  vector
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


################################# setting halo global variables ###############################
global halo_id,nside,radius,vector,work_path,zoom_size,z_halo,z_halo_range,z_range,rest_line_bin,npix,PLOT,xsize,pix_a,cart_proj,obs_bin_rest,obs_bin_received

## halo basics
halo_id = '10976067'
z_halo, z_halo_range, __,__, __,__, vector = load_halo_properties(int(halo_id),0)

### healpy plotting basics
nside=16384
radius = np.radians(1) #deg
npix = hp.pixelfunc.nside2npix(nside)
xsize = 120 # determine pixel number of final cartproj plot
zoom_size=1  # fov [-zoom_size,zoom_size] in deg
cartproj = hp.projector.CartesianProj(
                lonra=[-zoom_size, zoom_size], latra=[-zoom_size, zoom_size], 
                rot=hp.vec2ang(np.array(vector),lonlat=True),xsize=xsize
            )
pix_a = (2*zoom_size*60/xsize)**2 # calculate resolution and area for each pixel

### redshifts for plot titles
# redshift in sim box
z_range = np.array([0,0.1]) # z for tot & contam

### interpolate line bins
line_E = 0.653 
line_interval_lo = 0.15
line_interval_hi = 0.15
rest_line_bin = np.array([line_E-line_interval_lo,line_E+line_interval_hi])
# obs line bin
obs_bin_rest = np.array([0.651,0.655]) 
obs_bin_received = obs_bin_rest/(1+z_halo.value)

### create or enter a new directory
work_path = '/cosma8/data/dp004/dc-chen3/narrow_emi_map/xray_tot_halo'+ halo_id
dir = '/'+str(line_E-line_interval_lo)+"-"+str(line_E+line_interval_hi)
work_path = work_path+dir
# Create a new directory because it does not exist 
if os.path.exists(work_path) is False:
    os.makedirs(work_path)
    os.makedirs(work_path+'/data')
    os.makedirs(work_path+'/png')
    print('directory '+"is created!")  


################################## functions may need global vars ##############################################

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

    lum,restframe_energy = interp_xray(data.gas.densities, data.gas.temperatures, data.gas.smoothed_element_mass_fractions, data.gas.redshifts, data.gas.masses, fill_value = 0, bin_energy_lims = rest_line_bin)
    lum_arr = np.transpose(lum)
    print(restframe_energy)
    
    # compute luminosity distance
    distances = np.sqrt(part_lc[:, 0]**2 + part_lc[:, 1]**2 + part_lc[:, 2]**2)
    lum_distances = distances * (1 + data.gas.redshifts)
    flux_arr = lum_arr/(4*np.pi*lum_distances**2)
    print(np.shape(lum_arr))
    print(np.shape(lum_distances))
    print('E_bin_lo is ' + str([np.min(E_bin_lo),np.max(E_bin_lo)]))
    print('E_bin_hi is ' + str([np.min(E_bin_hi),np.max(E_bin_hi)]))

    return flux_arr,restframe_energy


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

def compute_mapdata(pix,property_data):
    '''
    This function computes map_data for properties, which is input for healpy cartproj plotting

    Parameters
    ---------------------------
    pix: np.array (pix number for all particles * 1)
        pixel calculated from each particle's coordinates, by healpy
     
    property_data: numpy array
        data lst for plotting

    Returns
    ---------------------------
    map_data: np.array (pixel number for all sky map * 1)
        map data list of properties, input for healpy cartproj plotting
        unit: property unit/arcmin^2

    '''

    dat = property_data
    if len(np.shape(dat))>1:
        dat = property_data.T[0]   
    map_data = np.zeros(npix)
    try:
        np.add.at(map_data, np.array(pix), np.array(dat))
    except:
        print(np.shape(pix),np.shape(dat))
    return map_data


def plot_all(pR,property_name,property_units,property_cmap,redshift_range,plot_settings):
    '''
    This function is for plotting (imshow) surface brightness values

    Parameters
    ------------------------------
    map_data: np.array (1 x npix)
        properties' map data, output from compute_mapdata
        unit: property unit/arcmin^2

    property_name: list (str)
        list contain data for plotting

    property_units: list (str)
        units for properties

    property_cmap: list (str)
        define color bar's color for properties
    
    redshift_range: np.array (1x2)
        describe data's redshift_range, used in data's title

    plot_settings: dict
        {"data_filter":[1e33,1e35], "mode":"lin"/"log"}
    
    Returns:
    ------------------------------
        
    plot lots of imshow plots: 

    '''
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    mode = plot_settings['mode']
    extent = np.array([-zoom_size,zoom_size,-zoom_size,zoom_size])*60
    if mode =='log':
        if len(plot_settings['data_filter'])>1:
            vmin = plot_settings['data_filter'][0]
            vmax = plot_settings['data_filter'][1]
            # pR[(pR<vmin) | (pR>vmax)]=0
            im = ax.imshow(pR, norm=LogNorm(vmin,vmax),cmap=property_cmap, extent=extent)
        else:
            vmin = np.sort(pR[pR>0], axis=None)[2]
            vmax = np.max(pR)
            # pR[(pR<vmin) | (pR>vmax)]=0
            im = ax.imshow(pR, norm=LogNorm(vmin,vmax),cmap=property_cmap, extent=extent)
    else:
        if len(plot_settings['data_filter'])>1:
            vmin = plot_settings['data_filter'][0]
            vmax = plot_settings['data_filter'][1]
            # pR[(pR<vmin) | (pR>vmax)]=0
            im = ax.imshow(pR, vmin = vmin,vmax = vmax,cmap=property_cmap, extent=extent)
        else:
            vmin = np.min(map_data)
            vmax = np.max(map_data)
            # pR[(pR<vmin) | (pR>vmax)]=0
            im = ax.imshow(pR, vmin = vmin,vmax = vmax,cmap=property_cmap, extent=extent)
    print(vmin,vmax)
    print(mode)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(property_name+' ('+property_units+')')

    ax.set_xlabel('RA (arcmin)')
    ax.set_ylabel('DEC (arcmin)')
    r = np.arange(1,61,1)
    for i in range (len(r)-1):
        circle = plt.Circle((0, 0), r[i], color='r',fill=False)
        ax.add_patch(circle)
    plt.tight_layout()
    plt.savefig(work_path+'/png/'+property_name+'_radialannu_1arcmin_halo_'+halo_id+
            '_z_'+str(redshift_range[0])+'_'+str(redshift_range[1])+'.png'
        )
    plt.close()

    

    

    ###### plot hist only for debugging (it will be slow)
    # fig, ax = plt.subplots(2, 2, figsize = (12, 6))
    # ax[0,0].hist(map_data,range=(vmin,vmax))#,bins = np.linspace(vmin,vmax,num=200))
    # ax[0,0].set_title("map_data filtered")
    # ax[0,1].hist(pR,range=(vmin,vmax))#,bins = np.linspace(vmin,vmax,num=200))
    # ax[0,1].set_title("pR filtered")
    # ax[1,0].hist(map_data)#,np.linspace(np.min(map_data),np.max(map_data),num=200))
    # ax[1,0].set_title("map_data all")
    # ax[1,1].hist(pR)#,np.linspace(np.min(map_data),np.max(map_data),num=200))
    # ax[1,1].set_title("pR all")
    # for a in ax.flat:
    #     if mode == "log":
    #         a.set_xscale("log")
    #         a.set_yscale("log")
    #     else:
    #         a.set_xscale("linear")
    #         a.set_yscale("linear")
    #     a.label_outer()
    # plt.savefig(work_path+'/png/'+property_name+'_hist.png')
    # plt.close()

    print('finish plotting '+property_name)
    del map_data

def compute_xray(input_filename,vector,radius,redshift_range,PLOT,COMPUTE,SEL_SPEC_PARTICLE):
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
    
    COMPUTE: boolen
        if COMPUTE==Ture, compute xray flux 
    
    SEL_SPEC_PARTICLE: boolen
        if SEL_SPEC_PARTICLE==True, select particles in certain temperature band to plot spectrum
        and output particle index and its properties for plot_spec func
    
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
    property_names = ("Coordinates","Masses", "SmoothedElementMassFractions", "Densities", "Temperatures", "ExpansionFactors","Velocities","ParticleIDs")
    lightcone_data = lightcone["Gas"].read_exact(property_names=property_names,
                                redshift_range=redshift_range,
                                vector=vector, radius=radius,)

    # Put relevant parts into snapshot data structure
    data = load_snapshot()
    data = update_data_structure(data, lightcone_data)
    nr_particles = len(data.gas.redshifts)
    print("Total number of particles added to map = %d" % nr_particles)
    coor = np.array(lightcone_data["Coordinates"].to(unyt.cm).value)
    
    if COMPUTE==True:
        print("calculating xray ...")
        pix = hp.pixelfunc.vec2pix(nside, coor[:,0], coor[:,1], coor[:,2])
        flux_arr,restframe_energy = compute_flux(coor,data)
        return flux_arr,pix,restframe_energy,data.gas.redshifts.value,data.gas.z_los.value

    if PLOT==True:
        print("plotting others ...")
        # count how many particles fall in each pixel, divide the property value by praticle number in each pixel 
        pix = hp.pixelfunc.vec2pix(nside, coor[:,0], coor[:,1], coor[:,2])
        __,idx,counts = np.unique(pix,return_inverse=True,return_counts=True) 
        weights = np.ones(np.shape(pix))/counts[idx]
        # plot mass & abundance averaged by pixel area; temperature, particle redshifts and los redshifts averaged by particle number in each pixel; particles counts map
        properties_data = [data.gas.masses/(unyt.Msun*1e10)/pix_a,data.gas.temperatures*weights,data.gas.smoothed_element_mass_fractions.oxygen*data.gas.masses/(unyt.Msun*1e10)/pix_a,data.gas.redshifts*weights,data.gas.z_los*weights,counts[idx]]
        properties_name = ["mass", "temperature", "Oxygen_abundance","particle_redshift","line_of_sight_redshift","particle_counts"]
        properties_units = [r'$10^{10} M_{sun}/arcmin^2$','K',r'$10^{10} M_{sun}/arcmin^2$',' ',' ',' ']
        properties_cmaps = ['cividis','hot','summer','coolwarm','coolwarm','Paired']
        plot_settings =[{"data_filter":[],"mode":"log"},{"data_filter":[],"mode":"log"},{"data_filter":[],"mode":"log"},{"data_filter":[0.04,0.06],"mode":"lin"},{"data_filter":[],"mode":"lin"},{"data_filter":[1,np.max(counts[idx])],"mode":"log"}]

        for i in range(len(properties_name)):
            map_data = compute_mapdata(pix,properties_data[i])
            plot_all(map_data,properties_name[i],properties_units[i],properties_cmaps[i],redshift_range,plot_settings[i])
            del map_data
        print("plotting others finishes!")
    
    if SEL_SPEC_PARTICLE==True:
        '''
        1. choose one particle from each temperture range for plotting spectrum
        3. print the particle's properties
        '''
        T_bins = [1e4,1e5,1e6,1e7,1e8,1e9]
        idx_arr,temp,abun,mass,reds,particleid = [[] for i in range(6)]

        for i in range(len(T_bins)-1):
            T_msk = (T_bins[i]<data.gas.temperatures) & (data.gas.temperatures<T_bins[i+1])
            idx1 = np.arange(nr_particles)[T_msk]
            if len(idx1)>0:
                idx = int(idx1[2])
                idx_arr.append(idx)
                temp.append(data.gas.temperatures.value[idx])
                abun.append(data.gas.smoothed_element_mass_fractions.oxygen[idx].value*(data.gas.masses[idx].to(unyt.Msun).value))
                mass.append(data.gas.masses[idx].to(unyt.Msun).value)
                reds.append(data.gas.redshifts.value[idx])
                particleid.append(lightcone_data["ParticleIDs"][idx])
            else:
                continue
        np.savez_compressed(work_path+'/data/'+'spec_particle',idx=idx_arr,mass=mass,temp=temp,abun=abun,reds=reds)
        print(np.array(idx_arr))
        print(particleid)
        print('Particles for plotting spectrum have been selected! \n Properties save as spec_particle.npz in /data subdirectory')

def particle_spec(flux_filename,idx_filename):
    '''
    This function plot spectrum of one particle

    Parameters
    ---------------------------------------
    flux_filename: npz
        file name of flux_file containing flux which size:(particle_num, joey bins), coor, restframe_energy
    
    idx_filename: npz
        file name of selected particles (for plotting spectrums) , containing their properties & index

    Returns
    ---------------------------------------
    plots of spectrums of each particle at each temperature
    '''
    flux_file = np.load(work_path+"/data/"+flux_filename)
    flux = flux_file['flux']
    restframe_energy = flux_file['restframe_energy']
    idx_file = np.load(work_path+"/data/"+idx_filename)
    idx_arr = idx_file['idx']
    temp = idx_file['temp']
    mass = idx_file['mass']
    abun = idx_file['abun']

    for i,idx in enumerate(idx_arr):
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        ax.stairs(flux[idx],edges=restframe_energy)
        ax.set_ylabel('xray flux for each bin $(erg/s/cm^2)$')
        ax.set_xlabel('rest frame energy (keV)')
        ax.set_title("spectrum of particle "+str(idx)+" \ntemperature is "+ "{:.2e}".format(temp[i])+" K mass is "+ "{:.2e}".format(mass[i])+" Msun abun is "+ "{:.2e}".format(abun[i])+" Msun")
        if np.count_nonzero(flux[:,idx])>0:
            ax.set_yscale("log")
        plt.savefig(work_path+"/png/"+"spectrum_of_particle_"+str(idx))
        plt.close()
        print("spectrum_of_particle_"+str(idx)+" has been plotted!")

def sel_flux(obs_bin_received,restframe_energy,flux,z_par,z_los,mode):
    '''
    This function select emission that will fall in observable bins (for plotting total, continuum)

    Parameters
    -------------------------
    obs_bin_received: np.array (1x2)
        energy bin observer received at z=0
    
    restframe_energy: np.array
        restframe_energy bin that has been fixed when interpolating xray (bin size fixed in supermaster)
    
    flux: 
        output of computing xray
    
    z_par: np.array
        particle redshifts
    
    z_los: np.array
        line of sight redshifts
    
    mode: str 
        if mode=="sum", calculate sum flux in this energy range
        if mode=="med", calculate median flux in this energy range

    Returns
    -------------------------
    flux: np.array (1xparticle num)
        xray flux for each particle in certain energy range
    
    '''

    ### for every particle
    restE_lo = obs_bin_received[0]*(1+z_par)*(1+z_los)
    restE_hi = obs_bin_received[1]*(1+z_par)*(1+z_los)
    
    ### for total emission
    flux_inbin,tstE_lo,tstE_hi = [np.zeros(len(restE_lo)) for i in range(3)]
    print("selecting flux ...")
    print(obs_bin_received) # both obs_bin_received and restframe_energy are np.array (keV), not unyt array
    print(np.min(restE_lo),np.max(restE_hi))

    for i in tqdm(range(len(z_par))):
        idx1 = np.arange(len(restframe_energy))
        idx = idx1[(restframe_energy-restE_lo[i]>=0) & (restframe_energy-restE_hi[i]<=0)]
        if len(idx)==0:
            continue
        else:
            if mode =='sum':
                flux_inbin[i]=np.sum(flux[i][idx])
                # flux_inbin[i]=np.sum(flux[:,i][idx])
            if mode =='med':
                flux_inbin[i]=np.median(flux[:,i][idx])
            tstE_lo[i] = restframe_energy[idx[0]]
            tstE_hi[i] = restframe_energy[idx[-1]]
    print(idx) # check there are many bins for continuum 
    print("particle energy ranges due to redshifts ",[np.min(tstE_lo),np.max(tstE_hi)])

    return flux_inbin


def pixel_spec(flux_filename,input_radec):
    '''
    This function draw spectrum for one or more pixels

    Parameters
    --------------------------------
    flux_filename: npz 
        npz file saved flux
    
    input_radec : list 1x4
        the ra,dec region for pixels to plot, unit is arcmin [ra1,ra2,dec1,dec2]

    Returns
    --------------------------------
    plots for pixel's spectrums

    '''
    # identify which particle will fall into corresponding pixel according to pixel coordinate
    flux_file = np.load(work_path+"/data/"+flux_filename)
    flux = flux_file['flux']
    restframe_energy = flux_file['restframe_energy']
    z_par = flux_file['z_par']
    z_los = flux_file['z_los']
    
    if os.path.exists(work_path+"/data/particles_xy_xsize"+str(xsize)+".npy") is False:
        print("computing xy ...")
        hp_pix = flux_file['pixel']
        xy = []
        for pix in hp_pix:
            xy.append(cartproj.vec2xy(hp.pix2vec(nside,pix)))
        np.save(work_path+"/data/particles_xy_xsize"+str(xsize),xy)
    else:
        print("loading xy ...")
        xy = np.load(work_path+"/data/particles_xy_xsize"+str(xsize)+".npy")

    input_radec = np.array(input_radec)
    print(np.shape(xy))
    input_xy = input_radec/60
    sel_idx = np.where((xy[:,0]>=input_xy[0]) & (xy[:,0]<=input_xy[1]) & (xy[:,1]>=input_xy[2]) & (xy[:,1]<=input_xy[3]))
    sel_idx = sel_idx[0]
    print("particle number in this region is",np.shape(sel_idx))
    # plot the input region
    print("plotting input region ... ")
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    extent = np.array([-zoom_size,zoom_size,-zoom_size,zoom_size])*60
    pR = np.zeros((xsize,xsize))
    pR[int(input_xy[2]):int(input_xy[3])][:,int(input_xy[0]):int(input_xy[1])]=1
    im = ax.imshow(pR,cmap="binary", extent=extent)
    ax.set_xlabel('RA (arcmin)')
    ax.set_ylabel('DEC (arcmin)')
    plt.tight_layout()
    plt.savefig(work_path+'/png/'+'pix.png')
    plt.close()

    # plot spectrum
    print("plotting spectrum of the region ...")

    tot,contin1,contin2 = [0 for i in range(3)]
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    Emin = restframe_energy[0]/(1+np.max(z_par[sel_idx]))/(1+np.max(z_los[sel_idx]))
    Emax = restframe_energy[-1]
    Enewbin = np.linspace(Emin,Emax,num=160)
    tot_spec = np.zeros(160)
    for idx in sel_idx:
        received_energy = np.array(restframe_energy)/(1+z_par[idx])/(1+z_los[idx])
        ax.stairs(flux[idx],edges=received_energy)
        Eidx = np.searchsorted(Enewbin, received_energy[0:-1], side='left')
        np.add.at(tot_spec,Eidx,flux[idx])
        # add and plot spectrum level
        msk_tot = np.arange(len(received_energy))[(received_energy<obs_bin_received[1]) & (received_energy>obs_bin_received[0])]
        msk_contin1 = np.arange(len(received_energy))[(received_energy>obs_bin_received[0]-0.04) & (received_energy<obs_bin_received[0])]
        msk_contin2 = np.arange(len(received_energy))[(received_energy<obs_bin_received[1]+0.04) & (received_energy>obs_bin_received[1])]
        tot += np.sum(flux[idx][msk_tot])
        contin1 += np.median(flux[idx][msk_contin1])
        contin2 += np.median(flux[idx][msk_contin2])
    
    ax.step(Enewbin,tot_spec,label="sum spectrum")
    ax.plot(obs_bin_received,tot*np.ones(2),'r',label = 'total emission in OVIII bin')
    ax.plot([obs_bin_received[0]-0.04,obs_bin_received[0]],contin2*np.ones(2),'k',label = 'continuum1')
    ax.plot([obs_bin_received[1],obs_bin_received[1]+0.04],contin2*np.ones(2),'k',label ='continuum2')
    ax.plot([obs_bin_received[0]-0.04,obs_bin_received[1]+0.04],(contin1+contin2)/2*np.ones(2),'g--',label = 'continuum')
    

    # plt.tight_layout()
    ax.set_yscale('log')
    plt.legend(loc ="upper right")
    ax.set_xlabel("received energy, keV")
    ax.set_ylabel(r"flux, $erg/s/cm^2$")
    plt.savefig(work_path+'/png/'+'pix_spec.png')
    plt.close()

def smooth_fluxmap(input_fluxmap):
    '''
    This function smooth input flux map with cubic gaussian kernel

    Parameter
    ---------------------------------
    input_fluxmap: np.array (xsize^2)
        healpix fluxmap matrix

    Returns
    ---------------------------------
    output_fluxmap: np.array (xsize^2)
        smoothed healpix fluxmap

    '''
    from scipy.ndimage import gaussian_filter
    output_fluxmap = gaussian_filter(input_fluxmap, sigma=1)

    return output_fluxmap

def plot_profile(input_fluxmap,annulus_radius):
    '''
    This function plot radial profile 

    Parameters
    --------------------
    input_fluxmap: np.array (xsize^2)
        healpix fluxmap matrix

    annulus_radius: float
        in arcmin
    Returns
    --------------------
    annu_arcmin: np.array (1 x num of annulus)
        annulus in arcmin

    profile: np.array (1 x num of annulus)
        sum xray luminosity per arcmin2 at annuluses

    '''
    center = np.array([0,0])
    img_c = center+(xsize/2)
    # how many pixels per annulus
    pix_step = annulus_radius/(zoom_size*2*60/xsize)
    pix_annu = np.arange(0,xsize/2+pix_step,pix_step)+xsize/2
    x = np.arange(-60,60)
    y = np.arange(-60,60)
    xs, ys = np.meshgrid(x, y, sparse=True)
    zs = np.sqrt(xs**2 + ys**2)
    lum_arr = []
    pix_num = []
    for i in tqdm(range(len(pix_annu)-1)):
        lum1 = np.sum(input_fluxmap[zs<pix_annu[i]])
        lum2 = np.sum(input_fluxmap[zs<pix_annu[i+1]])
        lum_arr.append(lum2-lum1)
        pix_num.append(np.count_nonzero(zs<pix_annu[i]))
    annu_arcmin = np.arange(0,zoom_size*60+annulus_radius,annulus_radius)

    return annu_arcmin, np.array(lum_arr),pix_num

def flux_pdf(pR_pure,pR_tot,pR_redcontin):
    '''
    This function plots pdf of flux or flux err

    Parameters
    --------------------------
    projected flux_map_*: np.array, xsize * xsize
        healpix map
    
    Returns
    --------------------------


    '''
    # plot histogram of error vs flux
    err = abs(pR_redcontin-pR_pure)
    err_1d = err.flatten()
    flux_pure_1d = pR_pure.flatten()
    flux_tot_1d = pR_tot.flatten()

    bin_num = 500
    bin = np.logspace(-30,-13,num=bin_num,endpoint=True)
    
    pure_idx = np.searchsorted(bin,flux_pure_1d, side='left')
    idx_arr,idx,counts = np.unique(pure_idx,return_inverse=True,return_counts=True) 
    cts = counts[idx]
    err_1d_norm = err_1d/cts
    arr = []
    for i in range(bin_num):
        arr.append(np.sum(err_1d_norm[idx==i]))
    pure_arr = np.array(arr)

    tot_idx = np.searchsorted(bin,flux_tot_1d, side='left')
    __,idx,counts = np.unique(tot_idx,return_inverse=True,return_counts=True) 
    cts = counts[idx]
    err_1d_norm = err_1d/cts
    arr = []
    for i in range(bin_num):
        arr.append(np.sum(err_1d_norm[idx==i]))
    tot_arr = np.array(arr)

    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    x_bin = np.median(np.stack((bin[0:-1],bin[1::])),axis=0)
    # ax.stairs(pure_arr[1::]/(bin[1::]-bin[0:-1]),bin,color = "b",label='bin by pure flux',alpha =0.5)
    # ax.stairs(tot_arr[1::]/(bin[1::]-bin[0:-1]),bin,color = "r", label='bin by total flux',alpha=0.5)
    y_pure = pure_arr[1::]/(bin[1::]-bin[0:-1])
    y_tot = tot_arr[1::]/(bin[1::]-bin[0:-1])
    print(y_pure)
    # y_pure_sf = savgol_filter(y_pure,80, 2)
    # y_tot_sf = savgol_filter(y_tot, 80, 2)
    # f_pure = interp1d(x_bin[y_pure>0], y_pure[y_pure>0], kind='cubic')
    # f_tot = interp1d(x_bin[y_tot>0], y_tot[y_tot>0], kind='cubic')

    ax.plot(x_bin, y_pure,'o',color='b',alpha=0.8)
    ax.plot(x_bin, y_tot, 'o',color='r',alpha=0.8)
    # ax.plot(x_bin[y_pure>0], y_pure[y_pure>0],'o--',color='b',alpha=0.6)
    # ax.plot(x_bin[y_tot>0], y_tot[y_tot>0], 'o--',color='r',alpha=0.6)
    # ax.plot(x_bin[y_pure>0], f_pure(x_bin[y_pure>0]),'o--',color='b',alpha=0.8)
    # ax.plot(x_bin[y_tot>0], f_tot(x_bin[y_tot>0]), 'o--',color='r',alpha=0.8)
    # ax.fill_between(x_bin, y_pure_sm*1.3, y_pure_sm*0.7,color = "b",alpha = 0.3)
    # ax.fill_between(x_bin, y_tot_sm*1.3, y_tot_sm*0.7,color = "r",alpha = 0.3)
    ax.set_ylabel(r"$\frac{\|total-continuum\|}{N_{tot}\ flux_{bin}}, erg/s/cm^2/arcmin^2$")
    ax.set_xlabel(r"flux, $erg/s/cm^2/arcmin^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("err_vs_flux")
    plt.legend()
    plt.tight_layout()
    plt.savefig(work_path+'/png/err_vs_flux_halo_'+halo_id+'.png'
        )
    plt.close()
    print("Finish plotting err vs flux ! ")

    # plot pdf of fluxes
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    pure_pdf,__ = np.histogram(flux_pure_1d, bin)
    redcontin_pdf,__ = np.histogram(pR_redcontin.flatten(),bin)
    ax.set_xlabel(r"flux, $erg/s/cm^2/arcmin^2$")
    ax.set_ylabel(r"$\frac{N}{N_{tot}\ flux_{bin}}$")
    ax.set_title("flux pdf")
    ax.stairs(pure_pdf/xsize**2/(bin[1::]-bin[0:-1]), edges=bin, label='pure')
    ax.stairs(redcontin_pdf/xsize**2/(bin[1::]-bin[0:-1]),edges=bin, label = "total-continuum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(work_path+'/png/flux_pdf_halo_'+halo_id+'.png'
        )
    plt.close()
    print("Finish plotting flux pdf ! ")
    
def main(check_filename,redshift_range,obs_bin_rest,obs_bin_received):
    '''
    This is function for calculating xray map data for one property and plotting

    Parameters
    ---------------------------------------
    check_filename: lst (str)
        lst of pure & tot map_data name

    redshift_range: np.array (1x2)
        describe data's redshift_range, used in data's title

    obs_bin_rest: np.array (1x2)
        energy range at z_halo 

    obs_bin_received: np.array (1x2)
        observed energy range at z=0 
    
    continuum_bin: np.array (1x2)
        if wants to reduce with continuum, input continuum energy range
        if don't want to reduce the continuum, input empty list: []

    Returns
    ---------------------------------------
    flux_inbin: np.array (particle num ,1)
        flux in obs_bin_received that inside restframe energy bin, output for calculating contam

    save xray flux as npz

    plot xray

    '''
    
    if os.path.exists(work_path+"/data/flux_pR_smooth.npz") is False:
        if os.path.exists(work_path+'/data/'+check_filename+'.npz') is False:
            # compute pure xrays
            print("computing " + check_filename)
            flux,pixel,restframe_energy,z_par,z_los = compute_xray(input_filename,vector,radius,redshift_range,PLOT=True, COMPUTE=False,SEL_SPEC_PARTICLE=False)
            flux = flux.T
            restframe_energy = restframe_energy.value
            np.savez_compressed(work_path+'/data/'+check_filename+'.npz',flux=flux,pixel=pixel,restframe_energy = restframe_energy,z_par = z_par,z_los=z_los)
            print(work_path+'/data/'+check_filename+'.npz  has been saved!')
        else:
            print("loading " + check_filename)
            data = np.load(work_path+'/data/'+check_filename+'.npz')
            flux = data['flux']
            ## for new data
            pixel = data['pixel']
            restframe_energy = data['restframe_energy']
            z_par = data['z_par']
            z_los = data['z_los']

        print(np.count_nonzero(flux))
        # ### for every particle
        restE_lo = obs_bin_received[0]*(1+z_par)*(1+z_los)
        restE_hi = obs_bin_received[1]*(1+z_par)*(1+z_los)

        ### for pure emission (if there is only OVIII line in spectrum)
        ## find obs_bin_rest corresponds to which restframe_energy bin
        idx_pure = np.where((restframe_energy>=obs_bin_rest[0]) & (restframe_energy<=obs_bin_rest[1])) 
        print("pure bin in restframe_energy array is ", idx_pure) # make sure there is only one bin
        ## if particles OVIII redshift out of OVIII bin, set emission as 0
        print(np.shape(flux))
        flux_pure = np.sum(flux[:,idx_pure],axis=1)
        # flux_pure = flux[idx_pure,:][0][0]
        print(np.shape(flux_pure))
        flux_pure[(restE_lo > obs_bin_rest[1]) | (restE_hi < obs_bin_rest[0])] = 0
        print(np.count_nonzero(flux_pure))
        print("plotting pure ... ")
        flux_map_pure = compute_mapdata(pixel,flux_pure)
        flux_map_pure_sm = gaussian_filter(flux_map_pure, sigma=1)
        pR_pure = cartproj.projmap(flux_map_pure_sm/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
        # plot_settings = {"data_filter":[1e-16,1e-13],"mode":"log"}
        # plot_all(pR_pure,check_filename+'_pure1',r'erg/s/$cm^2/arcmin^2$','plasma',redshift_range,plot_settings)

        ## for total emission
        if os.path.exists(work_path+"/data/flux_tot.npz") is False:
            flux_tot = sel_flux(obs_bin_received,restframe_energy,flux,z_par,z_los,'sum')
            np.savez_compressed(work_path+"/data/flux_tot",flux_tot = flux_tot)
        else:
            flux_tot = np.load(work_path+"/data/flux_tot.npz")['flux_tot']
        flux_map_tot = compute_mapdata(pixel,flux_tot)
        flux_map_tot_sm = gaussian_filter(flux_map_tot, sigma=1)
        pR_tot = cartproj.projmap(flux_map_tot_sm/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
        print("plotting total ... ")
        # plot_settings = {"data_filter":[1e-16,1e-13],"mode":"log"}
        # plot_all(pR_tot/pix_a,check_filename+'_tot1',r'erg/s/$cm^2/arcmin^2$','plasma',redshift_range,plot_settings)

        # ### for contam
        # print("plotting contam ... ")
        # flux_map_contam = flux_map_tot-flux_map_pure
        # flux_map_contam_sm = gaussian_filter(flux_map_contam, sigma=1)
        # pR_contam = cartproj.projmap(flux_map_contam_sm/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
        # plot_settings = {"data_filter":[1e-16,1e-13],"mode":"log"}
        # plot_all(pR_contam/pix_a,check_filename+'_contam_truevalues',r'erg/s/$cm^2/arcmin^2$','plasma',redshift_range,plot_settings)

        ### for continuum
        print("plotting continuum ... ")
        contin_bin_obs1 = np.array([obs_bin_received[0]-0.08,obs_bin_received[0]])
        contin_bin_obs2 = np.array([obs_bin_received[1],obs_bin_received[1]+0.08])
        if os.path.exists(work_path+"/data/flux_contin.npz") is False:
            flux_contin1 = sel_flux(contin_bin_obs1,restframe_energy,flux,z_par,z_los,'med')
            flux_contin2 = sel_flux(contin_bin_obs2,restframe_energy,flux,z_par,z_los,'med')
            np.savez_compressed(work_path+"/data/flux_contin",contin1 = flux_contin1,contin2 = flux_contin2)
        else:
            flux_contin1 = np.load(work_path+"/data/flux_contin.npz")['contin1']
            flux_contin2 = np.load(work_path+"/data/flux_contin.npz")['contin2']

        flux_contin = (flux_contin1+flux_contin2)/2
        flux_map_contin = compute_mapdata(pixel,flux_contin)
        flux_map_contin_sm = gaussian_filter(flux_map_contin, sigma=1)
        pR_contin = cartproj.projmap(flux_map_contin_sm/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
        # plot_settings = {"data_filter":[1e-16,1e-13],"mode":"log"}
        # plot_all(flux_map_contin_sm/pix_a,check_filename+'_contin1',r'erg/s/$cm^2/arcmin^2$','plasma',redshift_range,plot_settings)

        flux_map_redcontin = flux_map_tot-flux_map_contin
        flux_map_redcontin_sm = gaussian_filter(flux_map_redcontin, sigma=1)
        pR_redcontin = cartproj.projmap(flux_map_redcontin_sm/pix_a, lambda x, y, z: hp.vec2pix(nside, x, y, z))
        # plot_settings = {"data_filter":[1e-16,1e-13],"mode":"log"}
        # plot_all(flux_map_redcontin_sm/pix_a,check_filename+'_tot-contin1',r'erg/s/$cm^2/arcmin^2$','plasma',redshift_range,plot_settings)
        
        np.savez_compressed(work_path+"/data/flux_pR_smooth",pR_pure = pR_pure,pR_tot = pR_tot,pR_contin = pR_contin, pR_redcontin = pR_redcontin)
    else:
        print("loading flux maps (pR) ...")
        file = np.load(work_path+"/data/flux_pR_smooth.npz")
        pR_pure = file['pR_pure']
        pR_tot = file['pR_tot']
        pR_redcontin = file['pR_redcontin']

    # plot pdf
    flux_pdf(pR_pure,pR_tot,pR_redcontin)

    # plot radial profile
    annu,lum_tot,pix_num = plot_profile(pR_tot,1)
    __,lum_pure,__ = plot_profile(pR_pure,1)
    __,lum_redcontin,__ = plot_profile(pR_redcontin,1)

    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    ax.stairs(lum_tot/pix_num,edges=annu,label = "total")
    ax.stairs(lum_pure/pix_num,edges=annu,label = "pure")
    ax.stairs(lum_redcontin/pix_num,edges=annu, label = "total-continuum")
    ax.set_xlabel("distance from halo center, arcmin")
    ax.set_ylabel("xray flux, $erg/s/cm^2/arcmin^2$")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(work_path+'/png/radialprofile_1arcmin_halo_'+halo_id+
            '_z_'+str(redshift_range[0])+'_'+str(redshift_range[1])+'.png'
        )
    plt.close()
    print("Finish plotting radial profile ! ")
    
    

### compute, save and plot data
main('xray_flux',z_range,obs_bin_rest,obs_bin_received)

## select halo
# select_halo(input_filename,0,1e4,1e5,RELAX=False)
# print(np.load("haloid_m200c_100.0_1000.0_1e10msun_z_0.05_lightcone0_relaxed.npy"))
# load_halo_properties(10976025,0)

# plot particle spectrum
# compute_xray(input_filename,vector,radius,z_halo_range,PLOT=False, COMPUTE=False,SEL_SPEC_PARTICLE=True)
# particle_spec("xray_flux.npz","spec_particle.npz")

# plot others
# compute_xray(input_filename,vector,radius,z_range,PLOT=True, COMPUTE=False,SEL_SPEC_PARTICLE=False)

# plot pixel spectrum
# pixel_spec("xray_flux.npz",[-1,0,9,10])

