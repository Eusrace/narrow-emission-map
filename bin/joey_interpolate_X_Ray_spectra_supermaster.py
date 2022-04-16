import h5py
import numpy as np
from swiftsimio import load
from numba import jit
from unyt import g, cm, mp, erg, s, keV

class interpolate:
    def init(self):
        pass

    def load_table(self):
        '''
        This is a method for swift data to load abundance table
        '''
        self.table = h5py.File('/cosma7/data/dp004/dc-bras1/swift_data/x-rays/spectra_test/X_Ray_spectra_few_z.hdf5', 'r')
        self.X_Ray = self.table['spectra'][()]
        self.xray_energies = self.table['xray_energies'][()]
        self.He_bins = self.table['/Bins/He_bins'][()]
        self.missing_elements = self.table['/Bins/Missing_element'][()]
        self.element_masses = self.table['Bins/Element_masses'][()]

        self.density_bins = self.table['/Bins/Density_bins/'][()]
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()]
        self.redshift_bins = self.table['/Bins/Redshift_bins'][()]
        self.dn = 0.2
        self.dT = 0.1
        self.dz = 0.2

        self.solar_metallicity = self.table['/Bins/Solar_metallicities/'][()]

@jit(nopython = True)
def find_dx(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            dx_p[i] = 0
        elif (subdata[i] > bins[-1]):
            dx_p[i] = np.abs(bins[-1] - bins[-2])
        else:
            dx_p[i] = np.abs(bins[idx_0[i]] - subdata[i])

    return dx_p

@jit(nopython = True)
def find_idx(subdata, bins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            idx_p[i, :] = np.array([0, 1])
        elif (subdata[i] > bins[-1]):
            idx_p[i, :] = np.array([len(bins)-2, len(bins)-1])
        else:
            idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p

@jit(nopython = True)
def find_idx_z(subdata, bins):
    idx_p = np.zeros(2)
    if (subdata < bins[0]):
        idx_p = np.array([0, 1])
    elif (subdata > bins[-1]):
        idx_p = np.array([len(bins)-2, len(bins)-1])
    else:
        idx_p = np.sort(np.argsort(np.abs(bins - subdata))[:2])

    return idx_p

@jit(nopython = True)
def find_dx_z(subdata, bins, idx_0):
    dx_p = 0
    if (subdata < bins[0]):
        dx_p = 0
    elif (subdata > bins[-1]):
        dx_p = 1
    else:
        dx_p = np.abs(subdata - bins[idx_0]) / (bins[idx_0 + 1] - bins[idx_0])

    return dx_p

@jit(nopython = True)
def find_idx_he(subdata, bins):
    num_bins = len(bins)
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        
        # When closest to the highest bin, or above the highest bin, return the one but highest bin,
        # otherwise we will select a second bin which is outside the binrange
        bin_below = min(np.argsort(np.abs(bins[bins <= subdata[i]] - subdata[i]))[0], num_bins - 2)
        idx_p[i, :] = np.array([bin_below, bin_below + 1])
    return idx_p

@jit(nopython = True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            dx_p[i] = 0
        elif (subdata[i] > bins[-1]):
            dx_p[i] = 1
        else:
            dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i]+1] - bins[idx_0[i]])

    return dx_p

@jit(nopython = True)
def get_index_1d(bins, subdata):
    eps = 1e-4
    delta = (len(bins) - 1) / (bins[-1] - bins[0])

    idx = np.zeros_like(subdata)
    dx = np.zeros_like(subdata)
    for i, x in enumerate(subdata):
        if x < bins[0] + eps:
            idx[i] = 0
            dx[i] = 0
        elif x < bins[-1] - eps:
            idx[i] = int((x - bins[0]) * delta)
            dx[i] = (x - bins[int(idx[i])]) * delta
        else:
            idx[i] = len(bins) - 2
            dx[i] = 1
        
    return idx, dx

@jit(nopython = True)
def get_index_1d_irregular(bins, subdata):
    eps = 1e-6

    idx = np.zeros_like(subdata)
    dx = np.zeros_like(subdata)

    for i, x in enumerate(subdata):
        if x < bins[0] + eps:
            idx[i] = 0
            dx[i] = 0
        elif x < bins[-1] - eps:
            min_idx = -1
            for i in range(len(bins)):
                if x - bins[i] <= 0:
                    min_idx = i - 1
                    break
            idx[i] = min_idx
            dx[i] = (x - bins[min_idx]) / (bins[min_idx + 1] - bins[min_idx])
        else:
            idx[i] = len(bins) - 2
            dx[i] = 1
    
    return idx, dx

@jit(nopython = True, parallel = False)
def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, idx_z, dx_z, X_Ray, abundance_to_solar, bin_num):
    f_n_T_Z = np.zeros_like(dx_n)

    # Compute redshift offset relative to
    # t_z = (1 - dx_z)
    # d_z = dx_z


    
    for i in range(len(idx_n)):
        t_z = 1 - dx_z[i]
        d_z = dx_z[i]

        # Compute temperature offset relative to bin
        # t_T = (dT - dx_T[i]) / dT
        # d_T = dx_T[i] / dT
        t_T = 1 - dx_T[i]
        d_T = dx_T[i]

        # Compute density offset relative to bin
        # t_n = (dn - dx_n[i]) / dn
        # d_n = dx_n[i] / dn
        t_n = 1 - dx_n[i]
        d_n = dx_n[i]

        # Compute Helium offset relative to bin
        # d_he = dx_he[i]
        # t_he = (1 - dx_he[i])
        t_he = 1 - dx_he[i]
        d_he = dx_he[i]

        # Do the actual 4D linear interpolation
        f_n_T = t_T * t_n * t_he * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i], bin_num]
        if np.isnan(f_n_T.max()):
            print('f_n_T', i, f_n_T)
        f_n_T += t_T * t_n * d_he * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i], bin_num]
        f_n_T += t_T * d_n * t_he * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i] + 1, bin_num]
        f_n_T += d_T * t_n * t_he * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i], bin_num]
        f_n_T += t_T * d_n * d_he * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1, bin_num]
        f_n_T += d_T * t_n * d_he * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i], bin_num]
        f_n_T += d_T * d_n * t_he * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1, bin_num]
        f_n_T += d_T * d_n * d_he * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1, bin_num]

        f_n_T += t_T * t_n * t_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i], bin_num]
        f_n_T += t_T * t_n * d_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i], bin_num]
        f_n_T += t_T * d_n * t_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i] + 1, bin_num]
        f_n_T += d_T * t_n * t_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i], bin_num]
        f_n_T += t_T * d_n * d_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1, bin_num]
        f_n_T += d_T * t_n * d_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i], bin_num]
        f_n_T += d_T * d_n * t_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1, bin_num]
        f_n_T += d_T * d_n * d_he * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1, bin_num]
        # if np.isnan(f_n_T.max()):
        #     print('f_n_T', i, f_n_T)
        # Add each metal contribution individually
        f_n_T_Z_temp = np.power(10, f_n_T[-1])
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp += np.power(10, f_n_T[j]) * abundance_to_solar[i, j]
            if np.isnan(f_n_T_Z_temp) | (f_n_T_Z_temp <= 0):
                print('naner', f_n_T, f_n_T_Z_temp, abundance_to_solar[i, j])
        f_n_T_Z[i] = np.log10(f_n_T_Z_temp)
        # if np.isnan(f_n_T_Z[i]):
        #     print('f_n_T_Z', i, f_n_T_Z[i])
    return f_n_T_Z

def interpolate_X_Ray(densities, temperatures, element_mass_fractions, redshift, masses, fill_value = None, bin_energy_lims = [0]):
    '''
    This function interpolate xray luminosity for each particle
    
    Parameters
    -----------------------------
    bin_energy_lims: list (1,2)
        bin energy limit 
        
    '''
    scale_factor = 1 / (1 + redshift)
    data_n = np.log10(element_mass_fractions.hydrogen * (1 / scale_factor**3) * densities.to(g * cm**-3) / mp)
    data_T = np.log10(temperatures)
    volumes = masses / ((1 / scale_factor**3) * densities)

    # Initialise interpolation class
    interp = interpolate()
    print('loading table')
    interp.load_table()
    print('table_loaded')

    # Find the bins numbers corresponding to the energies
    bin_low = np.sum(interp.xray_energies - bin_energy_lims[0] < 0)
    bin_high = np.sum(interp.xray_energies - bin_energy_lims[1] < 0)
    bin_nums = np.arange(bin_low, bin_high)

    # Initialise the emissivity array which will be returned
    emissivities = np.zeros((data_n.shape[0], len(bin_nums)), dtype = float)

    # Create density mask, round to avoid numerical errors
    density_mask = (data_n >= np.round(interp.density_bins.min(), 1)) & (data_n <= np.round(interp.density_bins.max(), 1))
    # Create temperature mask, round to avoid numerical errors
    temperature_mask = (data_T >= np.round(interp.temperature_bins.min(), 1)) & (data_T <= np.round(interp.temperature_bins.max(), 1))

    # Combine masks
    joint_mask = density_mask & temperature_mask
    print('particles to interpolate = ', np.sum(joint_mask))

    # Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    if ~(density_bounds & temperature_bounds):
        #If no fill_value is set, return an error with some explanation
        if fill_value == None:
            raise ValueError("Temperature or density are outside of the interpolation range and no fill_value is supplied\n \
                               Temperature ranges between log(T) = 5 and log(T) = 9.5\n \
                               Density ranges between log(nH) = -8 and log(nH) = 6\n \
                               Set the kwarg 'fill_value = some value' to set all particles outside of the interpolation range to 'some value'\n \
                               Or limit your particle data set to be within the interpolation range")
        else:
            emissivities[~joint_mask] = fill_value
    
    # If only a single redshift is received, use it for all particles
    if redshift.size == 1:
        redshift = np.ones_like(data_n) * redshift





    

    mass_fraction = np.zeros((len(data_n[joint_mask]), 9))

    #get individual mass fraction
    mass_fraction[:, 0] = element_mass_fractions.hydrogen[joint_mask]
    mass_fraction[:, 1] = element_mass_fractions.helium[joint_mask]
    mass_fraction[:, 2] = element_mass_fractions.carbon[joint_mask]
    mass_fraction[:, 3] = element_mass_fractions.nitrogen[joint_mask]
    mass_fraction[:, 4] = element_mass_fractions.oxygen[joint_mask]
    mass_fraction[:, 5] = element_mass_fractions.neon[joint_mask]
    mass_fraction[:, 6] = element_mass_fractions.magnesium[joint_mask]
    mass_fraction[:, 7] = element_mass_fractions.silicon[joint_mask]
    mass_fraction[:, 8] = element_mass_fractions.iron[joint_mask]


    # Find density offsets
    # idx_n = find_idx(data_n[joint_mask], interp.density_bins)
    # dx_n = find_dx(data_n[joint_mask], interp.density_bins, idx_n[:, 0].astype(int))
    idx_n, dx_n = get_index_1d(interp.density_bins, data_n[joint_mask])
    # print('nh', data_n[joint_mask][0], idx_n[0], dx_n[0])
    # Find temperature offsets
    # idx_T = find_idx(data_T[joint_mask], interp.temperature_bins)
    # dx_T = find_dx(data_T[joint_mask], interp.temperature_bins, idx_T[:, 0].astype(int))
    idx_T, dx_T = get_index_1d(interp.temperature_bins, data_T[joint_mask])
    # print('T', data_T[joint_mask][0], idx_T[0], dx_T[0])
    # Find element offsets
    # mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    # element_masses = [1.008, 4.003, 12.01, 14.01, 16., 20.18, 24.31, 28.09, 55.85]

    # Calculate the abundance wrt to solar
    abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis = 1)) * (interp.element_masses[0] /  np.array(interp.element_masses))

    # Calculate abundance offsets using solar abundances
    abundance_to_solar = abundances / 10**interp.solar_metallicity

    # Add columns for Calcium and Sulphur and add Iron at the end
    abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] 

    # print(np.log10(abundances[:, 1]))
    # print(interp.solar_metallicity[1])
    #Find helium offsets
    # idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    # dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))
    idx_he, dx_he = get_index_1d_irregular(interp.He_bins, np.log10(abundances[:, 1]))
    # print('he', idx_he[0], dx_he[0])
    # Find redshift offsets
    # idx_z = find_idx_z(redshift, interp.redshift_bins)
    # dx_z = find_dx_z(redshift, interp.redshift_bins, idx_z[0].astype(int))
    idx_z, dx_z = get_index_1d(interp.redshift_bins, redshift)
    # print('z', idx_z[0], dx_z[0])
    
    luminosities = np.zeros((emissivities.shape[0], len(bin_nums))) * erg * s**-1 

    for i, bin in enumerate(bin_nums):
        print('Start interpolation ', i, ' of ', len(bin_nums))
        emissivities[joint_mask, i] = get_table_interp(interp.dn, interp.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int), idx_he.astype(int), dx_he, idx_z.astype(int), dx_z, interp.X_Ray, abundance_to_solar[:, 2:], bin)
        print('interpolation ', i, ' of ', len(bin_nums), ' finished')
        # Convert from erg cm^3 s^-1 to erg cm^-3 s^-1
        # To do so we multiply by nH^2, this is the actual nH not the nearest bin
        # It allows to extrapolate in density space without too much worry
        # log(emissivity * nH^2) = log(emissivity) + 2*log(nH)
        # print(np.max(emissivities[joint_mask, i]))
        emissivities[joint_mask, i] += 2*data_n[joint_mask]
        # print(np.max(emissivities[joint_mask, i]))

        # Convert to luminosities
        luminosities[joint_mask, i] = np.power(10, emissivities[joint_mask, i]) * erg * s**-1 *cm**-3 * volumes[joint_mask]


    # Restframe (of emitter) energies belonging to the luminosities
    restframe_energy = interp.xray_energies[bin_low:bin_high+1] * keV
    
    return luminosities, restframe_energy
