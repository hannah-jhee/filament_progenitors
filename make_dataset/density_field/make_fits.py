#python  make_fits -snapNum 169 -smoothing_scale auto -nbins=300
import pickle
import argparse
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u

f = open("/data2/Ncluster/a_and_z.dat","r")
lines = f.readlines()
az = dict() # Snapshot number to redshift(z)
for line in lines:
    words = list(map(float,line.split()))
    az[words[0]] = words[-1]
m_part = 1.072e9 # Msun/h

with open("/data2/disperse_smooth_test_N300_log/smoothing_scale_table2.pickle","rb") as handle:
    table = pickle.load(handle)

def construct_density_field_input_1(clsnum, snapNum, nbins, verbose=True):
    redshift = az[snapNum]

    if verbose: print("Loading particles...")
    parts = np.loadtxt(f"/data2/disperse_fits/s{snapNum:03d}_c{clsnum:02d}_sq.dat", skiprows=1)

    if verbose: print("Constructing a grid...")
    grid,_ = np.histogramdd(parts, bins=nbins)

    Lbox = parts[:,0].max() - parts[:,0].min()
    l = Lbox/nbins
    grid *= (m_part/l**3) # Msun/Mpc^3

    smoothing_length = table[snapNum] * 1.5
    smoothing_bin = int(np.around(smoothing_length / l))
    if verbose: print(f'Smoothing length : {smoothing_length:.2f} cMpc/h at z={redshift:.2f}')
    if verbose: print(f'Smoothing bin    : {smoothing_bin}')

    if verbose: print(f'Smoothing the density field...')
    grid_sm = gaussian_filter(grid, sigma=smoothing_bin, truncate=3*smoothing_bin, mode='constant')#mode='reflect')

    if verbose: print('Calculating log density field...')
    log_grid_sm = np.log10(grid_sm)

    if verbose: print("Calculating the significance...")
    #mean_density_comoving = ((cosmo.critical_density(redshift) * cosmo.Om(redshift) / (1+redshift)**3).to(u.Msun / u.Mpc**3)).value
    log_grid_sm_significance = (log_grid_sm-log_grid_sm.mean()) / log_grid_sm.std()
    
    return log_grid_sm_significance

def construct_density_field_input_2(clsnum, snapNum, nbins, verbose=True):
    redshift = az[snapNum]

    if verbose: print("Loading particles...")
    parts = np.loadtxt(f"/data2/disperse_fits/s{snapNum:03d}_c{clsnum:02d}_sq.dat", skiprows=1)

    if verbose: print("Constructing a grid...")
    grid,_ = np.histogramdd(parts, bins=nbins)

    Lbox = parts[:,0].max() - parts[:,0].min()
    l = Lbox/nbins
    grid *= (m_part/l**3) # Msun/Mpc^3

    smoothing_length = table[snapNum] * 1.5
    smoothing_bin = int(np.around(smoothing_length / l))
    if verbose: print(f'Smoothing length : {smoothing_length:.2f} cMpc/h at z={redshift:.2f}')
    if verbose: print(f'Smoothing bin    : {smoothing_bin}')

    if verbose: print(f'Smoothing the density field...')
    grid_sm = gaussian_filter(grid, sigma=smoothing_bin, truncate=3*smoothing_bin, mode='reflect')

    if verbose: print("Calculating the significance...")
    mean_density_comoving = ((cosmo.critical_density(redshift) * cosmo.Om(redshift) / (1+redshift)**3).to(u.Msun / u.Mpc**3)).value
    grid_sm_overdensity = grid_sm / mean_density_comoving
    return grid_sm_overdensity


if __name__=="__main__":
    p = argparse.ArgumentParser()

    p.add_argument("-snapNum", dest='snapNum', type=int, default=168)
    p.add_argument("-clsnum", dest='clsnum', type=int, default=0)
    p.add_argument("-nbins", dest="nbins", type=int, default=300)
    p.add_argument("-verbose", dest='verbose', action='store_true')

    args = p.parse_args()

    field_value = construct_density_field_input_1(args.clsnum, args.snapNum, args.nbins, verbose=args.verbose)
    hdu = fits.PrimaryHDU(field_value)
    file_name = f'/data2/disperse_smooth_test_N300_log/s{args.snapNum:03d}_c{args.clsnum:02d}_grid.fits'
    hdu.writeto(file_name, overwrite=True)
