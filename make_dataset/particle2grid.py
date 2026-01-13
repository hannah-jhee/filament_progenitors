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

with open("/data2/disperse_smooth_test_N300/smoothing_scale_table2.pickle","rb") as handle:
    table = pickle.load(handle)

def main(args):
    snapNum = args.snapNum
    clsnum = args.clsnum
    
    s = LSSLoader()
    s.load_collapsed_structures(halo=False)
    cx,cy,cz,cr = s.clusters[['X','Y','Z','RVIR']].iloc[clsnum]

    sq_filename = f"/data2/disperse_fits/s{args.snapNum:03d}_c{args.clsnum:02d}_sq.dat"
    
    if os.path.isfile(sq_filename):
        parts = np.loadtxt(sq_filename, skiprows=1)
    else:
        boxsize = 20e-3 * cr
        
        df = pd.read_pickle(f"/data2/Ncluster/particle/{snapNum:03d}/cls_{clsnum:02d}_dm_particles.snap_{snapNum:03d}.pkl")
        idx = (np.abs(df.x.values-cx)<boxsize/2) & (np.abs(df.y.values-cy)<boxsize/2) & (np.abs(df.z.values-cz)<boxsize/2)
        df = df.iloc[idx]
        parts = df[['x','y','z']].values
        del df
        if args.save_sq:
            np.savetxt(sq_filename, parts, header='px py pz')

    m_part = 1.072e9 # Msun/h
    grid,_ = np.histogramdd(parts, bins=args.nbins)

    Lbox = parts[:,0].max() - parts[:,0].min()
    l = Lbox/args.nbins
    grid *= (m_part/l**3) # Msun/Mpc^3

    if args.smoothing_scale != 'auto':
        smoothing_bin = int(np.around(float(args.smoothing_scale) / l)) # empirically
        file_name = f'./s{args.snapNum:03d}_c{args.clsnum:02d}_smt_{args.smoothing_scale}.fits'
    else:
        print(f"Automatic smoothing scale : {table[args.snapNum]*args.weight_smoothing:.6f} Mpc/h")
        smoothing_bin = int(np.around(table[args.snapNum]*args.weight_smoothing / l))
        file_name = f'./s{args.snapNum:03d}_c{args.clsnum:02d}_smt_w{args.weight_smoothing}_auto.fits'
    grid_smoothed = gaussian_filter(grid, sigma=smoothing_bin, truncate=3*smoothing_bin, mode='constant')
    grid_smoothed[grid_smoothed<=0] = grid_smoothed[grid_smoothed>0].min() # deal with zero density region
    
    #crit_den = cosmo.critical_density(az[args.snapNum]).to(u.Msun/u.Mpc**3).value
    #overdensity_smoothed = np.log10(1+(grid_smoothed-crit_den)/crit_den)
    field_value = (grid_smoothed-grid_smoothed.mean())/grid_smoothed.std()
    #field_value = (overdensity_smoothed - np.mean(overdensity_smoothed)) / np.std(overdensity_smoothed)

    hdu = fits.PrimaryHDU(field_value)#overdensity_smoothed)
    hdu.writeto(file_name, overwrite=True)



if __name__=="__main__":
    p = argparse.ArgumentParser()

    p.add_argument("-save_sq", dest="save_sq", type=bool, default=True)
    p.add_argument("-snapNum", dest='snapNum', type=int, default=168)
    p.add_argument("-clsnum", dest='clsnum', type=int, default=0)
    p.add_argument("-smoothing_scale", dest="smoothing_scale", help='in Mpc/h', type=str, default='auto')
    p.add_argument("-nbins", dest="nbins", type=int, default=300)
    p.add_argument("-weight_smoothing", dest="weight_smoothing", type=float, default=1.37)

    args = p.parse_args()

    main(args)
