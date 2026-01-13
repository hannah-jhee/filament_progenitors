import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import interpolate
from operator import itemgetter

import ctypes
from numpy.ctypeslib import ndpointer

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
#mpl.rc('text', usetex=True)
#mpl.rc('font', size=14)
#mpl.rc('legend', fontsize=13)
#mpl.rc('text.latex', preamble=r'\usepackage{cmbright}')
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/gpuadmin/AVHersheySimplexMedium.ttf")
# set font
plt.rcParams['font.family'] = 'AVHershey Simplex'
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.rm'] = 'AVHershey Simplex'
plt.rcParams['mathtext.it'] = 'AVHershey Simplex'

############ Simulation Info ############
m_part = 1.072e9 # Msun/h
f = open("/data2/Ncluster/a_and_z.dat","r")
lines = f.readlines()
az = dict() # Snapshot number to redshift(z)
for line in lines:
    words = list(map(float,line.split()))
    az[words[0]] = words[-1]

class LSSLoader:
    def __init__(self, basePath="/data2/Ncluster"):
        self.basePath = basePath
        
        self.clusters = None
        self.halos = dict()
        self.filaments = dict()

        self.host = dict()
        
    def read_raw_halo_file(self):
        halos = pd.read_pickle(f"{self.basePath}/halo/halo_more_infos.pkl")
        columns = 'num_prog', 'Mvir', 'Rvir', 'vrms', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz', 'Spin', 'Tree_root_ID', 'Snap_idx', 'Xoff', 'Voff', 'b_to_a', 'c_to_a'
        halos.columns = columns
        return halos


    def load_collapsed_structures(self, cluster=True, halo=True):
        if cluster:
            print("** Loading Cluster data")
            self.clusters = pd.read_pickle(f"{self.basePath}/cluster/cluster_all.pkl").iloc[:47]
        if halo:
            print("** Loading Halo data")
            for clsnum in tqdm(range(47)):
                with open('/data2/Ncluster/halo/halos_around_cls_%02d.pickle'%clsnum, 'rb') as handle:
                    self.halos[clsnum] = pickle.load(handle)
      

    def load_filaments(self, file_format=None):
        print("** Loading Filament data")
        for snapNum in tqdm(range(50,169)):
            if file_format == None:
                self.filaments[snapNum] = self.read_filament(f"{self.basePath}/filament/halos_i01_f{snapNum:03d}_c01.NDnet_s6.up.NDskl.S010.BRK.ASMB.a.NDskl")
            else:
                self.filaments[snapNum] = self.read_filament(file_format % snapNum)

    def read_filament(self,fname):
        cls = 0
        columns = 'CP1 CP2 x y z'.split()
        fil_dataset = dict()

        lines = open(fname).readlines()
        
        idx_fil = lines.index("[FILAMENTS]\n")
        idx_fil_dat = lines.index("[FILAMENTS DATA]\n")
        N_fil = int(lines[idx_fil+1])
        N_field_data = int(lines[idx_fil_dat+1])
        for i in range(N_field_data):
            columns.append(lines[idx_fil_dat+2+i].strip())

        start_idx = idx_fil+2
        start_data_idx = idx_fil_dat + 2 + N_field_data

        for i in range(N_fil):
            cp1, cp2, n = list(map(int,lines[start_idx].split()))
            dataset = []
            j=0
            while j<n:
                x,y,z = list(map(float, lines[start_idx+1+j].split()))
                dataset.append([cp1,cp2,x,y,z])
                dataset[-1].extend(list(map(float,lines[start_data_idx+j].split())))
                j+=1
            start_idx += j+1
            start_data_idx += j

            fil_dataset[i] = pd.DataFrame(columns=columns, data=dataset)
            #fil_dataset[i]['x'] = fil_dataset[i]['x'] - 60 + self.clusters.X.iat[cls]
            #fil_dataset[i]['y'] = fil_dataset[i]['y'] - 60 + self.clusters.Y.iat[cls]
            #fil_dataset[i]['z'] = fil_dataset[i]['z'] - 60 + self.clusters.Z.iat[cls]
        return fil_dataset
    
    def read_filament_critical(self, fname):
    
        columns = 'type x y z value pairID boundary connectivity destination filamentID'.split()
        crit_dataset = []

        lines = open(fname).readlines()

        idx_crt = lines.index("[CRITICAL POINTS]\n")
        N_critical = int(lines[idx_crt+1])

        start_idx = idx_crt + 2
        for i in range(N_critical):
            crit_dataset.append(list(map(float,lines[start_idx].split())))
            connectivity = int(lines[start_idx+1])
            crit_dataset[-1].append(connectivity)
            dest,fils = [],[]
            for con in range(connectivity):
                d,f = list(map(int,lines[start_idx+2+con].split()))
                dest.append(d)
                fils.append(f)
            crit_dataset[-1].extend([dest,fils])

            start_idx += connectivity+2

        idx_crt = lines.index("[CRITICAL POINTS DATA]\n")
        n_columns = int(lines[idx_crt+1])
        for i in range(n_columns):
            columns.append(lines[idx_crt+2+i].strip())

        start_idx = idx_crt+n_columns+2
        for i in range(N_critical):
            crit_dataset[i].extend(list(map(float,lines[start_idx+i].split())))

        return pd.DataFrame(columns=columns, data=crit_dataset)
    
def MostMassiveProgenitor(halo):
    if len(halo) != len(np.unique(halo.Snap_idx.values)):
        halo = halo.sort_values(by="Mvir", ascending=False).groupby("Snap_idx").head(1).sort_values(by="Snap_idx")
        return halo
    else:
        return halo
    
def read_progenitor(filename, clsnum):
    with open(filename,"r") as f:
        lines = f.readlines()
    line_num = 0
    
    progs = dict()
    prog_keys = dict()
    while True:
        snapNum, nsig, score, n_fil, maxkey = lines[line_num].split()
        snapNum = int(snapNum)
        #nsig = np.round(float(nsig),2)
        score = float(score)
        n_fil = int(n_fil)
        maxkey = int(maxkey)
        if '_w' not in filename:
            fil  = s.read_filament(f"/data2/disperse_smooth_test_N300/s{snapNum:03d}_c{clsnum:02d}_smt_auto.fits_c{nsig}.up.NDskl.S010.rmB.a.NDskl")
            crit = s.read_filament_critical(f"/data2/disperse_smooth_test_N300/s{snapNum:03d}_c{clsnum:02d}_smt_auto.fits_c{nsig}.up.NDskl.S010.rmB.a.NDskl")
        else:
            weight = filename.split('/')[-1].split('_')[2].split('.dat')[0]
            fil  = s.read_filament(f"/data2/disperse_smooth_test_N300/s{snapNum:03d}_c{clsnum:02d}_smt_{weight}_auto.fits_c{nsig}.up.NDskl.S010.rmB.a.NDskl")
            crit = s.read_filament_critical(f"/data2/disperse_smooth_test_N300/s{snapNum:03d}_c{clsnum:02d}_smt_{weight}_auto.fits_c{nsig}.up.NDskl.S010.rmB.a.NDskl")
        pos = lines[line_num+1:line_num+n_fil+1]
        fil1 = fil[maxkey]
        cp1,cp2 = fil1[['CP1','CP2']].values[0]
        fil1 = fil1[['x','y','z']].values

        filnum2 = list(set(crit.iloc[cp2].filamentID) - set([maxkey]))
        if len(filnum2)>1:
            print(snapNum, crit.iloc[cp2].filamentID)
        filnum2 = filnum2[0]

        fil2 = fil[filnum2][['x','y','z']].values
        fil = np.r_[fil1, fil2[::-1,:]]

        progs[snapNum] = fil
        prog_keys[snapNum] = (nsig,maxkey)
        line_num += n_fil + 1
        #if snapNum == 50:
        #    print(f"Tracing ended at snapshot {snapNum:03d}.")
        #    break
        if line_num == len(lines):
            #print(f"Tracing ended at snapshot {snapNum:03d}.")
            break
    
    return progs, prog_keys

class Prog(dict):
    def __init__(self, filename, clsnum):
        progs,prog_keys = read_progenitor(filename=filename, clsnum=clsnum)
        self.prog_keys = prog_keys
        for snapNum in progs.keys():
            self[snapNum] = bin2Mpch(progs[snapNum], clsnum=clsnum, snapNum=snapNum)

def open_pickle(filename):
    with open(filename, "rb") as handle:
        file = pickle.load(handle)
    return file

def bin2Mpch(bin_coords, clsnum=0, snapNum=168):
    rvir = s.clusters.RVIR.iat[clsnum]*1e-3
    lbox = rvir*20
    Ovec = s.clusters[['X','Y','Z']].values[clsnum] - rvir*10
    bin_coords = bin_coords[:,[2,1,0]] / 300 * lbox + Ovec
    return bin_coords

class DATA:
    def __init__(self, clsnum, filnum, halos):
        self.clsnum = clsnum
        self.filnum = filnum

        self._read_progenitor()
        self._get_relevant_halos(halos)
        self._calc_phase_info()
        
    def _read_progenitor(self):
        progs = open_pickle(f"/home/gpuadmin/cosmicweb/src/progenitor_tracing/tree_output/c{self.clsnum:02d}/tree_{self.filnum}_w1.5.pickle")
        for snapNum in tqdm(progs.keys(), desc='fil progs'):
            progs[snapNum] = do_interpolation(progs[snapNum])
        self.progs = progs; del progs

    def _get_relevant_halos(self, halos, dcut=1):
        """
        dcut is in Mpc/h unit.
        """
        hs = halos[halos.Snap_idx==168]
        line_fix = calc_dist_to_line(self.progs[168])
        ds = np.array( list(map(line_fix, hs[['x','y','z']].values)) )

        near_definition = dcut #Mpc/h
        hs_nearby = hs.iloc[ds<near_definition]
        hids_ = hs_nearby.Tree_root_ID.values

        self.all_halos = dict()
        for hid in tqdm(hids_, desc='discriminating'):
            halo = MostMassiveProgenitor(halos[halos.Tree_root_ID==hid]).sort_values(by='Snap_idx', ascending=True).reset_index().drop('index', axis=1)
            halo = halo[halo.Snap_idx>=min(self.progs.keys())]
            #if halo.Mvir.iat[-1] > 4e10:
            add = True
            if 64*m_part<halo.Mvir.iat[-1]<1e13:
                for i,row in halo.iterrows():
                    if row.Snap_idx>=50:
                        line_fix = calc_dist_to_line(self.progs[row.Snap_idx])
                        ds = line_fix(row[['x','y','z']].values.flatten())
                        if ds==10000:
                            add = False
                            break
                if add:
                    self.all_halos[hid] = halo

        # z=3에서도 헤일로를 이루는 암흑물질 입자의 개수가 40개는 넘도록 설정
        self.halos = dict()
        for hid in self.all_halos.keys():
            halo = MostMassiveProgenitor(halos[(halos.Tree_root_ID==hid)&(halos.Snap_idx>=50)]).sort_values(by='Snap_idx',ascending=True).reset_index().drop('index', axis=1)
            halo = halo[halo.Snap_idx>=50]
            n_initial = int(halo.Mvir.iat[0] / m_part)
            if n_initial >= 35:
                self.halos[hid] = halo

    def _calc_phase_info(self):
        for hid in tqdm(self.all_halos.keys(), desc='phase info'):
            halo = self.all_halos[hid]
            halo = halo.sort_values(by='Snap_idx', ascending=True)

            times = []
            lookbacks = []
            filargs = []
            u_perps = []
            u_pars = []
            u_orbs = []
            r_perps = []

            v_perps2 = [] # 미분 말고 직접 속도 성분으로 계산한 값
            v_pars = []
            v_orbs = []
            for i,row in halo.iterrows():
                prog = self.progs[row.Snap_idx]
                dist = np.linalg.norm(row[['x','y','z']].values.flatten() - prog, axis=1)
                filarg = np.argmin(dist)

                u_perp = prog[filarg,:] - row[['x','y','z']].values.flatten()
                u_perp /= np.linalg.norm(u_perp)

                u_par,_,_ = find_unit_vectors(prog, filarg)

                v_perp = np.dot(row[['vx','vy','vz']].values, u_perp)
                v_par = np.dot(row[['vx','vy','vz']].values, u_par)
                v_orb = np.linalg.norm(row[['vx','vy','vz']].values - v_perp*np.array(u_perp) - v_par*np.array(u_par))

                filargs.append(filarg)
                r_perps.append(dist.min())
                u_perps.append(u_perp)
                u_pars.append(u_par)

                v_perps2.append(v_perp)
                v_pars.append(v_par)
                v_orbs.append(v_orb)
                lookbacks.append(cosmo.age(0).value-cosmo.age(z=az[row.Snap_idx]).value)
                times.append(cosmo.age(z=az[row.Snap_idx]).value)
            times = np.array(times)
            r_perps = np.array(r_perps)
            #r_perps = savgol_filter(r_perps, window_length=17, polyorder=3, mode='nearest')
            try:
                tck = splrep(times, r_perps, s=0.2)
            except TypeError as e:
                print(times, r_perps)
                raise e
            r_perps_smooth = BSpline(*tck)(times)
            v_perps = ((r_perps[1:]-r_perps[:-1])/(times[:-1]-times[1:])*u.Mpc/0.68/u.Gyr).to(u.km/u.s).value
            vperps_sm = ((r_perps_smooth[1:]-r_perps_smooth[:-1])/(times[:-1]-times[1:])*u.Mpc/0.68/u.Gyr).to(u.km/u.s).value
            u_perps = np.array(u_perps)
            u_pars = np.array(u_pars)

            self.all_halos[hid]['lookback'] = lookbacks
            self.all_halos[hid]['time'] = times
            self.all_halos[hid]['filarg'] = filargs
            self.all_halos[hid]['u_perp_x'] = u_perps[:,0]
            self.all_halos[hid]['u_perp_y'] = u_perps[:,1]
            self.all_halos[hid]['u_perp_z'] = u_perps[:,2]
            self.all_halos[hid]['u_par_x'] = u_pars[:,0]
            self.all_halos[hid]['u_par_y'] = u_pars[:,1]
            self.all_halos[hid]['u_par_z'] = u_pars[:,2]
            
            self.all_halos[hid]['r_perp'] = r_perps
            self.all_halos[hid]['r_perp_sm'] = r_perps_smooth
            self.all_halos[hid]['v_perp'] = [np.nan] + list(v_perps)
            self.all_halos[hid]['v_perp_sm'] = [np.nan] + list(vperps_sm)
            self.all_halos[hid]['velem_perp']  = v_perps2
            self.all_halos[hid]['velem_par'] = v_pars
            self.all_halos[hid]['velem_orb'] = v_orbs

            if hid in self.halos.keys():
                self.halos[hid] = self.all_halos[hid]


def PBC_center(centering, particles):
    cx,cy,cz = centering

    if cx>60:
        idx_x = particles[:,0]<cx-60
        particles[idx_x,0] += 120
    else:
        idx_x = particles[:,0]>cx+60
        particles[idx_x,0] -= 120

    if cy>60:
        idx_y = particles[:,1]<cy-60
        particles[idx_y,1] += 120
    else:
        idx_y = particles[:,1]>cy+60
        particles[idx_y,1] -= 120

    if cz>60:
        idx_z = particles[:,2]<cz-60
        particles[idx_z,2] += 120
    else:
        idx_z = particles[:,2]>cz+60
        particles[idx_z,2] -= 120
    return particles


def find_unit_vectors(fil, filarg, dim=3):
    """
    finfo is a list of [filsnap,filnum,filarg].
    if dim==1, it only returns parallel vector.
    if dim==3, you additionally get 2 more vectors for plotting.
    """
    #fil = s.filaments[finfo[0]][finfo[1]][['x','y','z']].values
    N_fil = fil.shape[0]
    filarg = int(filarg)
    
    num_set = 20

    if N_fil<num_set:
        u_par = fil[-1,:] - fil[0,:]
    else:
        if filarg < int(num_set/2): # 처음에 맞닿은 부분
            u_par = fil[num_set,:] - fil[0,:]
        elif filarg < N_fil-int(num_set/2):
            u_par = fil[filarg+int(num_set/2),:] - fil[filarg-int(num_set/2),:]
        else:
            u_par = fil[-1,:] - fil[-num_set,:]
    
    
    u_par /= np.linalg.norm(u_par)
    
    if dim==1:
        return u_par
    elif dim==3:
        v1 = np.array([1, 0, -u_par[0]/u_par[2]])
        v1 /= np.linalg.norm(v1)
        v1 *= np.sign(v1[2])
        v2 = np.cross(u_par, v1)
        v2 *= np.sign(v2[1])
        return u_par, v1, v2
    else:
        raise ValueError("Argument `dim` should be one of [1, 3]")


def load_particles(snapNum, centering=None, boxsize=None, velocity=None):
    """
    snapNum : snapshot number
    centering : 1d-array -> selected position
    boxsize : in Mpc/h
    """
    if velocity==None:
        particles = pd.read_pickle(f"/data2/Ncluster/particle/particles_{snapNum}.pkl")[['x','y','z']].values
    else:
        particles = pd.read_pickle(f"/data2/Ncluster/particle/particles_{snapNum}.pkl")[['x','y','z','vx','vy','vz']].values

    if type(centering) in [list,np.ndarray]:
        cx,cy,cz = centering

        if cx>60:
            idx_x = particles[:,0]<cx-60
            particles[idx_x,0] += 120
        else:
            idx_x = particles[:,0]>cx+60
            particles[idx_x,0] -= 120

        if cy>60:
            idx_y = particles[:,1]<cy-60
            particles[idx_y,1] += 120
        else:
            idx_y = particles[:,1]>cy+60
            particles[idx_y,1] -= 120

        if cz>60:
            idx_z = particles[:,2]<cz-60
            particles[idx_z,2] += 120
        else:
            idx_z = particles[:,2]>cz+60
            particles[idx_z,2] -= 120
    if boxsize!=None:
        try:
            idx = (np.abs(particles[:,0]-cx) < boxsize[0]/2) & (np.abs(particles[:,1]-cy) < boxsize[1]/2) & (np.abs(particles[:,2]-cz) < boxsize[2]/2)
        except:
            idx = (np.abs(particles[:,0]-cx) < boxsize/2) & (np.abs(particles[:,1]-cy) < boxsize/2) & (np.abs(particles[:,2]-cz) < boxsize/2)
        return particles[idx,:]
    else:
        return particles

def particles_in_cylinder(particles, center, normal, L, R, verbose=True):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Shift the particles by the center of the cylinder
    shifted_particles = particles - center

    # Project the shifted particles onto the normal vector
    projected_particles = np.dot(shifted_particles, normal)

    # Calculate the distance of each particle from the cylinder axis
    distances = np.linalg.norm(shifted_particles - np.outer(projected_particles, normal), axis=1)

    in_cylinder = (distances <= R) & (projected_particles >= -L/2) & (projected_particles <= L/2)

    return in_cylinder

def transform_coordinates(v1, v2, vpar, *transformed):
    """
    parts : particles
    vvec : unit vectors to be transformed about
    """
    T = np.column_stack((v1,v2,vpar))
    
    answer = []
    for tv in transformed:
        tv = np.matmul(np.linalg.inv(T), tv.T).T
        answer.append(tv)
    return answer


def MostMassiveProgenitor(halo):
    if len(halo) != len(np.unique(halo.Snap_idx.values)):
        halo2 = halo.sort_values(by="Mvir", ascending=False).groupby("Snap_idx").head(1).sort_values(by="Snap_idx")
        return halo2
    else:
        return halo
    
##
s = LSSLoader()
s.load_collapsed_structures(cluster=True, halo=False)
#s.load_filaments(file_format="/data2/Ncluster/filament/s%03d_c00.dat.NDnet_s9.up.NDskl.S5000.BRK.a.NDskl")


####### voronoi densities
from scipy.spatial import Voronoi,ConvexHull
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import splrep, BSpline

def delete_halos(particles, halos):
    def delete(x):
        return np.linalg.norm(particles-x[1][['x','y','z']].values, axis=1) > x[1].Rvir*1e-3
    
    AA = np.array(list(map(delete, tqdm(halos.iterrows(), total=len(halos)))))
    return np.all(AA, axis=0)
    
def calc_dist_to_line(line):
    def calc_dist(point):
        dist = np.linalg.norm(point-line, axis=1)
        arg = np.argmin(dist)
        if arg in [0, line.shape[0]-1]:
            return 10000.
        else:
            return dist[arg]
    return calc_dist

def calc_distarg_to_line(line):
    def calc_dist(point):
        dist = np.linalg.norm(point-line, axis=1)
        arg = np.argmin(dist)
        if arg in [0, line.shape[0]-1]:
            return 10000.
        else:
            return arg
    return calc_dist

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

from scipy.interpolate import CubicSpline

def do_interpolation(coords):
    x,y,z = coords.T

    # Parametric variable
    t = np.arange(coords.shape[0])

    # Interpolation functions for each dimension
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    # Generating more points for interpolation
    t_new = np.linspace(t.min(), t.max(), 100)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)
    return np.c_[x_new, y_new, z_new]
    


####### Halo Trajectories

from scipy.signal import savgol_filter
def calc_psd(halo, progs):
    lookbacks, rperps = [],[]
    spins = []
    for i,row in halo.iterrows():
        if row.Snap_idx>=50:
            prog_interp = do_interpolation(progs[row.Snap_idx])

            line_fix = calc_dist_to_line(prog_interp)
            ds = line_fix(row[['x','y','z']].values.flatten())
            if ds == 10000:
                print('break')
                break
            lookbacks.append(cosmo.age(0).value-cosmo.age(z=az[row.Snap_idx]).value)
            rperps.append(ds)

    lookbacks = np.array(lookbacks)
    rperps_orig = np.array(rperps)
    rperps = savgol_filter(rperps, window_length=17, polyorder=3, mode='nearest')#np.array(rperps)
    vperps_orig = ((rperps_orig[1:]-rperps_orig[:-1])/(lookbacks[1:]-lookbacks[:-1])*u.Mpc/u.Gyr).to(u.km/u.s).value
    vperps = ((rperps[1:]-rperps[:-1])/(lookbacks[1:]-lookbacks[:-1])*u.Mpc/u.Gyr).to(u.km/u.s).value

    return lookbacks, rperps_orig, rperps, vperps_orig, vperps

def calc_psd_fixed(halo, progs):
    lookbacks, rperps = [],[]
    spins = []
    prog_interp = do_interpolation(progs[168])
    for i,row in halo.iterrows():
        if row.Snap_idx>=50:
            lookbacks.append(cosmo.age(0).value-cosmo.age(z=az[row.Snap_idx]).value)
            rperps.append(np.linalg.norm(prog_interp-row[['x','y','z']].values, axis=1).min())

    lookbacks = np.array(lookbacks)
    rperps_orig = np.array(rperps)
    rperps = savgol_filter(rperps, window_length=17, polyorder=3, mode='nearest')#np.array(rperps)
    vperps_orig = ((rperps_orig[1:]-rperps_orig[:-1])/(lookbacks[1:]-lookbacks[:-1])*u.Mpc/u.Gyr).to(u.km/u.s).value
    vperps = ((rperps[1:]-rperps[:-1])/(lookbacks[1:]-lookbacks[:-1])*u.Mpc/u.Gyr).to(u.km/u.s).value

    return lookbacks, rperps_orig, rperps, vperps_orig, vperps

def select_nearest_pnt(line):
    def select(point):
        dist = np.linalg.norm(point-line, axis=1)
        arg = np.argmin(dist)
        return arg
    return select

def fil_vec(fil, arg, N=20):
    segment1 = fil[int(arg-N),:]
    segment2 = fil[int(arg+N),:]
    vec = segment2-segment1
    vec /= np.linalg.norm(vec)
    return vec

def dot_product(Avec, Bvec):
    return [np.dot(A,B) for A,B in zip(Avec,Bvec)]
    
def calc_spin_alignment(halo, progs):
    halo2 = halo.loc[halo.Snap_idx>=50]
    Jvec = halo2[['Jx','Jy','Jz']].values
    Jvec_norm = np.linalg.norm(Jvec, axis=1)
    args = np.array(list(map( lambda x: select_nearest_pnt(do_interpolation(progs[x[1].Snap_idx]))(x[1][['x','y','z']].values.flatten()), halo2.iterrows()   )))
    rperps = np.array(list(map( lambda x: calc_dist_to_line(do_interpolation(progs[x[1].Snap_idx]))(x[1][['x','y','z']].values.flatten()), halo2.iterrows() )))
    halo2['args'] = args
    fvec = np.array(list(map( lambda x: fil_vec(do_interpolation(progs[x[1].Snap_idx]), x[1].args, N=1), halo2.iterrows() )))
    return Jvec_norm, dot_product(Jvec, fvec)/Jvec_norm


###########
def plot_2d_filament(df, snapNum, filarg, L=5, R=5):
    fil = df.progs[snapNum]
    center = fil[filarg,:]
    par,v1,v2 = find_unit_vectors(fil, filarg, dim=3)

    with open(f"/data2/Ncluster/particle/particles_pos_{snapNum}.bin", "rb") as f:
        particles = np.fromfile(f, np.float32)
    particles = PBC_center(center, particles.reshape(int(particles.shape[0]/3),3))
    with open(f"./cpp/particles_{snapNum}.bin", "rb") as f:
        data = np.fromfile(f, np.float32)
    parts = particles[data==0]
    prow = parts.shape[0]
    idx1 = np.zeros(parts.shape[0]*3,dtype=np.int32)
    del particles, data

    parts_ptr = parts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    center_ptr = center.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    normal_ptr = par.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_lib = ctypes.CDLL("./cpp/fastlib2.so")
    c_lib.particles_in_cylinder.restype = ndpointer(dtype=ctypes.c_int, shape=(prow,))
    idx = c_lib.particles_in_cylinder(parts_ptr, prow, center_ptr, normal_ptr, ctypes.c_float(L), ctypes.c_float(R)).astype(bool)

    del parts_ptr, center_ptr, normal_ptr, c_lib
    parts = parts[idx]

    parts_t, center_t = transform_coordinates(v1, v2, par, parts, center)
    parts_t -= center_t
    center_t -= center_t # becomes [0,0,0]

    fig,axes = plt.subplots(nrows=1,ncols=2,sharey=True, figsize=(10,7), gridspec_kw={'hspace': 0, 'wspace': 0, 'width_ratios':[2*R,L]})
    axes[0].hist2d(parts_t[:,0], parts_t[:,1], bins=200, cmap=plt.get_cmap("gray_r"), norm=mpl.colors.LogNorm())#color='k', alpha=0.4, s=1)
    axes[0].scatter(0,0, color='r', s=100)

    axes[1].hist2d(parts_t[:,2], parts_t[:,1], bins=200, cmap=plt.get_cmap("gray_r"), norm=mpl.colors.LogNorm())
    axes[1].scatter(0,0, color='r', s=100)
    axes[1].label_outer()
    axes[0].set_xlabel("x [Mpc/h]")
    axes[0].set_ylabel("y [Mpc/h]")
    axes[1].set_xlabel('z [Mpc/h]')
    axes[0].set_aspect("equal","box")
    axes[1].set_aspect("equal","box")

    plt.show()