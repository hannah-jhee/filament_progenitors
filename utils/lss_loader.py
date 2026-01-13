import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import FortranFile

class GadgetDataset(dict):
    def __getattr__(self,name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)

    def __set_attr(self, name, value):
        self[name] = value

    def append(self, **kwargs):
        if self.keys():
            for key in self.keys():
                self[key] = np.append(self[key], kwargs[key], axis=0)

        else:
            for key in kwargs.keys():
                self[key] = kwargs[key]

class ParticleLoader:
    def __init__(self, basePath="/data2/Ncluster"):
        f = FortranFile(f"{basePath}/snapshot_subfile_boundaryinfo.fbin", "r")
        self.xmin = f.read_reals(np.float32).reshape((64,171)).T
        self.xmax = f.read_reals(np.float32).reshape((64,171)).T
        self.ymin = f.read_reals(np.float32).reshape((64,171)).T
        self.ymax = f.read_reals(np.float32).reshape((64,171)).T
        self.zmin = f.read_reals(np.float32).reshape((64,171)).T
        self.zmax = f.read_reals(np.float32).reshape((64,171)).T
        f.close()

    def _needed_indices(self, snapNum, x, y, z, l):
        self.x, self.y, self.z, self.l = x,y,z,l

        try: 
            xstart = x-l[0]/2; xend = x+l[0]/2
            ystart = y-l[1]/2; yend = y+l[1]/2
            zstart = z-l[2]/2; zend = z+l[2]/2
        except:
            xstart = x-l/2; xend = x+l/2
            ystart = y-l/2; yend = y+l/2
            zstart = z-l/2; zend = z+l/2

        xmin,xmax = self.xmin[snapNum], self.xmax[snapNum]
        ymin,ymax = self.ymin[snapNum], self.ymax[snapNum]
        zmin,zmax = self.zmin[snapNum], self.zmax[snapNum]

        if xstart>=0 and xend<120: # pbc와 관련 없을 때
            xidx = ((xmin-xstart)*(xmax-xstart)<0) | ((xmin-xend)*(xmax-xend)<0) | ((xmin-xstart)*(xmin-xend)<0)
        if xstart<0: # x가 왼쪽 끝에 위치할 때
            xidx = (xmax>xend) | (xstart+120>xmin)
        if xend>=120: # x가 오른쪽 끝에 위치할 때
            xidx = (xmin>xstart) | (xend-120<xmax)

        if ystart>=0 and yend<120: # pbc와 관련 없을 때
            yidx = ((ymin-ystart)*(ymax-ystart)<0) | ((ymin-yend)*(ymax-yend)<0) | ((ymin-ystart)*(ymin-yend)<0)
        if ystart<0: # y가 왼쪽 끝에 위치할 때
            yidx = (ymax>yend) | (ystart+120>ymin)
        if yend>=120: # y가 오른쪽 끝에 위치할 때
            yidx = (ymin>ystart) | (yend-120<ymax)

        if zstart>=0 and zend<120: # pbc와 관련 없을 때
            zidx = ((zmin-zstart)*(zmax-zstart)<0) | ((zmin-zend)*(zmax-zend)<0) | ((zmin-zstart)*(zmin-zend)<0)
        if zstart<0: # y가 왼쪽 끝에 위치할 때
            zidx = (zmax>zend) | (zstart+120>zmin)
        if zend>=120: # y가 오른쪽 끝에 위치할 때
            zidx = (zmin>zstart) | (zend-120<zmax)

        idx = xidx&yidx&zidx

        return np.arange(64)[idx]


    def load_partial_particles(self, snapNum, x, y, z, l, basePath="/data2/Ncluster/"):
        """
        Loads dm particle data from Gadget simulation, from subfiles
        which a halo lives within at a given snapshot.
        x, y, z and l are all in [Mpc/h].
        """

        file_sub_indices = self._needed_indices(snapNum, x, y, z, l)

        data = GadgetDataset()
        for index in file_sub_indices:
            fname = f"{basePath}/{snapNum:03}/snapshot_{snapNum:03}.{index}"
            #tmp_data = GadgetDataset()
            with open(fname, "rb") as f:
                _ = f.read(4)
                n_particles      = np.fromfile(f, dtype=np.int32,   count=6)
                mass_arr         = np.fromfile(f, dtype=np.float64, count=6)
                time             = np.fromfile(f, dtype=np.float64, count=1)
                redshift         = np.fromfile(f, dtype=np.float64, count=1)[0]
                flag_sfr         = np.fromfile(f, dtype=np.int32,   count=1)
                flag_feedback    = np.fromfile(f, dtype=np.int32,   count=1)
                n_particles_tot  = np.fromfile(f, dtype=np.int32,   count=6)
                bytes_left       = 256 - 6*4 - 6*8 - 2*8 - 2*4 - 6*4
                la               = np.fromfile(f, dtype=np.int16,   count=int(bytes_left/2))

                N = sum(n_particles)
                ind = np.where((n_particles>0) * (mass_arr!=0))[0]
                if len(ind)>0: N_with_mass = n_particles[ind].sum()
                else         : N_with_mass = 0

                _ = f.read(8)
                _pos = np.fromfile(f, dtype=np.float32, count=N*3).reshape((N,3))
                
                try: # if the box lengths are different in 3 dim
                    boxidx = ((np.min( [np.abs(_pos[:,0]-x), 120 - np.abs(x - _pos[:,0])], axis=0) < l[0]/2) &\
                              (np.min( [np.abs(_pos[:,1]-y), 120 - np.abs(y - _pos[:,1])], axis=0) < l[1]/2) &\
                              (np.min( [np.abs(_pos[:,2]-z), 120 - np.abs(z - _pos[:,2])], axis=0) < l[2]/2))

                except: # if the same
                    boxidx = ((np.min( [np.abs(_pos[:,0]-x), 120 - np.abs(x - _pos[:,0])], axis=0) < l/2) &\
                              (np.min( [np.abs(_pos[:,1]-y), 120 - np.abs(y - _pos[:,1])], axis=0) < l/2) &\
                              (np.min( [np.abs(_pos[:,2]-z), 120 - np.abs(z - _pos[:,2])], axis=0) < l/2))
                _pos = _pos[boxidx]

                _ = f.read(8)
                _vel = np.fromfile(f, dtype=np.float32, count=N*3).reshape((N,3))[boxidx,:]
                _ = f.read(8)
                _pid  = np.fromfile(f, dtype=np.int32, count=N)[boxidx]
                
                data.append(pos=_pos, vel=_vel, pid=_pid)
        return data

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

def MostMassiveProgenitor(halo):
    if len(halo) != len(np.unique(halo.Snap_idx.values)):
        halo = halo.sort_values(by="Mvir", ascending=False).groupby("Snap_idx").head(1).sort_values(by="Snap_idx")
        return halo
    else:
        return halo

