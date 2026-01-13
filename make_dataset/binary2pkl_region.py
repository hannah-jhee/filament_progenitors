import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import FortranFile

import sys
sys.path.append("../utils")
from lss_loader import *
from tqdm import tqdm

def main():
    #args.snapNum = snapNum
    snapNum = 168

    s = LSSLoader()
    p = ParticleLoader()
    s.load_collapsed_structures(halo=False) # only clusters

    print("Loading particles")
    data = p.load_partial_particles(snapNum=snapNum, x=60, y=60, z=60, l=120)
    print("done")
    
    for clsnum in range(1): # top 10 halos
        cx,cy,cz,cr = s.clusters[['X','Y','Z','RVIR']].iloc[clsnum]
        cr *= 1e-3 # cMpc/h
        x_pos,y_pos,z_pos = data.pos.T
        if cx>60:
            idx_x = data.pos[:,0] < cx-60
            x_pos[idx_x] += 120
        else:
            idx_x = data.pos[:,0] > cx+60
            x_pos[idx_x] -= 120
        del idx_x

        if cy>60:
            idx_y = data.pos[:,1] < cy-60
            y_pos[idx_y] += 120
        else:
            idx_y = data.pos[:,1] > cy+60
            y_pos[idx_y] -= 120
        del idx_y


        if cz>60:
            idx_z = data.pos[:,2] < cz-60
            z_pos[idx_z] += 120
        else:
            idx_z = data.pos[:,2] > cz+60
            z_pos[idx_z] -= 120
        del idx_z

        #idx = np.linalg.norm(np.c_[x_pos-cx, y_pos-cy, z_pos-cz], axis=1) <= 1*cr
        idx = (np.abs(x_pos-cx)<20*cr) & (np.abs(y_pos-cy)<20*cr) & (np.abs(z_pos-cz)<20*cr) # it returns 40Rvir^3 box!
        df = pd.DataFrame(columns='x y z vx vy vz pid'.split(),\
                          data=np.c_[x_pos[idx], y_pos[idx], z_pos[idx], data.vel[idx,:], data.pid[idx]])
        df.to_pickle(f"{args.output_path}/{snapNum:03d}/cls_{clsnum:02d}_dm_particles.snap_{snapNum:03d}.pkl")
        
    

if __name__=="__main__":
    main()