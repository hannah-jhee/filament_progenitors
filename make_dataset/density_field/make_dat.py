import os
import pandas as pd
import argparse
import subprocess
import numpy as np
import sys
sys.path.append("/home/gpuadmin/cosmicweb/src")
from lss_loader import *
from tqdm import tqdm

s = LSSLoader()
s.load_collapsed_structures(halo=False)


def run(clsnum, snapNum):
    file_name = f"s{snapNum:03d}_c{clsnum:02d}_sq.dat"
    boxsize = 10e-3 * s.clusters.RVIR.iat[clsnum]
    cx,cy,cz = s.clusters[['X','Y','Z']].iloc[clsnum]

#print(f"[] Loading particles at snapshot {snapNum:03d}")
    df = pd.read_pickle(f"/data2/Ncluster/particle/{snapNum:03d}/cls_{clsnum:02d}_dm_particles.snap_{snapNum:03d}.pkl")
#idx = np.linalg.norm(df[['x','y','z']].values - s.clusters[['X','Y','Z']].iloc[clsnum].values, axis=1) < 20e-3*s.clusters.RVIR.iat[clsnum]  # 구형
    idx = (np.abs(df.x.values-cx)<boxsize) & (np.abs(df.y.values-cy)<boxsize) & (np.abs(df.z.values-cz)<boxsize)
    df = df.iloc[idx]
    np.savetxt(file_name, df[['x','y','z']].values, header='px py pz')
    del df
    
for snapNum in tqdm(range(169,49,-1)):
    run(4,snapNum)
#run(0,168)
