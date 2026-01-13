import ctypes
import numpy as np
import sys; sys.path.append("../")
from import_all import *


for snapNum in range(161,49,-1):
    snapNum = int(snapNum)

    print("Reading particles")
    particles = load_particles(snapNum, [60,60,60])
    particles2 = particles.flatten().astype(np.float32)

    print("Reading halos")
    halos = s.read_raw_halo_file()
    halos = halos[halos.Snap_idx==snapNum].sort_values(by='Mvir', ascending=False)[['x','y','z','Rvir','Mvir']].values
    halos[:,3] *= 1e-3
    halos2 = halos.flatten().astype(np.float32)

    print("types")
    c_float_p1 = ctypes.POINTER(ctypes.c_float)
    c_float_p2 = ctypes.POINTER(ctypes.c_float)
    #return_particles_ptr = return_particles.ctypes.data_as(c_double_p)
    particles_ptr = particles2.ctypes.data_as(c_float_p1)
    halos_ptr = halos2.ctypes.data_as(c_float_p2)

    c_lib = ctypes.CDLL("./picklib.so")
    c_lib.function(particles_ptr, halos_ptr, particles.shape[0], halos.shape[0], snapNum)

    del particles, particles2, halos, halos2, c_float_p1, c_float_p2, particles_ptr, halos_ptr, c_lib
