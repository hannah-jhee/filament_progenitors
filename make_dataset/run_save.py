import ctypes
import numpy as np
import sys; sys.path.append("../")
from import_all import *


for snapNum in range(50,51):
    snapNum = int(snapNum)

    print("Reading particles")
    particles = load_particles(snapNum, [60,60,60])
    particles2 = particles.flatten().astype(np.float32)

    print("types")
    c_float_p1 = ctypes.POINTER(ctypes.c_float)
    particles_ptr = particles2.ctypes.data_as(c_float_p1)

    c_lib = ctypes.CDLL("./savelib.so")
    c_lib.function(particles_ptr, particles.shape[0]*3, snapNum)

    del particles, particles2, c_float_p1, particles_ptr, c_lib
