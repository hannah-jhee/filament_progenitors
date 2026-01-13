import os
import subprocess
import pandas as pd

#/data2/disperse_smooth_test_N300/s168_c00_smt_w1.0_auto.fits_c0.5.up.NDskl
def read_filament(filename):
    if not os.path.isfile(filename):
        """
        This part must be modified for your filament data files.
        """
        print(f"Can't find skeleton file {filename} ... now running DisPerSE")
        path_split = filename.split('/')
        basePath = os.path.join('/',*path_split[:-1])
        snapNum  = int(path_split[-1][1:4])
        clsnum   = int(path_split[-1][6:8])
        nsig     = path_split[-1][path_split[-1].index('fits_c')+6:path_split[-1].index('.up')]

        run_disperse(basePath, clsnum, snapNum, nsig, smooth=10)#, assemble=None)"""
        
    cls = 0
    columns = 'CP1 CP2 x y z'.split()
    fil_dataset = dict()

    lines = open(filename).readlines()
    
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

def read_filament_critical(filename):
    if not os.path.isfile(filename):
        print(f"Can't find skeleton file {filename} ... now running DisPerSE")
        run_disperse()
    columns = 'type x y z value pairID boundary connectivity destination filamentID'.split()
    crit_dataset = []
    
    lines = open(filename).readlines()

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

def run_disperse(basePath, clsnum, snapNum, nsig, smooth=10):
    """
    It runs DisPerSE : delaunay_3D -> mse -> skelconv.
    It only acts when a skeleton file doesn't exist.
    """
    filename = f"{basePath}/s{snapNum:03d}_c{clsnum:02d}_grid.fits"

    # fits file
    if os.path.isfile(f"{filename}"):
        print(f"    [disperse] Found {filename} ... Skipping generating smoothed density field")
    else:
        print(f"    [disperse] Cannot find {filename}")
        print(f"    [disperse] Smoothing the density field...")
        subprocess.run(f"python {basePath}/make_fits.py -clsnum {clsnum} -snapNum {snapNum} -nbins=300 -verbose".split())

    # mse
    if os.path.isfile(f"{filename}_c{nsig}.up.NDskl"):
        print(f"    [disperse] Found {filename}_c{nsig}.up.NDskl ... Skipping mse")
    else:
        print(f"    [disperse] Cannot find {filename}_c{nsig}.up.NDskl...")
        if os.path.isfile(f"{filename}.MSC"):
            print(f"    [disperse] Found {filename}.MSC file")
            subprocess.run(f"mse {filename} -periodicity 0 -cut {nsig} -upSkl -forceLoops -robustness -outDir {basePath} -loadMSC {filename}.MSC".split())
        else:
            print(f"    [disperse] Cannot find {filename}.MSC file")
            subprocess.run(f"mse {filename} -periodicity 0 -cut {nsig} -upSkl -forceLoops -robustness -outDir {basePath}".split())
    
    # skelconv
    print(f"{filename}_c{nsig}.up.NDskl.S{smooth:03d}.rmB.a.NDskl")
    if os.path.isfile(f"{filename}_c{nsig}.up.NDskl.S{smooth:03d}.rmB.a.NDskl"):
        print(f"    [disperse] Found {filename}_c{nsig}.up.NDskl.S{smooth:03d}.rmB.a.NDskl ... Skipping skelconv")
    else:
        print(f"    [disperse] Cannot find {filename}_c{nsig}.up.NDskl.S{smooth:03d}.rmB.a.NDskl")
        subprocess.run(f"skelconv {filename}_c{nsig}.up.NDskl -smooth {smooth} -rmBoundary -outDir {basePath} -to NDskl_ascii".split())
