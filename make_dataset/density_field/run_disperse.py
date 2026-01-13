import os
import time
import argparse
import subprocess

def run(clsnum, snapNum):
    file_name = f"s{snapNum:03d}_c{clsnum:02d}_smt.fits"
    
    if not os.path.isfile(f"{file_name}.MSC"):
        _ = subprocess.run(f"mse {file_name} -cut 1e6 -upSkl -forceLoops -robustness ".split())
    else:
        _ = subprocess.run(f"mse {file_name} -cut 1e6 -upSkl -forceLoops -robustness -loadMSC {file_name}.MSC".split())
    _ = subprocess.run(f"skelconv {file_name}_c1e+06.up.NDskl -smooth 10 -rmBoundary -to NDskl_ascii".split())
#_ = subprocess.run(f"skelconv {file_name}_c1e+08.up.NDskl -trimBelow robustness 1e6 -breakdown -assemble 0 20 -smooth 10 -rmBoundary -to NDskl_ascii".split())


if __name__=="__main__":
    for snapNum in [168]:#range(169, 49,-1):
        run(clsnum=0, snapNum=snapNum)
