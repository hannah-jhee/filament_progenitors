# run by :: python run_tree_smt.py -filnum 7 -basePath /data2/disperse_smooth_test_N300 -smoothing_weight 1.5 -clsnum 0
# From the original version, this version includes consideration on different smoothing weight (1.5) when saving the output files
import argparse
import sys
import pandas as pd
sys.path.append("/home/hannahj/cosmicweb2/src/filament")
import disperse as dr
import numpy as np
from scipy.interpolate import CubicSpline

basePath = "/data101/hannahj/disperse_smooth_test_N300"

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

class FilTree:
    def __init__(self, smoothing_weight=1.37, starting_snap=168, starting_nsig=9, smoothing=10, clsnum=0, starting_filnum=41, basePath="/data101/hannahj/disperse_smooth_test_N300",
                 threshold=0.7, file_format="{}/s{:03d}_c{:02d}_smt_w{}_auto.fits_c{}.up.NDskl.S{:03d}.rmB.a.NDskl", output="./tree.dat"):
                 
        self.smoothing_weight= smoothing_weight
        self.starting_snap   = starting_snap
        self.starting_nsig   = starting_nsig
        self.smoothing       = smoothing
        self.clsnum          = clsnum
        self.starting_filnum = starting_filnum
        self.basePath        = basePath
        self.file_format     = file_format
        self.threshold       = threshold
        self.output          = output

        self.filnum = self.starting_filnum

        filename = self.file_format.format(self.basePath, self.starting_snap, self.clsnum, self.smoothing_weight, self.starting_nsig, self.smoothing)
        print(filename)
        
        self.progs = dict()
        self.progs[self.starting_snap] = dr.read_filament(filename)[self.starting_filnum][['x','y','z']].values
        self.write_tree_file(self.starting_snap, self.starting_nsig, 1.0, self.progs[self.starting_snap], self.filnum)

    def run_tree(self):
        snapNum = self.starting_snap
        nsig = self.starting_nsig
        maxkey = self.starting_filnum
        fil1 = self.progs[snapNum]
        while True:
            snapNum -= 1
            print(f"Finding Progenitor at snapshot {snapNum}")
            result = self.select_progenitor(fil1, snapNum, nsig, threshold=self.threshold)
            #nsig, scores, maxkey = result
            nsig, score, prog, maxkey = result
            self.progs[snapNum] = prog
            fil1 = do_interpolation(self.progs[snapNum])

            self.write_tree_file(snapNum, nsig, score, prog, maxkey)
            print(f"{snapNum} :: {score:.4f}")
            if snapNum == 50:
                break
        

    def bhattacharyya_coefficient(self, data1, data2, nbins, return_bins=False):
        min_val = data1.min()
        max_val = data1.max()

        bins = np.linspace(min_val, max_val, nbins)
        hist1,bins1 = np.histogram(data1, bins=bins)
        hist2,bins2 = np.histogram(data2, bins=bins)

        hist1 = hist1 / data1.shape[0]
        hist2 = hist2 / data2.shape[0]

        score = np.sqrt(hist1*hist2).sum()
        if return_bins == True:
            return score, bins, hist1, hist2
        return score

    def select_progenitor(self, fil1, snapNum, nsig, threshold=0.7):
        fil1 = do_interpolation(fil1)
        if nsig%1 == 0:
            fil2_filename = self.file_format.format(self.basePath, snapNum, self.clsnum, self.smoothing_weight, int(nsig), self.smoothing)
        else:
            fil2_filename = self.file_format.format(self.basePath, snapNum, self.clsnum, self.smoothing_weight, nsig, self.smoothing)
        fil2s = dr.read_filament(fil2_filename)

        scores = dict()
        for key in fil2s.keys():
            fil2 = do_interpolation(fil2s[key][['x','y','z']].values)
            score = []

            for i in range(3):
                score.append(self.bhattacharyya_coefficient(fil1[:,i], fil2[:,i], int(0.1*len(fil1))))
            #score.append((score[0]*score[1]*score[2])**(1/3))
            scores[key] = (score[0]*score[1]*score[2])**(1/3) #score #.append(score)
        maxkey = max(scores, key=scores.get)

        # Threshold condition
        if scores[maxkey] >= threshold:
            print(f"    Found progenitor at (nsig={nsig})")
            prog = fil2s[maxkey][['x','y','z']].values

            L_fil1 = np.linalg.norm(fil1[1:,:] - fil1[:-1,:], axis=1)
            L_fil2 = np.linalg.norm(prog[1:,:] - prog[:-1,:], axis=1)
            L_fil1 = L_fil1[L_fil1<1].sum()
            L_fil2 = L_fil2[L_fil2<1].sum()   # 앞뒤가 연결된 경우가 있어서 처리하기
            if L_fil2/L_fil1 > 1.6 : # threshold를 통과해도 길이가 길면 persistence level을 낮춰야 한다.
                nsig -= 0.1
                nsig = np.round(nsig,1)
                print(f"    Progenitor is longer (prog/desc = {L_fil2/L_fil1}) --> lowering persistence level to {nsig} to get over the missing node")
                return self.select_progenitor(fil1, snapNum, nsig)
            else:
                return nsig, scores[maxkey], fil2s[maxkey][['x','y','z']].values, maxkey
        
        # 점수가 낮으면
        else:
            print(f"    Found progenitor but score ({scores[maxkey]:.3f}) is too low... Lowering persistence level")
            nsig -= 0.1
            nsig = np.round(nsig,1)
            return self.select_progenitor(fil1, snapNum, nsig)

    def write_tree_file(self, snapNum, nsig, score, prog, maxkey):
        if snapNum==self.starting_snap:
            f = open(self.output, "w")
        else:
            f = open(self.output, "a")
        
        
        f.write(f"{snapNum} {str(nsig)} {score:.8f} {prog.shape[0]} {maxkey}\n")
        
        for fx,fy,fz in prog:
            f.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")
        f.close()

def read_progenitor(filename):
    with open(filename,"r") as f:
        lines = f.readlines()
    line_num = 0
    
    progs = dict()
    prog_keys = dict()
    while True:
        snapNum, nsig, score, n_fil, maxkey = lines[line_num].split()
        snapNum = int(snapNum)
        nsig = np.round(float(nsig),1)
        score = float(score)
        n_fil = int(n_fil)
        pos = lines[line_num+1:line_num+n_fil+1]
        pos = np.array(list(map(lambda x: list(map(float, x.split())), pos)))
        progs[snapNum] = pos
        prog_keys[snapNum] = maxkey
        line_num += n_fil + 1
        if snapNum == 51:
            break
    
    return progs, prog_keys


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-filnum", dest='filnum', type=int)
    p.add_argument("-basePath", dest='basePath', default='/data2/disperse_smooth_test_N300')
    p.add_argument("-smoothing_weight", dest='smoothing_weight', type=float, default=1.5)
    p.add_argument("-clsnum", dest='clsnum', type=int, default=0)
    args = p.parse_args()
    fil = FilTree(smoothing_weight=args.smoothing_weight, smoothing=10, clsnum=args.clsnum, basePath=args.basePath, starting_nsig=0.5, starting_filnum=args.filnum, threshold=0.75, output=f"/home/gpuadmin/cosmicweb2/data_products/progenitors/c{args.clsnum:02d}/tree_{args.filnum}_w{args.smoothing_weight}.dat")# output=f"./tree_output/tree_{args.filnum}.dat")
    fil.run_tree()
