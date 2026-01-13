#!/home/gpuadmin/.conda/envs/cosmos/bin/python
# run by :: python run_tree_test.py -clsnum 0 -filnum 0
import os,sys
import pickle,argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
sys.path.append("/home/gpuadmin/cosmicweb/src/progenitor_tracing")
import disperse_reader as dr

#basePath = "/data2/disperse_smooth_test_N300_log"
clusters = pd.read_pickle(f"/data2/Ncluster/cluster/cluster_all.pkl").iloc[:47]

def bin2Mpch(bin_coords, clsnum=0, snapNum=168):
    rvir = clusters.RVIR.iat[clsnum]*1e-3
    lbox = rvir*20
    Ovec = clusters[['X','Y','Z']].values[clsnum] - rvir*10
    bin_coords = bin_coords[:,[2,1,0]] / 300 * lbox + Ovec
    return bin_coords

class CWTree:
    def __init__(self, clsnum, starting_snap, min_snap, starting_persistence, starting_filnum,
                 basePath, score_threshold, segment_length, nbin_length,
                 file_format,
                 output_path):
        self.clsnum               = clsnum
        self.starting_snap        = starting_snap
        self.min_snap             = min_snap
        self.starting_persistence = starting_persistence
        self.starting_filnum      = starting_filnum
        self.basePath             = basePath
        self.score_threshold      = score_threshold
        self.segment_length       = segment_length
        self.nbin_length          = nbin_length
        self.file_format          = file_format
        self.output_path          = output_path

        self.output = dict()
        self.output['progs'] = dict()
        self.output['CPs'] = dict()
        self.output['scores'] = dict()
        self.output['file_reference'] = dict() # [cut, filnum1, filnum2]

    def run(self):
        # starting from...
        clsnum         = self.clsnum
        snapNum        = self.starting_snap
        min_snap       = self.min_snap
        persistence    = self.starting_persistence
        filnum         = self.starting_filnum
        threshold      = self.score_threshold
        segment_length = self.segment_length
        output_path    = self.output_path

        print(f"Code is now running...output will be saved at {output_path}")

        filename = self.file_format.format(basePath, snapNum, clsnum, persistence)
        fil_1s = dr.read_filament(filename)
        crit_1s = dr.read_filament_critical(filename)

        filnum_op, descendant, critical = self.node2node_construction(filnum, fil_1s, crit_1s, return_op_filnum=True)
        descendant.loc[:,['x','y','z']] = bin2Mpch(descendant[['x','y','z']].values, clsnum=clsnum, snapNum=snapNum)
        critical.loc[:,['x','y','z']]   = bin2Mpch(critical[['x','y','z']].values, clsnum=clsnum, snapNum=snapNum)

        self.output['progs'][snapNum] = descendant # append
        self.output['CPs'][snapNum] = critical
        self.output['scores'][snapNum] = 1
        self.output['file_reference'][snapNum] = [persistence, (filnum, filnum_op)]
        # HANNAH : node2node_construction()과 select_progenitor()의 output을 .iloc으로 바꾸면 됨.
        # now loop over snapshots
        while True:
            snapNum -= 1
            print(f"Finding a progenitor at snapshot {snapNum}")

            #descendant_intp = self.do_interp(descendant[['x','y','z']].values, segment_length=segment_length)
            result = self.select_progenitor(descendant, snapNum, persistence, threshold)
            if result != None: # persistence, highest_score, progenitor, (progenitor_filnum, friend_node[progenitor_filnum])
                persistence, score, progenitor, critical, filnums = result # filnums = [filnum, friend_filnum]
                self.output['progs'][snapNum] = progenitor
                self.output['CPs'][snapNum] = critical
                self.output['scores'][snapNum] = score
                self.output['file_reference'][snapNum] = [persistence, filnums]
                descendant = progenitor # update descendant to the current progenitor (sequentially)
            else: # if persistence became zero
                print("    [run()] Persistence became ZERO...stopping")
                break

            if snapNum==min_snap: # which is 50
                print(f"    [run()] Reached the mininum snapshot({min_snap})...stopping")
                break
        
        with open(output_path, "wb") as handle:
            pickle.dump(self.output, handle, pickle.HIGHEST_PROTOCOL)


    def node2node_construction(self, filnum, fils, crits, return_op_filnum=False):
        """
        It should return (N,4) array: (x,y,z,CP)
        For CP,
           - 3 : node
           - 2 : saddle
           - 4 : D+1, bifurcation
        Note that DisPerSE filament always starts from a NODE and ends at a SADDLE.
        """
        node2node_crit = pd.DataFrame(columns=crits.columns)

        fil = fils[filnum]
        CP1,CP2 = fil[['CP1','CP2']].values[0]
        crit = crits.iloc[[CP1,CP2]]
        CP1_type = crits.iloc[CP1].type
        CP2_type = crits.iloc[CP2].type

        fil_cp = np.zeros(len(fil))
        fil_cp[0] = CP1_type   # node
        fil_cp[-1] = CP2_type  # saddle
        #fil = np.concatenate([fil_pos.T, [fil_cp.tolist()]], axis=0).T
        fil['CP_type'] = fil_cp
        
        filnum_ops = list(set(crits.iloc[CP2].filamentID) - set([filnum]))
        if len(filnum_ops)==1:
            filnum_op = filnum_ops[0]
            fil_op    = fils[filnum_op]#[['x','y','z']].values
            fil_op_cp = np.zeros(len(fil_op))
            CP1,CP2 = fil_op[['CP1','CP2']].values[0]
            crit_op = crits.iloc[[CP1]]
            
            CP1_type = crits.iloc[CP1].type
            CP2_type = crits.iloc[CP2].type
            fil_op_cp[0] = CP1_type
            fil_op_cp[-1] = CP2_type
            fil_op['CP_type'] = fil_op_cp
            #fil_op = np.concatenate([fil_op.T, [fil_op_cp.tolist()]], axis=0).T
            
            node2node_fil = pd.concat([fil, fil_op[::-1][1:]])#np.r_[fil, fil_op[::-1,:]]
            node2node_crit = pd.concat([crit, crit_op[::-1]])
            if return_op_filnum:
                return filnum_op, node2node_fil.reset_index().drop('index',axis=1), node2node_crit.reset_index().drop('index',axis=1)
            return  node2node_fil.reset_index().drop('index',axis=1), node2node_crit.reset_index().drop('index',axis=1)# (node - saddle - node) constructed !
        elif len(filnum_ops)==0:
            #print('   this bifurcation has no opposite side..')
            if return_op_filnum: return None, fil.reset_index().drop('index',axis=1), crit.reset_index().drop('index',axis=1)
            else: return fil.reset_index().drop('index',axis=1), crit.reset_index().drop('index',axis=1)
        else:
            raise ValueError("The number of opposite filaments too many...needs check")


    def do_interp(self, pos, bc_type='not-a-knot', segment_length=None):
        L = np.linalg.norm(pos[1:,:]-pos[:-1,:], axis=1).sum()
        n_seg = int(L / segment_length) + 1
        ts = np.linspace(0, 1, pos.shape[0])
        ts_new = np.linspace(0, 1, n_seg)
        pos_new = CubicSpline(ts, pos, bc_type=bc_type)(ts_new)
        return pos_new
    
    def calc_similarity(self, pos1, pos2, nbin_length=None):
        L = np.linalg.norm(pos1[1:,:]-pos1[:-1,:], axis=1).sum()
        n_bins = int(L / nbin_length) + 1
        S = []
        for i in range(3):
            min_val = pos1[:,i].min()
            max_val = pos1[:,i].max()

            if max_val-min_val <= 0.04: # Mpc/h
                bins = np.linspace(min_val, min_val+0.04, 2)
            else:
                bins = np.linspace(min_val, max_val, n_bins)
            
            hist1,bins1 = np.histogram(pos1[:,i], bins=bins)
            hist2,bins2 = np.histogram(pos2[:,i], bins=bins)

            hist1 = hist1 / pos1.shape[0]
            hist2 = hist2 / pos2.shape[0]

            score = np.sqrt(hist1*hist2).sum()
            S.append(score)
        return np.asarray(S)

    def select_progenitor(self, descendant, snapNum, persistence, threshold):
        """
        descendant  : (N,3) position array of a descendant
        snapNum     : snapshot number to be investigated
        persistence : -ncut
        threshold   : score threshold, user's choice
        """
        clsnum = self.clsnum
        segment_length = self.segment_length
        nbin_length = self.nbin_length
        #try:
        descendant_intp = self.do_interp(descendant[['x','y','z']].values, segment_length=segment_length)
        #except IndexError as e:
        #    print(descendant)
        #    raise e
        
        progenitor_filename = self.file_format.format(basePath, snapNum, clsnum, persistence)
        progenitor_candidates = dr.read_filament(progenitor_filename)
        progenitor_candidates_critical = dr.read_filament_critical(progenitor_filename)

        processed_filnums = [] # this collects the "other" parts of filaments
        scores = dict()
        progenitor_candidates_node2node = dict()
        progenitor_CP_node2node = dict()
        friend_node = dict()
        for filnum in progenitor_candidates.keys():
            if filnum not in processed_filnums:
                _num, progenitor_candidate, progenitor_CP = self.node2node_construction(filnum, progenitor_candidates, progenitor_candidates_critical,
                                                                                        return_op_filnum=True)
                progenitor_candidate.loc[:,['x','y','z']] = bin2Mpch(progenitor_candidate[['x','y','z']].values, clsnum=clsnum, snapNum=snapNum)
                progenitor_CP.loc[:,['x','y','z']] = bin2Mpch(progenitor_CP[['x','y','z']].values, clsnum=clsnum, snapNum=snapNum)
                processed_filnums.append(_num)
                progenitor_candidate_intp = self.do_interp(progenitor_candidate[['x','y','z']].values, segment_length=segment_length)
                score_arr = self.calc_similarity(descendant_intp, progenitor_candidate_intp, nbin_length)
                score = (score_arr.prod())**(1/3)
                
                scores[filnum] = score
                progenitor_candidates_node2node[filnum] = progenitor_candidate
                progenitor_CP_node2node[filnum] = progenitor_CP
                friend_node[filnum] = _num
        
        progenitor_filnum = max(scores, key=scores.get) # it gives the (key) with the maximum (value)
        highest_score = scores[progenitor_filnum]
        progenitor = progenitor_candidates_node2node[progenitor_filnum]
        critical = progenitor_CP_node2node[progenitor_filnum]

        # Threshold condition
        if scores[progenitor_filnum] >= threshold:
            print(f"    [select_progenitor()] Found a progenitor at persistence={persistence} : S={highest_score:.3f}")
            return persistence, highest_score, progenitor, critical, (progenitor_filnum, friend_node[progenitor_filnum])
        else:
            print(f"    [select_progenitor()] The score ({highest_score:.3f}) is too low --> lowering the persistence level")
            persistence -= 0.1
            persistence = np.round(persistence,1)
            if persistence==0: # If using `nsig`, change this value to ==1 because `nsig` traces the ratio
                return None
            return self.select_progenitor(descendant, snapNum, persistence, threshold)



if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-filnum", dest='filnum', type=int)
    p.add_argument("-clsnum", dest='clsnum', type=int, default=0)
    p.add_argument("-Sthr", dest='Sthr', type=float, default=0.85)

    p.add_argument("-segment_length", dest='segment_length', default='1/40')
    p.add_argument("-nbin_length", dest="nbin_length", default='1/20')
    p.add_argument("-output_dir", dest='output_dir', default="../data_products/progenitors2")
    args = p.parse_args()

    output_dir2 = f"{args.output_dir}/c{args.clsnum:02d}_{args.Sthr}"
    if not os.path.isdir(output_dir2):
        os.makedirs(output_dir2)

    print('Score :%.2f'%args.Sthr)
    basePath = f"/data2/disperse_smooth_test_N300_log"
    fil = CWTree(clsnum=args.clsnum, starting_snap=168, min_snap=50, starting_persistence=1, starting_filnum=args.filnum,
                 basePath=basePath, score_threshold=args.Sthr, segment_length=float(args.segment_length), nbin_length=float(args.nbin_length),
                 #file_format="{}/s{:03d}_c{:02d}_smt_w1.5_auto.fits_c{}.up.NDskl.S010.rmB.a.NDskl",
                 file_format="{}/s{:03d}_c{:02d}_grid.fits_c{}.up.NDskl.S010.rmB.a.NDskl",
                 #output_path=f"../data_products/progenitors/c{args.clsnum:02d}/tree_{args.filnum}_w1.5.dat"
                 output_path=f"{output_dir2}/tree_{args.filnum}_3.pickle"
                 )
    fil.run()
