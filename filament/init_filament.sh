#!/bin/bash
#PBS -N auto_skeleton_6
#PBS -l nodes=1:ppn=16
#PBS -l walltime=05:00:00
#PBS -o skeleton_6.out
#PBS -e skeleton_6.err

# Change to the working directory
#cd /data101/hannahj/disperse_smooth_test_N300/

# Load any necessary modules
#module load inteloneapi/2022.1.2
#module load openmpi/4.1.4-intel
module load disperse/0.9.25
export OMP_NUM_THREADS=8

for snapNum in {168} #168 148 128 108 088 068 051
do
    mse s${snapNum}_c04_smt_w1.5_auto.fits -cut 0.5 -upSkl -forceLoops -robustness #-loadMSC s${snapNum}_c00_smt_auto.fits.MSC
    skelconv s${snapNum}_c04_smt_w1.5_auto.fits_c0.5.up.NDskl -smooth 10 -rmBoundary -to NDskl_ascii
done
