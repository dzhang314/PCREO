#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --job-name=pcreo

module load Anaconda3

PARAM_S=2.0
PARAM_D=3
PARAM_N=500

mkdir /gpfs23/scratch/zhangdk/pcreo_runs/output_data_${SLURM_JOBID}
cd /gpfs23/scratch/zhangdk/pcreo_runs/output_data_${SLURM_JOBID}
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_DOUBLE_PREC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
rm ./pcreo_sphere_exe
