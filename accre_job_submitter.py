#!/usr/bin/env python3

import os
import subprocess
from uuid import uuid4

ACCRE_JOB_TEMPLATE = """\
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=PCreo_Sphere
#SBATCH --output=slurm-%j.out

module load Anaconda3

PARAM_S=<|PARAM_S|>
PARAM_D=<|PARAM_D|>
PARAM_N=<|PARAM_N|>

echo "Creating output directory..."
mkdir /tmp/pcreo_output_${SLURM_JOBID}
if [ $? -ne 0 ]
then
    echo "Could not create output directory. Exiting."
    exit 101
fi

echo "Moving to output directory..."
cd /tmp/pcreo_output_${SLURM_JOBID}
if [ $? -ne 0 ]
then
    echo "Could not move to output directory. Exiting."
    exit 102
fi

echo "Compiling PCreo_Sphere for initial run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_1 /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_DOUBLE_PREC PCREO_SYMMETRY PCREO_BFGS \
PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
if [ $? -ne 0 ]
then
    echo "PCreo_Sphere compilation failed. Exiting."
    exit 103
fi

echo "Compiling PCreo_Sphere for extension run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_2 /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_SYMMETRY PCREO_BFGS \
PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
if [ $? -ne 0 ]
then
    echo "PCreo_Sphere compilation failed. Exiting."
    exit 104
fi

echo "Compiling PCreo_Sphere for cleanup run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_3 /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_SYMMETRY PCREO_GRAD_DESC \
PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
if [ $? -ne 0 ]
then
    echo "PCreo_Sphere compilation failed. Exiting."
    exit 105
fi

echo "Performing initial run..."
./pcreo_sphere_exe
mv $(ls pcreo-??????????.csv | sort | tail -n 1) pcreo-input.csv
rm -f pcreo-??????????.csv

echo "Performing extension run..."
./pcreo_sphere_exe
mv $(ls pcreo-??????????.csv | sort | tail -n 1) pcreo-input.csv
rm -f pcreo-??????????.csv

echo "Performing redundant extension run..."
./pcreo_sphere_exe
mv $(ls pcreo-??????????.csv | sort | tail -n 1) pcreo-input.csv
rm -f pcreo-??????????.csv

echo "Performing cleanup run..."
./pcreo_sphere_exe
mv $(ls pcreo-??????????.csv | sort | tail -n 1) pcreo-input.csv
rm -f pcreo-??????????.csv

echo "Performing redundant cleanup run..."
./pcreo_sphere_exe
mkdir -p /scratch/zhangdk/pcreo_runs
mv $(ls pcreo-??????????.csv | sort | tail -n 1) /scratch/zhangdk/pcreo_runs/pcreo-${SLURM_JOBID}.csv
rm -f pcreo-??????????.csv

rm -f pcreo_sphere_1 pcreo_sphere_2 pcreo_sphere_3
cd /tmp
rmdir /tmp/pcreo_output_${SLURM_JOBID}

echo "PCreo job successfully completed. Exiting."
"""


def submit_job(s, d, n, k=1):
    s = float(s)
    d = int(d)
    n = int(n)
    script_name = str(uuid4()) + '.sh'
    with open(script_name, 'w+') as script_file:
        script_file.write(ACCRE_JOB_TEMPLATE.replace(
            '<|PARAM_S|>', str(s)).replace(
            '<|PARAM_D|>', str(d)).replace(
            '<|PARAM_N|>', str(n)))
    if k == 1:
        subprocess.run(['sbatch', script_name])
    else:
        subprocess.run(['sbatch', '--array=1-' + str(k), script_name])
    os.remove(script_name)


for n in range(10, 11):
    submit_job(1.0, 2, n, k=2)
