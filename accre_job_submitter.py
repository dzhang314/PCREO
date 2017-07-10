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
#SBATCH --time=02:00:00
#SBATCH --job-name=PCreo_Sphere

module load Anaconda3

PARAM_S=<|PARAM_S|>
PARAM_D=<|PARAM_D|>
PARAM_N=<|PARAM_N|>

echo "Creating output directory..."
until mkdir /gpfs23/scratch/zhangdk/pcreo_runs/output_data_${SLURM_JOBID}; do sleep 1; done
until cd /gpfs23/scratch/zhangdk/pcreo_runs/output_data_${SLURM_JOBID}; do sleep 1; done
echo "Compiling PCreo_Sphere for initial run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_DOUBLE_PREC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
echo "Compiling PCreo_Sphere for extension run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
echo "Compiling PCreo_Sphere for cleanup run..."
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_GRAD_DESC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
echo "Performing redundant cleanup run..."
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
echo "PCreo job successfully completed."
"""

def submit_job(s, d, n, k):
    s = float(s)
    d = int(d)
    n = int(n)
    script_name = str(uuid4()) + '.sh'
    with open(script_name, 'w+') as script_file:
        script_file.write(ACCRE_JOB_TEMPLATE.replace(
            '<|PARAM_S|>', str(s)).replace(
            '<|PARAM_D|>', str(d)).replace(
            '<|PARAM_N|>', str(n)))
    subprocess.run(['sbatch', '--array=1:' + str(k), script_name])
    os.remove(script_name)

for n in range(20, 51):
    submit_job(2.0, 3, n, 1000)
