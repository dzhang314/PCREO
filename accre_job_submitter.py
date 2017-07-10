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
#SBATCH --job-name=pcreo

module load Anaconda3

PARAM_S=<|PARAM_S|>
PARAM_D=<|PARAM_D|>
PARAM_N=<|PARAM_N|>

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
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_GRAD_DESC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
/home/zhangdk/pcreo/compile.py ./pcreo_sphere_exe /home/zhangdk/pcreo/src/pcreo_sphere.f90 \
PCREO_QUAD_PREC PCREO_GRAD_DESC PCREO_PARAM_S=${PARAM_S}_rk PCREO_PARAM_D=${PARAM_D} PCREO_PARAM_N=${PARAM_N}
./pcreo_sphere_exe
rm ./pcreo_sphere_exe
/home/zhangdk/pcreo/out2in.py
echo "PCreo job successfully completed"
"""

def submit_job(s, d, n):
    s = float(s)
    d = int(d)
    n = int(n)
    script_name = str(uuid4()) + '.sh'
    with open(script_name, 'w+') as script_file:
        script_file.write(ACCRE_JOB_TEMPLATE.replace(
            '<|PARAM_S|>', str(s)).replace(
            '<|PARAM_D|>', str(d)).replace(
            '<|PARAM_N|>', str(n)))
    subprocess.run(['sbatch', script_name])
    os.remove(script_name)

for _ in range(20):
    submit_job(2.0, 3, 100)
