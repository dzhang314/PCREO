#!/usr/bin/env python3

import os
import subprocess
os.chdir('/home/zhangdk/pcreo_runs')
COMPLETION_MESSAGE = "PCreo job successfully completed. Exiting."


def job_completed(log_path):
    with open(log_path) as log_file:
        lines = log_file.readlines()
    return len(lines) > 0 and lines[-1].strip('\n') == COMPLETION_MESSAGE


def starts_with_digit(s):
    return len(s) > 0 and s[0].isdigit()


def log_path(slurm_job_id):
    return '/home/zhangdk/pcreo_runs/slurm-{0}.out'.format(slurm_job_id)


def output_path(slurm_job_id):
    return '/scratch/zhangdk/pcreo_runs/output_data_{0}/pcreo_input.csv'.format(slurm_job_id)


def output_directory(slurm_job_id):
    return '/scratch/zhangdk/pcreo_runs/output_data_{0}'.format(slurm_job_id)


def completed_data_path(slurm_job_id):
    return '/home/zhangdk/completed_pcreo_runs/pcreo-{0}.out'.format(slurm_job_id)


def read_parameter(log_file_lines, param_name):
    param_values = [line.strip()[len(param_name):].strip()
                    for line in log_file_lines
                    if line.strip().startswith(param_name)]
    return max(param_values, key=len)


def dict_append(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
    return None


for dir_entry in os.scandir():
    if dir_entry.is_file(follow_symlinks=False) \
            and dir_entry.name.startswith('slurm-') \
            and dir_entry.name.endswith('.out') \
            and job_completed(dir_entry):
        slurm_job_id = dir_entry.name[6:-4]
        #print("Querying sacct for raw ID of job {0}...".format(slurm_job_id))
        #sacct_info = subprocess.run(
        #    ['sacct', '--format=JobID,JobIDRaw', '--parsable2',
        #     '--job', slurm_job_id],
        #    stdout=subprocess.PIPE)
        #raw_job_id = sacct_info.stdout.decode('utf-8').split('\n')[1].split('|')[1]
        #print("Received raw ID", raw_job_id)
        raw_job_id = slurm_job_id
        with open(dir_entry) as slurm_file:
            slurm_lines = slurm_file.readlines()
        print("Read SLURM log file", dir_entry.name)
        try: num_points = read_parameter(slurm_lines, "Total number of points:")
        except ValueError: continue
        sph_dim = read_parameter(slurm_lines, "Sphere dimension:")
        s_val = read_parameter(slurm_lines,
                               "Value of s (Riesz potential power parameter):")
        numeric_lines = [line.strip().split(' | ')
                         for line in slurm_lines
                         if starts_with_digit(line.strip())]
        energy = numeric_lines[-1][1].strip()
        rms_gradient = numeric_lines[-1][2].strip()
        with open(output_path(raw_job_id)) as output_file:
            output_data = output_file.read()
        print("Read PCreo output file", output_path(raw_job_id))
        with open(completed_data_path(slurm_job_id), 'w+') as data_file:
            data_file.write('{0}, {1}, {2}\n{3}, {4}\n{5}'.format(
                sph_dim, s_val, num_points, energy, rms_gradient, output_data))
        print("Wrote PCreo data file", completed_data_path(slurm_job_id))
        os.remove(dir_entry)
        os.remove(output_path(raw_job_id))
        os.rmdir(output_directory(raw_job_id))
        print("Deleted log and output files")
