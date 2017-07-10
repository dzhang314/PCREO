#!/usr/bin/env python3

import glob
import subprocess

while True:
    names = glob.glob('slurm-*.out')
    for name in names:
        with open(name) as f:
            lines = f.readlines()
        print(name, ': ', sep='', end='')
        if len(lines) > 0:
            print(lines[-1].strip('\n'))
        else:
            print()
    for _ in range(len(names)):
        print('\033[F', end='')
