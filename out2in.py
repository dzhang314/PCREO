#!/usr/bin/env python3

import glob
import os

out_files = glob.glob('pcreo_??????????.csv')
out_files.sort()
if len(out_files) == 0:
    print("WARNING: No PCreo output files found.")
else:
    os.rename(out_files[-1], 'pcreo_input.csv')
    for name in out_files[:-1]:
        os.remove(name)
