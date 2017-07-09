#!/usr/bin/env python3

import glob
import itertools
import os
import platform
import subprocess
import sys
from uuid import uuid4


def platform_select(linux_option, windows_option):
    if platform.system() == 'Linux':
        return linux_option
    elif platform.system() == 'Windows':
        return windows_option
    else:
        raise RuntimeError("Unknown platform")


def rm_glob(glob_str):
    for path in glob.iglob(glob_str):
        os.remove(path)


def compile_pcreo_sphere(exe_name='pcreo_sphere',
                         src_name='src/pcreo_sphere.f90',
                         flags=None, use_mkl=False, parallelize=False,
                         clear_mod_files=True, clear_obj_files=True):
    if clear_mod_files: rm_glob('*.mod')
    if clear_obj_files: rm_glob('*.obj')
    if flags is None: flags = []
    script_name = str(uuid4()) + platform_select('.sh', '.bat')
    with open(script_name, 'w+') as script_file:
        # import Intel Fortran environment variables
        script_file.write(platform_select(
            'module load Intel', # for Lmod systems
            'call "C:/Program Files (x86)/IntelSWTools/'
            'compilers_and_libraries_2017.4.210/windows/'
            'bin\ipsxe-comp-vars.bat" intel64\n'))
        if use_mkl:
            script_file.write(platform_select(
                'ifort -mkl \\\n',
                'ifort /Qmkl ^\n'))
        else:
            script_file.write(platform_select(
                'ifort \\\n',
                'ifort ^\n'))
        if parallelize:
            script_file.write(platform_select(
                '-warn:all -no-wrap-margin -fast -parallel \\\n',
                '/warn:all /wrap-margin- /fast /Qparallel ^\n'))
        else:
            script_file.write(platform_select(
                '-warn:all -no-wrap-margin -fast \\\n',
                '/warn:all /wrap-margin- /fast ^\n'))
        script_file.write(platform_select(
            '-o {0} \\\n', '/o {0} ^\n').format(exe_name))
        flag_line = ' '.join([
            platform_select(' -D', ' /D') + flag for flag in flags])
        script_file.write(platform_select(
            '-fpp{0} \\\n', '/fpp{0} ^\n').format(flag_line))
        script_file.write(src_name + '\n')
    subprocess.run(platform_select(['bash', script_name], [script_name]))
    os.remove(script_name)
    if clear_mod_files: rm_glob('*.mod')
    if clear_obj_files: rm_glob('*.obj')


PCREO_PRECISIONS = {
    ('PCREO_SINGLE_PREC', 'single'),
    ('PCREO_DOUBLE_PREC', 'double'),
    ('PCREO_QUAD_PREC', 'quad'),
}

MKL_PRECISIONS = [
    ('PCREO_SINGLE_PREC', 'single'),
    ('PCREO_DOUBLE_PREC', 'double'),
]

PCREO_ALGORITHMS = [
    ('PCREO_BFGS', 'bfgs'),
    ('PCREO_GRAD_DESC', 'gd'),
]


if __name__ == '__main__':
    if platform.system() == 'Windows': rm_glob('bin/*.exe')
    if len(sys.argv) >= 2 and sys.argv[1].lower() == 'all':
        for alg_flag, alg_name in PCREO_ALGORITHMS:
            for prec_flag, prec_name in MKL_PRECISIONS:
                for options in itertools.product([True, False], repeat=3):
                    exe_name = 'bin/pcreo_sphere'
                    exe_name += '_' + alg_name
                    exe_name += '_' + prec_name
                    exe_name += '_mkl'
                    flags = [alg_flag, prec_flag]
                    parallelize, track_angle, symmetry = options
                    if parallelize:
                        exe_name += '_par'
                    if track_angle:
                        exe_name += '_ang'
                        flags.append("PCREO_TRACK_ANGLE")
                    if symmetry:
                        exe_name += '_sym'
                        flags.append("PCREO_TRACK_ANGLE")
                    compile_pcreo_sphere(exe_name=exe_name, flags=flags,
                                         use_mkl=True, parallelize=parallelize)
            for prec_flag, prec_name in PCREO_PRECISIONS:
                for options in itertools.product([True, False], repeat=3):
                    exe_name = 'bin/pcreo_sphere'
                    exe_name += '_' + alg_name
                    exe_name += '_' + prec_name
                    flags = [alg_flag, prec_flag]
                    parallelize, track_angle, symmetry = options
                    if parallelize:
                        exe_name += '_par'
                    if track_angle:
                        exe_name += '_ang'
                        flags.append("PCREO_TRACK_ANGLE")
                    if symmetry:
                        exe_name += '_sym'
                        flags.append("PCREO_TRACK_ANGLE")
                    compile_pcreo_sphere(exe_name=exe_name, flags=flags,
                                         use_mkl=False, parallelize=parallelize)
    elif len(sys.argv) >= 3:
        compile_pcreo_sphere(exe_name=sys.argv[1], src_name=sys.argv[2],
                             flags=sys.argv[3:], use_mkl=False)
    else:
        print("Unsupported usage")
