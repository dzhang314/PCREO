#!/bin/bash

source /opt/intel/parallel_studio_xe_2017/psxevars.sh

ifort -warn all -heap-arrays -no-wrap-margin -fast -parallel -qopt-report=5 -o pcreo_sphere_gd_hc src/pcreo_sphere_gd_hc.f90
ifort -warn all -mkl -static-intel -heap-arrays -no-wrap-margin -fast -parallel -qopt-report=5 -o pcreo_sphere_bfgs_hc src/pcreo_sphere_bfgs_hc.f90
