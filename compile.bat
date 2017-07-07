call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.4.210\windows\bin\ipsxe-comp-vars.bat" intel64 vs2015

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL
del /Q bin\*.exe 2> NUL
del /Q *.optrpt 2> NUL

ifort /Qmkl /I"%MKLROOT%"\include ^
/warn:all /wrap-margin- /fast ^
/o bin\pcreo_sphere_bfgs_mkl ^
/fpp /DPCREO_USE_MKL ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort /Qmkl /I"%MKLROOT%"\include ^
/warn:all /wrap-margin- /fast /Qparallel ^
/o bin\pcreo_sphere_bfgs_mkl_par ^
/fpp /DPCREO_USE_MKL ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort /Qmkl /I"%MKLROOT%"\include ^
/warn:all /wrap-margin- /fast ^
/o bin\pcreo_sphere_bfgs_single_mkl ^
/fpp /DPCREO_USE_MKL /DPCREO_SINGLE_PREC ^
src\pcreo_sphere_bfgs_hc.f90 mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort /Qmkl /I"%MKLROOT%"\include ^
/warn:all /wrap-margin- /fast /Qparallel ^
/o bin\pcreo_sphere_bfgs_single_mkl_par ^
/fpp /DPCREO_USE_MKL /DPCREO_SINGLE_PREC ^
src\pcreo_sphere_bfgs_hc.f90 mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast ^
/o bin\pcreo_sphere_bfgs_single ^
/fpp /DPCREO_SINGLE_PREC ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast /Qparallel ^
/o bin\pcreo_sphere_bfgs_single_par ^
/fpp /DPCREO_SINGLE_PREC ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast ^
/o bin\pcreo_sphere_bfgs ^
/fpp ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast /Qparallel ^
/o bin\pcreo_sphere_bfgs_par ^
/fpp ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast ^
/o bin\pcreo_sphere_bfgs_quad ^
/fpp /DPCREO_QUAD_PREC ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort ^
/warn:all /wrap-margin- /fast /Qparallel ^
/o bin\pcreo_sphere_bfgs_quad_par ^
/fpp /DPCREO_QUAD_PREC ^
src\pcreo_sphere_bfgs_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL
