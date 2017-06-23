call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.4.210\windows\bin\ipsxe-comp-vars.bat" intel64 vs2015

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL
del /Q bin\*.exe 2> NUL
del /Q *.optrpt 2> NUL

ifort /Qmkl /I"%MKLROOT%"\include /Qopt-report:5 /warn:all /wrap-margin- /o bin\pcreo_sphere_bfgs_hc /fast src\pcreo_sphere_bfgs_hc.f90 mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL

ifort /Qopt-report:5 /warn:all /wrap-margin- /o bin\pcreo_sphere_gd_hc /fast src\pcreo_sphere_gd_hc.f90

del /Q *.mod 2> NUL
del /Q src\*.mod 2> NUL
del /Q *.obj 2> NUL