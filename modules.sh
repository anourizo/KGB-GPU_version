#!/bin/bash

uenv start --view=modules prgenv-gnu/25.6:v2 
module load cray-mpich 
module load cuda 
module load fftw 
module load gcc 
module load gsl
module load hdf5

