# programming environment
COMPILER     := nvcc
INCLUDE      := -I. -I/user-environment/linux-neoverse_v2/hdf5-1.14.6-cobrby6yjq7vyf4x7m6wy7aoembrddul/include -I/user-environment/linux-neoverse_v2/gsl-2.8-63ctjbspwlt5rsrkpws4rtl7im3r3iqp/include -I/user-environment/linux-neoverse_v2/fftw-3.3.10-5pionfb6nd6vcu55hgygxbb6tzitbbf2/include -I../LATfield2 -I../hiclass_new/include
LIB          := -L/user-environment/linux-neoverse_v2/hdf5-1.14.6-cobrby6yjq7vyf4x7m6wy7aoembrddul/lib -L/user-environment/linux-neoverse_v2/gsl-2.8-63ctjbspwlt5rsrkpws4rtl7im3r3iqp/lib -lgslcblas -lcufft  -lfftw3 -lm -lhdf5 -lchealpix -lgsl -L/user-environment/linux-neoverse_v2/fftw-3.3.10-5pionfb6nd6vcu55hgygxbb6tzitbbf2/lib -L../hiclass_new -lclass

# local installs HEALPix and CFITSIO
CFITSIO_DIR  := $(HOME)/software/cfitsio
HEALPIX_DIR  := $(HOME)/software/Healpix_3.83

INCLUDE      += -I$(CFITSIO_DIR)/include -I$(HEALPIX_DIR)/include -I$(HEALPIX_DIR)/include/healpix_cxx
LIB          += -L$(CFITSIO_DIR)/lib -L$(HEALPIX_DIR)/lib -lcfitsio -lchealpix
HPXCXXLIB    := -lhealpix_cxx -lcfitsio  

# target and source
EXEC         := gevolution
SOURCE       := main.cu
HEADERS      := $(wildcard *.hpp)

# mandatory compiler settings (LATfield2)
DLATFIELD2   := -DFFT3D -DHDF5

# optional compiler settings (LATfield2)
DLATFIELD2   += -DH5_HAVE_PARALLEL
#DLATFIELD2   += -DEXTERNAL_IO # enables I/O server (use with care)
#DLATFIELD2   += -DSINGLE      # switches to single precision, use LIB -lfftw3f

# optional compiler settings (gevolution)
DGEVOLUTION  := -DPHINONLINEAR
DGEVOLUTION  += -DBENCHMARK
DGEVOLUTION  += -DEXACT_OUTPUT_REDSHIFTS
#DGEVOLUTION  += -DVELOCITY      # enables velocity field utilities
DGEVOLUTION  += -DCOLORTERMINAL
#DGEVOLUTION  += -DCHECK_B
DGEVOLUTION  += -DHAVE_CLASS -DHAVE_HICLASS_BG    # requires LIB -lclass
DGEVOLUTION  += -DHAVE_HEALPIX  # requires LIB -lchealpix
DGEVOLUTION  += -DGRADIENT_ORDER=1
#DGEVOLUTION  += -DDEBUG_ALIGNMENT

# further compiler options
OPT          := -O2 -std=c++17 -g -ccbin mpic++ -arch=sm_90 --extended-lambda -Xcompiler -fopenmp

$(EXEC): $(SOURCE) $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB)

unit-tests: unit_tests.cu $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB) -DGADGET_LENGTH_CONVERSION=1 -DGADGET_VELOCITY_CONVERSION=1
	
lccat: lccat.cpp
	$(COMPILER) $< -o $@ $(OPT) $(DGEVOLUTION) $(INCLUDE)
	
lcmap: lcmap.cpp
	$(COMPILER) $< -o $@ $(OPT) $(DGEVOLUTION) $(INCLUDE) $(LIB) $(HPXCXXLIB)

run-tests: unit-tests
	rm -f test_output_*
	srun -N 1 -n 4 -C gpu -A sm97 --time=5:00 --partition=debug ./unit-tests -n 2 -m 2 -Ngrid 128 -Npcl 2097152 -bench 8

run: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/anourizo/testing/.
	export OMP_NUM_THREADS=72
	export OMP_PLACES=cores
	srun -N 1 -n 4 --cpus-per-task=72 -C gpu -A sm97 --time=2:00 --partition=debug --hint=exclusive --cpu-bind=socket ./gpu-bind.sh /capstor/scratch/cscs/anourizo/testing/$(EXEC) -n 2 -m 2 -s /capstor/scratch/cscs/anourizo/testing/test.ini 


profile: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/anourizo/testing/.
	export OMP_NUM_THREADS=72
	export OMP_PLACES=cores
	srun -N 16 -n 64 --cpus-per-task=72 -C gpu -A sm97 --time=40:00 --partition=normal --hint=exclusive --cpu-bind=socket ./gpu-bind.sh ./nsys_wrapper.sh /capstor/scratch/cscs/anourizo/testing/$(EXEC) -n 8 -m 8 -s /capstor/scratch/cscs/anourizo/testing/test.ini

brun: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/anourizo/testing/.
	export OMP_NUM_THREADS=72
	export OMP_PLACES=cores
	srun -N 16 -n 64 --cpus-per-task=72 -C gpu -A sm97 --time=01:00 --partition=normal --hint=exclusive --cpu-bind=socket ./gpu-bind.sh /capstor/scratch/cscs/anourizo/testing/$(EXEC) -n 8 -m 8 -s /capstor/scratch/cscs/anourizo/testing/test.ini

clean:
	-rm -f $(EXEC) lccat lcmap unit-tests

