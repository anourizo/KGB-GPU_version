# programming environment
COMPILER     := nvcc
INCLUDE      := -I. -I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/hdf5-1.14.5-iyjsbrml3dbr3l7cp65dgeclqlyfcdnn/include -I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/gsl-2.8-pjzdxlsptkmjuvnrxif5x7ellp7rab3c/include -I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/fftw-3.3.10-3yw4wbosrsa2257uitrgpge6a3mfw7ck/include -I../LATfield2 -I/users/adamek/local/include -I../class_public/include -I../class_public/external/HyRec2020 -I../class_public/external/RecfastCLASS -I../class_public/external/heating # -I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cuda-12.6.2-csv6jo3czkfdk46ep7pmm6ipo3yjlbjj/include  # add the path to LATfield2 and other libraries (if necessary)
LIB          := -L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/hdf5-1.14.5-iyjsbrml3dbr3l7cp65dgeclqlyfcdnn/lib -L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/gsl-2.8-pjzdxlsptkmjuvnrxif5x7ellp7rab3c/lib -L/users/adamek/local/lib -L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/fftw-3.3.10-3yw4wbosrsa2257uitrgpge6a3mfw7ck/lib -lfftw3f -lm -lhdf5 -lgsl -lgslcblas -lchealpix -lcfitsio -lclass # -L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cuda-12.6.2-csv6jo3czkfdk46ep7pmm6ipo3yjlbjj/lib64 -lcufft -lcufftw
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
DLATFIELD2   += -DSINGLE      # switches to single precision, use LIB -lfftw3f

# optional compiler settings (gevolution)
DGEVOLUTION  := -DPHINONLINEAR
DGEVOLUTION  += -DBENCHMARK
DGEVOLUTION  += -DEXACT_OUTPUT_REDSHIFTS
#DGEVOLUTION  += -DVELOCITY      # enables velocity field utilities
DGEVOLUTION  += -DCOLORTERMINAL
#DGEVOLUTION  += -DCHECK_B
DGEVOLUTION  += -DHAVE_CLASS    # requires LIB -lclass
DGEVOLUTION  += -DHAVE_HEALPIX  # requires LIB -lchealpix
DGEVOLUTION  += -DGRADIENT_ORDER=2

# further compiler options
OPT          := -O2 -std=c++17 -g -ccbin mpic++ -arch=sm_90 --extended-lambda -Xcompiler -fopenmp

$(EXEC): $(SOURCE) $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB)

unit-tests: unit_tests.cu $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB) -DGADGET_LENGTH_CONVERSION=1 -DGADGET_VELOCITY_CONVERSION=1
	
lccat: lccat.cpp
	$(COMPILER) $< -o $@ $(OPT) $(DGEVOLUTION) $(INCLUDE)
	
lcmap: lcmap.cpp
	$(COMPILER) $< -o $@ $(OPT) -fopenmp $(DGEVOLUTION) $(INCLUDE) $(LIB) $(HPXCXXLIB)

run-tests: unit-tests
	rm -f test_output_*
	srun -N 1 -n 4 -C gpu -A sm97 --time=5:00 --partition=debug ./unit-tests -n 2 -m 2 -Ngrid 128 -Npcl 2097152 -bench 8

run: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/adamek/testing/.
	srun -N 1 -n 4 -C gpu -A sm97 --time=5:00 --partition=debug /capstor/scratch/cscs/adamek/testing/$(EXEC) -n 2 -m 2 -s /capstor/scratch/cscs/adamek/testing/settings.ini

profile: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/adamek/testing/.
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/users/adamek/local/lib
	export OMP_NUM_THREADS=9
	export OMP_PLACES=cores
	srun -N 2 -n 64 --cpus-per-task=9 -C gpu -A sm97 --time=12:00 --partition=debug --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh ./nsys_wrapper.sh /capstor/scratch/cscs/adamek/testing/$(EXEC) -n 8 -m 8 -s /capstor/scratch/cscs/adamek/testing/settings.ini

clean:
	-rm -f $(EXEC) lccat lcmap unit-tests

