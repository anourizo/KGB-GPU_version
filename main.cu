//////////////////////////
// Copyright (c) 2015-2025 Julian Adamek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//  
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//////////////////////////

//////////////////////////
// main.cu
//////////////////////////
// 
// main control sequence of Geneva N-body code with evolution of metric perturbations (gevolution)
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: January 2025
//
//////////////////////////

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <nvtx3/nvToolsExt.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include <omp.h>
#ifdef HAVE_CLASS
#include "class.h"
#undef MAX			// due to macro collision this has to be done BEFORE including LATfield2 headers!
#undef MIN
#endif
#include "LATfield2.hpp"
#include "metadata.hpp"
#include "class_tools.hpp"
#include "tools.hpp"
#include "background.hpp"
#include "Particles_gevolution.hpp"
#include "gevolution.hpp"
#include "ic_basic.hpp"
#include "ic_read.hpp"
#ifdef ICGEN_PREVOLUTION
#include "ic_prevolution.hpp"
#endif
#ifdef ICGEN_FALCONIC
#include "fcn/togevolution.hpp"
#endif
#ifdef ICGEN_RELIC
#include "ic_relic.hpp"
#endif
#include "radiation.hpp"
#include "parser.hpp"
#include "output.hpp"
#include "hibernation.hpp"
#ifdef VELOCITY
#include "velocity.hpp"
#endif

using namespace std;
using namespace LATfield2;

int main(int argc, char **argv)
{
	double a_kgb;	
#ifdef BENCHMARK
	//benchmarking variables
	double ref_time, ref2_time, cycle_start_time;
	double initialization_time;
	double run_time;
	double cycle_time=0;
	double projection_time = 0;
	double snapshot_output_time = 0;
	double spectra_output_time = 0;
	double lightcone_output_time = 0;
	double gravity_solver_time = 0;
	double fft_time = 0;
	int fft_count = 0;   
	double update_q_time = 0;
	int update_q_count = 0;
	double moveParts_time = 0;
	double kgb_update_time=0; 
	int  moveParts_count = 0;
#endif  //BENCHMARK
	
	int n = 0, m = 0;
#ifdef EXTERNAL_IO
	int io_size = 0;
	int io_group_size = 0;
#endif
	
	int cycle = 0, snapcount = 0, pkcount = 0, restartcount = 0, usedparams, numparam = 0, numspecies, done_hij;
	int numsteps_ncdm[MAX_PCL_SPECIES-2];
	long numpts3d;
	int box[3];
	double dtau, dtau_old, dx, tau, a, fourpiG, tmp, start_time;
	double maxvel[MAX_PCL_SPECIES];
	FILE * outfile;
	char filename[2*PARAM_MAX_LENGTH+24];
	string h5filename;
	char * settingsfile = NULL;
	char * precisionfile = NULL;
	parameter * params = NULL;
	metadata sim;
	cosmology cosmo;
	icsettings ic;
	double T00hom = 0.;
	Real phi_hom;

// Ahmad added these lines
	/*#ifdef HAVE_HICLASS_BG
		gsl_interp_accel * acc = gsl_interp_accel_alloc();
		gsl_spline * H_spline = NULL;
		gsl_spline * H_prime_spline = NULL;
		gsl_spline * H_prime_prime_spline = NULL;
		gsl_spline * rho_cdm_spline = NULL;
		gsl_spline * rho_b_spline = NULL;
		gsl_spline * rho_g_spline = NULL;
		gsl_spline * rho_crit_spline = NULL;
		gsl_spline * rho_ur_spline = NULL;
		gsl_spline * cs2_spline = NULL;
		gsl_spline * cs2_prime_spline = NULL;
		gsl_spline * rho_smg_spline = NULL;
		gsl_spline * rho_smg_prime_spline = NULL;
		gsl_spline * p_smg_spline = NULL;
		gsl_spline * p_smg_prime_spline = NULL;
		gsl_spline * alpha_K_spline = NULL;
		gsl_spline * alpha_K_prime_spline = NULL;
  		gsl_spline * alpha_B_spline = NULL;
  		gsl_spline * alpha_B_prime_spline = NULL;
		gsl_spline * cs2num_spline = NULL;
		gsl_spline * kin_D_spline = NULL;
		gsl_spline * lambda_2_spline = NULL;
	#endif*/
// Ahmad end

#ifdef ANISOTROPIC_EXPANSION
	Real hij_hom[5] = {0.,0.,0.,0.,0.}; // hij_hom = {h_00, h_01, h_02, h_11, h_12} - only the symmetric part of the tensor is stored, h_33 = -h_00-h_11 due to the tracelessness condition
	Real hijprime_hom[5] = {0.,0.,0.,0.,0.}; // hijprime_hom = {h_00', h_01', h_02', h_11', h_12'}
#endif

#ifndef H5_DEBUG
	H5Eset_auto2 (H5E_DEFAULT, NULL, NULL);
#endif
	
	for (int i = 1 ; i < argc ; i++){
		if ( argv[i][0] != '-' )
			continue;
		switch(argv[i][1]) {
			case 's':
				settingsfile = argv[++i]; //settings file name
				break;
			case 'n':
				n = atoi(argv[++i]); //size of the dim 1 of the processor grid
				break;
			case 'm':
				m =  atoi(argv[++i]); //size of the dim 2 of the processor grid
				break;
			case 'p':
#ifndef HAVE_CLASS
				cout << "HAVE_CLASS needs to be set at compilation to use CLASS precision files" << endl;
				exit(-100);
#endif
				precisionfile = argv[++i];
				break;
			case 'i':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#else
				io_size =  atoi(argv[++i]);
#endif
				break;
			case 'g':
#ifndef EXTERNAL_IO
				cout << "EXTERNAL_IO needs to be set at compilation to use the I/O server"<<endl;
				exit(-1000);
#else
				io_group_size = atoi(argv[++i]);
#endif
		}
	}

#ifndef EXTERNAL_IO
	parallel.initialize(n,m);
#else
	if (!io_size || !io_group_size)
	{
		cout << "invalid number of I/O tasks and group sizes for I/O server (-DEXTERNAL_IO)" << endl;
		exit(-1000);
	}
	parallel.initialize(n,m,io_size,io_group_size);
	if(parallel.isIO()) ioserver.start();
	else
	{
#endif

	COUT << COLORTEXT_CYAN << endl;
	COUT << "                                                      "<<endl;
	COUT <<COLORTEXT_LIGHT_BROWN<<"KKKKKKKKK    KKKKKKK             GGGGGGGGGGGGG     BBBBBBBBBBBBBBBBB" <<COLORTEXT_CYAN  <<endl; 
	COUT <<"K:::::::K    K:::::K          GGG::::::::::::G     B::::::::::::::::B"  <<endl;
	COUT << "K:::::::K    K:::::K        GG:::::::::::::::G     B::::::BBBBBB:::::B"  <<endl;
	COUT <<"K:::::::K   K::::::K       G:::::GGGGGGGG::::G     BB:::::B     B:::::B"<<endl;
	COUT <<"KK::::::K  K:::::KKK      G:::::G       GGGGGG     B:::::B     B:::::B"<<endl;
	COUT <<"K:::::K K:::::K        G:::::G                     B::::B     B:::::B"  <<endl;
	COUT <<"K::::::K:::::K         G:::::G                     B::::BBBBBB:::::B"   <<endl;
	COUT <<"K:::::::::::K          G:::::G      GGGGGGGGGG     B:::::::::::::BB"  <<COLORTEXT_RESET <<" - evolution" << COLORTEXT_CYAN <<endl; 
	COUT <<"K:::::::::::K          G:::::G      G::::::::G     B::::BBBBBB:::::B"   <<endl;
	COUT <<"K::::::K:::::K         G:::::G      GGGGG::::G     B::::B     B:::::B"  <<endl;
	COUT <<"K:::::K K:::::K        G:::::G          G::::G     B::::B     B:::::B"  << COLORTEXT_CYAN << endl;
	COUT <<"KK::::::K  K:::::KKK      G:::::G       G::::G     B:::::B     B:::::B"<<endl;
	COUT <<"K:::::::K   K::::::K       G:::::GGGGGGGG::::G     BB:::::BBBBBB::::::B"<<endl;
	COUT <<"K:::::::K    K:::::K        GG:::::::::::::::G     B:::::::::::::::::B" <<endl;
	COUT <<"K:::::::K    K:::::K          GGG::::::GGG:::G     B::::::::::::::::B"  <<endl;
	COUT <<COLORTEXT_LIGHT_BROWN<<"KKKKKKKKK    KKKKKKK            GGGGGGGGGGGGGG     BBBBBBBBBBBBBBBBB"   <<endl;
	COUT <<COLORTEXT_RESET << endl;

	COUT << " version 2.0 alpha    running on " << n*m << " tasks with " << omp_get_max_threads() << " OpenMP threads per task." << endl;
	

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no CUDA-capable device found!" << endl;
		parallel.abortForce();
	}

	for (int device = 0; device < deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		COUT << " Device " << device << ": " << deviceProp.name << " with " << deviceProp.multiProcessorCount << " SMs, CC " << deviceProp.major << "." << deviceProp.minor << ", global memory " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << endl << endl;
	}
	
#if GRADIENT_ORDER > 1
	COUT << " compiled with GRADIENT_ORDER=" << GRADIENT_ORDER << endl;
#endif
#ifdef CIC_PROJECT_TIJ
	COUT << " compiled with CIC_PROJECT_TIJ" << endl;
#endif
	
	if (settingsfile == NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no settings file specified!" << endl;
		parallel.abortForce();
	}
	
	COUT << " initializing..." << endl;
	
	start_time = MPI_Wtime();
	
	numparam = loadParameterFile(settingsfile, params);
	
	usedparams = parseMetadata(params, numparam, sim, cosmo, ic);
	
	COUT << " parsing of settings file completed. " << numparam << " parameters found, " << usedparams << " were used." << endl;
	
	sprintf(filename, "%s%s_settings_used.ini", sim.output_path, sim.basename_generic);
	saveParameterFile(filename, params, numparam);
	
	free(params);

#ifdef HAVE_CLASS
	cosmo.Hspline = NULL;

	background class_background;
  	perturbs class_perturbs;
  	
  	if (precisionfile != NULL)
	  	numparam = loadParameterFile(precisionfile, params);
	else
#endif
		numparam = 0;


	// Ahmad added these lines

/*#ifdef HAVE_HICLASS_BG
	initializeCLASSstructures(sim, ic, cosmo, class_background, class_perturbs, params, numparam);
	loadBGFunctions(class_background, H_spline, "H [1/Mpc]", sim.z_in);
	loadBGFunctions(class_background, H_prime_spline, "H_prime", sim.z_in);
  	loadBGFunctions(class_background, H_prime_prime_spline, "H_prime_prime", sim.z_in);
	loadBGFunctions(class_background, rho_cdm_spline, "(.)rho_cdm", sim.z_in);
	loadBGFunctions(class_background, rho_b_spline, "(.)rho_b", sim.z_in);
	loadBGFunctions(class_background, rho_g_spline, "(.)rho_g", sim.z_in);
	loadBGFunctions(class_background, rho_crit_spline, "(.)rho_crit", sim.z_in);
	loadBGFunctions(class_background, rho_ur_spline, "(.)rho_ur", sim.z_in);
	loadBGFunctions(class_background, cs2_spline, "c_s^2", sim.z_in);
	loadBGFunctions(class_background, cs2_prime_spline, "c_s^2_prime", sim.z_in);
	loadBGFunctions(class_background, rho_smg_spline, "(.)rho_smg", sim.z_in);
	loadBGFunctions(class_background, rho_smg_prime_spline, "(.)rho_smg_prime", sim.z_in);
	loadBGFunctions(class_background, p_smg_spline, "(.)p_smg", sim.z_in);
	loadBGFunctions(class_background, p_smg_prime_spline, "(.)p_smg_prime", sim.z_in);
	loadBGFunctions(class_background, alpha_K_spline, "kineticity_smg", sim.z_in);
	loadBGFunctions(class_background, alpha_K_prime_spline, "kineticity_prime_smg", sim.z_in);
	loadBGFunctions(class_background, alpha_B_spline, "braiding_smg", sim.z_in);
	loadBGFunctions(class_background, alpha_B_prime_spline, "braiding_prime_smg", sim.z_in);
	loadBGFunctions(class_background, cs2num_spline, "cs2num", sim.z_in);
	loadBGFunctions(class_background, kin_D_spline, "kin (D)", sim.z_in);
	loadBGFunctions(class_background, lambda_2_spline, "lambda_2", sim.z_in);
#endif */

	// Ahmad end 
	
	h5filename.reserve(2*PARAM_MAX_LENGTH);
	h5filename.assign(sim.output_path);
	
	box[0] = sim.numpts;
	box[1] = sim.numpts;
	box[2] = sim.numpts;
	
	Lattice lat(3,box,GRADIENT_ORDER);
	Lattice latFT;
	latFT.initializeRealFFT(lat,0);
	
	perfParticles_gevolution<part_simple,part_simple_info> pcls_cdm;
	perfParticles_gevolution<part_simple,part_simple_info> pcls_b;
	Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm = nullptr;
	if (cosmo.num_ncdm > 0) pcls_ncdm = new Particles_gevolution<part_simple,part_simple_info,part_simple_dataType>[cosmo.num_ncdm];

	Field<Real> * update_cdm_fields[3];
	Field<Real> * update_b_fields[3];
	Field<Real> * update_ncdm_fields[3];
	Field<Real> * project_Tij_fields[2];
	Field<Real> * project_T0i_fields[2];
	double f_params[7] = {0., 0., 0., 0., 0., 0., 0.};
	set<long> ** IDbacklog;

	IDbacklog = new set<long> * [sim.num_IDlogs];
	for (int i = 0; i < sim.num_IDlogs; i++)
		IDbacklog[i] = new set<long> [MAX_PCL_SPECIES];

	Field<Real> phi;
	Field<Real> source;
	Field<Real> chi;
	Field<Real> Sij;
	Field<Real> Bi;
	Field<Cplx> scalarFT;
	Field<Cplx> SijFT;
	Field<Cplx> BiFT;
	Field<Cplx> * zetaFT = NULL;
	source.initialize(lat,1);
	phi.initialize(lat,1);
	chi.initialize(lat,1);
	scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> plan_source(&source, &scalarFT);
	PlanFFT<Cplx> plan_phi(&phi, &scalarFT);
	PlanFFT<Cplx> plan_chi(&chi, &scalarFT);
	Sij.initialize(lat,3,3,symmetric);
	SijFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_Sij(&Sij, &SijFT);
	Bi.initialize(lat,3);
	BiFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_Bi(&Bi, &BiFT);

	// KGB
	//FIXME: check if all these fields are necessary!
	Field<Real> phi_old;
	Field<Cplx> scalarFT_phi_old;
	phi_old.initialize(lat,1);
	scalarFT_phi_old.initialize(latFT,1);
	PlanFFT<Cplx> plan_phi_old(&phi_old, &scalarFT_phi_old);
	
	Field<Real> zeta_half;
	Field<Cplx> scalarFT_zeta_half;
	zeta_half.initialize(lat,1);
	scalarFT_zeta_half.initialize(latFT,1);
	PlanFFT<Cplx> plan_zeta_half(&zeta_half, &scalarFT_zeta_half);
  
	Field<Real> chi_old;
	Field<Cplx> scalarFT_chi_old;
	chi_old.initialize(lat,1);
	scalarFT_chi_old.initialize(latFT,1);
	PlanFFT<Cplx> plan_chi_old(&chi_old, &scalarFT_chi_old);


	Field<Real> phi_prime;
	Field<Cplx> phi_prime_scalarFT;
	phi_prime.initialize(lat,1);
	phi_prime_scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> phi_prime_plan(&phi_prime, &phi_prime_scalarFT);


	Field<Real> psi_prime;
	Field<Cplx> psi_prime_scalarFT;
	psi_prime.initialize(lat,1);
	psi_prime_scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> psi_prime_plan(&psi_prime, &psi_prime_scalarFT);

	Field<Real> psi_half;
	Field<Cplx> psi_half_scalarFT;
	psi_half.initialize(lat,1);
	psi_half_scalarFT.initialize(latFT,1);
	PlanFFT<Cplx> psi_half_plan(&psi_half, &psi_half_scalarFT);


	Field<Real> pi_k;
	Field<Cplx> scalarFT_pi;
	pi_k.initialize(lat,1);
	scalarFT_pi.initialize(latFT,1);
	PlanFFT<Cplx> plan_pi_k(&pi_k, &scalarFT_pi);

	Field<Real> T00_kgb;
	Field<Cplx> T00_kgbFT;
	T00_kgb.initialize(lat,1);
	T00_kgbFT.initialize(latFT,1);
	PlanFFT<Cplx> plan_T00_kgb(&T00_kgb, &T00_kgbFT);

	Field<Real> T0i_kgb;
	Field<Cplx> T0i_kgbFT;
	T0i_kgb.initialize(lat,3);
	T0i_kgbFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_T0i_kgb(&T0i_kgb, &T0i_kgbFT);

	Field<Real> Tij_kgb;
	Field<Cplx> Tij_kgbFT;
	Tij_kgb.initialize(lat,3,3,symmetric);
	Tij_kgbFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_Tij_kgb(&Tij_kgb, &Tij_kgbFT);	


	Field<Real> deltaPm;
	Field<Cplx> scalarFT_deltaPm;
	deltaPm.initialize(lat,1);
	scalarFT_deltaPm.initialize(latFT,1);
	PlanFFT<Cplx> plan_deltaPm(&deltaPm, &scalarFT_deltaPm);

#ifdef CHECK_B
	Field<Real> Bi_check;
	Field<Cplx> BiFT_check;
	Bi_check.initialize(lat,3);
	BiFT_check.initialize(latFT,3);
	PlanFFT<Cplx> plan_Bi_check(&Bi_check, &BiFT_check);
#endif
#ifdef VELOCITY
	Field<Real> vi;
	Field<Cplx> viFT;
	vi.initialize(lat,3);
	viFT.initialize(latFT,3);
	PlanFFT<Cplx> plan_vi(&vi, &viFT);
	double a_old;
#endif
#ifdef TENSOR_EVOLUTION
	Field<Cplx> hijFT;
	Field<Cplx> hijprimeFT;
	hijFT.initialize(latFT,3,3,symmetric);
	hijprimeFT.initialize(latFT,3,3,symmetric);
	PlanFFT<Cplx> plan_hij(&Sij, &hijFT);
	hijprimeFT.alloc();
#endif

	update_cdm_fields[0] = &phi;
	update_cdm_fields[1] = &chi;
	update_cdm_fields[2] = &Bi;
	
	update_b_fields[0] = &phi;
	update_b_fields[1] = &chi;
	update_b_fields[2] = &Bi;
	
	update_ncdm_fields[0] = &phi;
	update_ncdm_fields[1] = &chi;
	update_ncdm_fields[2] = &Bi;

	project_Tij_fields[0] = &Sij;
	project_Tij_fields[1] = &phi;

	project_T0i_fields[0] = &Bi;
	project_T0i_fields[1] = &phi;
	
	Site x(lat);
	rKSite kFT(latFT);
	
	dx = 1.0 / (double) sim.numpts;
	numpts3d = (long) sim.numpts * (long) sim.numpts * (long) sim.numpts;
	
	for (int i = 0; i < 3; i++) // particles may never move farther than to the adjacent domain
	{
		if (lat.sizeLocal(i)-1 < sim.movelimit)
			sim.movelimit = lat.sizeLocal(i)-1;
	}
	parallel.min(sim.movelimit);

	fourpiG = 1.5 * sim.boxsize * sim.boxsize / C_SPEED_OF_LIGHT / C_SPEED_OF_LIGHT;
	a = 1. / (1. + sim.z_in);
	tau = -1;
	dtau = -1;
	dtau_old = -1;

	nvtxRangePushA("IC generation");
	
	if (ic.generator == ICGEN_BASIC)
		generateIC_basic(sim, ic, cosmo, fourpiG, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &pi_k, &zeta_half, &chi, &Bi, &source, &Sij, &scalarFT, &scalarFT_pi, &scalarFT_zeta_half, &BiFT, &SijFT, &plan_phi, &plan_pi_k, &plan_zeta_half, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, 
#ifdef HAVE_CLASS
		class_background, class_perturbs,
#endif		
		params, numparam); // generates ICs on the fly
	else if (ic.generator == ICGEN_READ_FROM_DISK)
		readIC(sim, ic, cosmo, fourpiG, a, tau, dtau, dtau_old, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, zetaFT, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, cycle, snapcount, pkcount, restartcount, IDbacklog);
#ifdef ICGEN_RELIC
	else if (ic.generator == ICGEN_RELIC)
		generateIC_relic(sim, ic, cosmo, fourpiG, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, zetaFT, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, params, numparam);
#endif
#ifdef ICGEN_PREVOLUTION
	else if (ic.generator == ICGEN_PREVOLUTION)
		generateIC_prevolution(sim, ic, cosmo, fourpiG, a, tau, dtau, dtau_old, &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij, params, numparam);
#endif
#ifdef ICGEN_FALCONIC
	else if (ic.generator == ICGEN_FALCONIC)
		maxvel[0] = generateIC_FalconIC(sim, ic, cosmo, fourpiG, dtau, &pcls_cdm, pcls_ncdm, maxvel+1, &phi, &source, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_source, &plan_chi, &plan_Bi, &plan_source, &plan_Sij);
#endif
	else
	{
		COUT << " error: IC generator not implemented!" << endl;
		parallel.abortForce();
	}

	nvtxRangePop();
	
	if (sim.baryon_flag > 1)
	{
		COUT << " error: baryon_flag > 1 after IC generation, something went wrong in IC generator!" << endl;
		parallel.abortForce();
	}
	
	numspecies = 1 + sim.baryon_flag + cosmo.num_ncdm;	
	parallel.max<double>(maxvel, numspecies);
	
	if (sim.gr_flag > 0)
	{
		for (int i = 0; i < numspecies; i++)
			maxvel[i] /= sqrt(maxvel[i] * maxvel[i] + 1.0);
	}

#ifdef CHECK_B
	if (sim.vector_flag == VECTOR_ELLIPTIC)
	{
		for (kFT.first(); kFT.test(); kFT.next())
		{
			BiFT_check(kFT, 0) = BiFT(kFT, 0);
			BiFT_check(kFT, 1) = BiFT(kFT, 1);
			BiFT_check(kFT, 2) = BiFT(kFT, 2);
		}
	}
#endif
#ifdef VELOCITY
	a_old = a;
	//projection_init(&vi);
	thrust::fill_n(thrust::device, vi.data(), 3*lat.sitesLocalGross(), Real(0));
#endif
#ifdef TENSOR_EVOLUTION
	/*for (kFT.first(); kFT.test(); kFT.next())
	{
		for (int i = 0; i < hijprimeFT.components(); i++)
			hijprimeFT(kFT, i) = Cplx(0,0);
	}*/
	#pragma omp parallel for
	for (long i = 0; i < hijprimeFT.components() * latFT.sitesLocalGross(); i++)
	{
		hijprimeFT.data()[i] = Cplx(0,0);
	}
#endif
	
#ifdef BENCHMARK
	initialization_time = MPI_Wtime() - start_time;
	parallel.sum(initialization_time);
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << " BENCHMARK: " << hourMinSec(initialization_time) << endl << endl;
#else
	COUT << COLORTEXT_GREEN << " initialization complete." << COLORTEXT_RESET << endl << endl;
#endif

#ifdef HAVE_CLASS
	if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
	{
		cosmology cosmo2 = cosmo;
		if (cosmo.Hspline == NULL)
		{
			initializeCLASSstructures(sim, ic, cosmo, class_background, class_perturbs, params, numparam);
			loadBGFunctions(class_background, cosmo.Hspline, "H [1/Mpc]", sim.z_in, sim.boxsize/cosmo.h);
			cosmo.acc_H = gsl_interp_accel_alloc();
		}
		else
		cosmo2.Hspline = NULL;
		loadBGFunctions(class_background, cosmo.tauspline, "conf. time [Mpc]", sim.z_in, cosmo.h/sim.boxsize);
		cosmo.acc_tau = gsl_interp_accel_alloc();
		COUT << "Initial Hubble rate = " << Hconf(a, fourpiG, cosmo2) << " (gevolution), " << Hconf(a, fourpiG, cosmo) << " (CLASS) -- using CLASS" << endl;
		if ((
#ifdef ICGEN_RELIC
			ic.generator == ICGEN_RELIC || 
#endif
			ic.generator == ICGEN_READ_FROM_DISK) && zetaFT != NULL)  // zetaFT contains phi(k) at this point, so we need to divide out the transfer function
		{
			gsl_spline * tk1 = NULL;
			gsl_spline * tk2 = NULL;
			gsl_interp_accel * acc;
			loadTransferFunctions(class_background, class_perturbs, tk1, tk2, NULL, sim.boxsize, (1. / a) - 1., cosmo.h);

			#pragma omp parallel default(shared) firstprivate(kFT) private(acc, tmp)
			{
				acc = gsl_interp_accel_alloc();
				#pragma omp for collapse(2)
				for (int i = 0; i < zetaFT->lattice().sizeLocal(1); i++)
				{
					for (int j = 0; j < zetaFT->lattice().sizeLocal(2); j++)
					{
						if (!kFT.setCoord(0, j + zetaFT->lattice().coordSkip()[0], i + zetaFT->lattice().coordSkip()[1]))
						{
							std::cerr << "proc#" << parallel.rank() << ": Error in setting um zeta! Could not set coordinates at k=(0, " << j + zetaFT->lattice().coordSkip()[0] << ", " << i + zetaFT->lattice().coordSkip()[1] << ")" << std::endl;
						}

						if (kFT.coord(1) < (sim.numpts/2) + 1)
							tmp = kFT.coord(1)*kFT.coord(1);
						else
							tmp = (sim.numpts-kFT.coord(1))*(sim.numpts-kFT.coord(1));
						if (kFT.coord(2) < (sim.numpts/2) + 1)
							tmp += kFT.coord(2)*kFT.coord(2);
						else
							tmp += (sim.numpts-kFT.coord(2))*(sim.numpts-kFT.coord(2));

						for (int z = 0; z < zetaFT->lattice().sizeLocal(0); z++)
						{
							double k2 = z*z + tmp;

							if (k2 > 0)
							{
								k2 = 2. * M_PI * sqrt(k2);
								(*zetaFT)(kFT) /= -gsl_spline_eval(tk1, k2, acc) * numpts3d * M_PI * sqrt(Pk_primordial(k2 * cosmo.h / sim.boxsize, ic) / k2) / k2;
							}
							else
								(*zetaFT)(kFT) = Cplx(0.,0.);

							kFT.next();
						}
					}
				}
				gsl_interp_accel_free(acc);
			}
			
			gsl_spline_free(tk1);
			gsl_spline_free(tk2);
		}
		if (sim.gr_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.) && (ic.generator == ICGEN_BASIC || (ic.generator == ICGEN_READ_FROM_DISK && cycle == 0)))
		{
			prepareFTchiLinear(class_background, class_perturbs, scalarFT, sim, ic, cosmo, fourpiG, a, 1., zetaFT);
			plan_source.execute(FFT_BACKWARD);
			//for (x.first(); x.test(); x.next())
			//	chi(x) += source(x);
			thrust::transform(thrust::device, chi.data(), chi.data() + lat.sitesLocalGross(), source.data(), chi.data(), thrust::plus<Real>());
			chi.updateHalo();
		}
	}
	/*else if (cosmo.Hspline != NULL)
	{
		gsl_spline_free(cosmo.Hspline);
		gsl_interp_accel_free(cosmo.acc_H);
		cosmo.Hspline = NULL;
		freeCLASSstructures(class_background, class_perturbs);
	}*/

	if (numparam > 0) free(params);

#endif

	if (tau < 0.)
		tau = particleHorizon(a, fourpiG, cosmo);
	
	if (dtau < 0.)
	{
		if (sim.Cf * dx < sim.steplimit / Hconf(a, fourpiG, cosmo))
			dtau = sim.Cf * dx;
		else
			dtau = sim.steplimit / Hconf(a, fourpiG, cosmo);
	}

	if (dtau_old < 0.)
		dtau_old = 0.;

	while (true)    // main loop
	{
   old_fields_update(phi, phi_old, chi, chi_old);
	/*for (x.first(); x.test(); x.next())
		{
		phi_old(x) = phi(x);
		chi_old(x) = chi(x);
		}*/

#ifdef BENCHMARK		
		cycle_start_time = MPI_Wtime();
#endif
		// construct stress-energy tensor
		nvtxRangePushA("Construct T00");
		//projection_init(&source);
		thrust::fill_n(thrust::device, source.data(), lat.sitesLocalGross(), Real(0));
#ifdef HAVE_CLASS
		if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
			projection_T00_project(class_background, class_perturbs, source, scalarFT, &plan_source, sim, ic, cosmo, fourpiG, a, 1., zetaFT);
#endif
		if (sim.gr_flag > 0)
		{
			projection_T00_project(&pcls_cdm, &source, a, &phi);
			if (sim.baryon_flag)
				projection_T00_project(&pcls_b, &source, a, &phi);
			
			tmp = 0;
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T00_project(pcls_ncdm+i, &source, a, &phi);
				else if (sim.radiation_flag == 0 || (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] == 0))
				{
					//tmp = bg_ncdm(a, cosmo, i);
					//for(x.first(); x.test(); x.next())
					//	source(x) += tmp;

					tmp += bg_ncdm(a, cosmo, i);
				}
			}

			if (tmp > 0)
			{
				Field<Real> * fieldptr = &source;

				lattice_for_each<<<dim3(source.lattice().sizeLocal(1), source.lattice().sizeLocal(2)), 128>>>(lattice_add_functor(), sim.numpts, &fieldptr, 1, &tmp, nullptr, nullptr);

				cudaDeviceSynchronize();
			}
		}
		else
		{
			scalarProjectionCIC_project(&pcls_cdm, &source);
			if (sim.baryon_flag)
				scalarProjectionCIC_project(&pcls_b, &source);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_deltancdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
					scalarProjectionCIC_project(pcls_ncdm+i, &source);
			}
		}
		projection_T00_comm(&source);
		nvtxRangePop();

		nvtxRangePushA("Zero Tij");
		//projection_init(&Sij);
		thrust::fill_n(thrust::device, Sij.data(), 6*lat.sitesLocalGross(), Real(0));
		nvtxRangePop();

		if (a >= 1. / (sim.z_switch_linearchi + 1.))
		{
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] > 0)
				{
					nvtxRangePushA("Tij projection of ncdm particle species");
#ifdef ANISOTROPIC_EXPANSION
					projection_Tij_project(pcls_ncdm+i, &Sij, a, &phi, 1., hij_hom);
#else
					projection_Tij_project(pcls_ncdm+i, &Sij, a, &phi);
#endif
					nvtxRangePop();
				}
			}
		}

		nvtxRangePushA("offload Tij projection to GPU");
		f_params[0] = a;
		f_params[1] = 1.;
		projection_Tij_project_Async(&pcls_cdm, project_Tij_fields, 2, f_params);
		if (sim.baryon_flag)
			projection_Tij_project_Async(&pcls_b, project_Tij_fields, 2, f_params);
		nvtxRangePop();

		nvtxRangePushA("sync and finalize Tij projection");
		auto success = cudaDeviceSynchronize();

		if (success != cudaSuccess)
		{
			std::cerr << "CUDA kernel failed: " << cudaGetErrorString(success) << std::endl;
        	throw std::runtime_error("Error in CUDA kernel called via projection_Tij_project_Async");
		}

		projection_Tij_comm(&Sij);
		nvtxRangePop();


		// KGB
		if (sim.kgb_source_gravity==1)
		{
			nvtxRangePushA("Compute Tmunu for KGB");
		#ifdef HAVE_HICLASS_BG 

			if (sim.vector_flag == VECTOR_ELLIPTIC)
			{
				projection_Tmunu_kgb(T00_kgb, T0i_kgb, Tij_kgb, dx, a, fourpiG, gsl_spline_eval(cosmo.H_spline, 1., cosmo.acc_H_s), phi, chi, phi_prime, pi_k, zeta_half, deltaPm, source, 
				Hconf(a, fourpiG, cosmo), Hconf_prime(a, fourpiG, cosmo), Hconf_prime_prime(a, fourpiG, cosmo),
				gsl_spline_eval(cosmo.rho_smg_spline, a, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.p_smg_spline, a, cosmo.acc_p_smg), gsl_spline_eval(cosmo.p_smg_prime_spline, a, cosmo.acc_p_smg_prime), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
				gsl_spline_eval(cosmo.alpha_K_spline, a, cosmo.acc_alpha_K), gsl_spline_eval(cosmo.alpha_B_spline, a, cosmo.acc_alpha_B), gsl_spline_eval(cosmo.alpha_K_prime_spline, a, cosmo.acc_alpha_K_prime), 
				gsl_spline_eval(cosmo.alpha_B_prime_spline, a, cosmo.acc_alpha_B_prime));
			}
			else
			{
				projection_Tmunu_kgb(T00_kgb, T0i_kgb, Tij_kgb, dx, a, fourpiG, gsl_spline_eval(cosmo.H_spline, 1., cosmo.acc_H_s), phi, chi, phi_prime, pi_k, zeta_half, deltaPm, source,
				Hconf(a, fourpiG, cosmo), Hconf_prime(a, fourpiG, cosmo), Hconf_prime_prime(a, fourpiG, cosmo),
				gsl_spline_eval(cosmo.rho_smg_spline, a, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.p_smg_spline, a, cosmo.acc_p_smg), gsl_spline_eval(cosmo.p_smg_prime_spline, a, cosmo.acc_p_smg_prime), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
				gsl_spline_eval(cosmo.alpha_K_spline, a, cosmo.acc_alpha_K), gsl_spline_eval(cosmo.alpha_B_spline, a, cosmo.acc_alpha_B), gsl_spline_eval(cosmo.alpha_K_prime_spline, a, cosmo.acc_alpha_K_prime), 
				gsl_spline_eval(cosmo.alpha_B_prime_spline, a, cosmo.acc_alpha_B_prime));
			}
		#else // default KGB-evolution or CLASS // No hiclass BG used
			if (sim.vector_flag == VECTOR_ELLIPTIC)
			{
				projection_Tmunu_kgb(T00_kgb, T0i_kgb, Tij_kgb, dx, a, phi, pi_k, zeta_half, cosmo.Omega_kgb * pow(a , -3. * cosmo.w_kgb) * (1. + cosmo.w_kgb) / (cosmo.cs2_kgb), cosmo.Omega_kgb * pow(a , -3. * cosmo.w_kgb) * (1. + cosmo.w_kgb), cosmo.w_kgb, cosmo.cs2_kgb, Hconf(a, fourpiG,cosmo), sim.NL_kgb, 1);
			}
			else
			{
				projection_Tmunu_kgb(T00_kgb, T0i_kgb, Tij_kgb, dx, a, phi, pi_k, zeta_half, cosmo.Omega_kgb * pow(a , -3. * cosmo.w_kgb) * (1. + cosmo.w_kgb) / (cosmo.cs2_kgb), cosmo.Omega_kgb * pow(a , -3. * cosmo.w_kgb) * (1. + cosmo.w_kgb), cosmo.w_kgb, cosmo.cs2_kgb, Hconf(a, fourpiG,cosmo), sim.NL_kgb, 0);
			}
		#endif

			/*for (x.first(); x.test(); x.next())
			{   // CHECK! the coeffs and etc!
				// The coefficient is because it wanted to to be sourced according to eq C.2 of gevolution paper
				// Note that it is multiplied to dx^2 and is divived by -a^3 because of definition of T00 which is scaled by a^3
				// We have T00 and Tij according to code's units, but source is important to calculate potentials and moving particles.
				// There is coefficient between Tij and Sij as source.
				source(x) += T00_kgb(x);
				if (sim.vector_flag == VECTOR_ELLIPTIC) for (int c=0;c<3;c++) Bi(x,c) += T0i_kgb(x,c);
				for(int c=0;c<6;c++) Sij(x,c) += (2.) * Tij_kgb(x,c);
			}*/

			nvtxRangePop();
		}
		


		
		
#ifdef VELOCITY
		if ((sim.out_pk & MASK_VEL) || (sim.out_snapshot & MASK_VEL))
		{
			//projection_init(&Bi);
			thrust::fill_n(thrust::device, Bi.data(), 3*lat.sitesLocalGross(), Real(0));
            projection_Ti0_project(&pcls_cdm, &Bi, &phi, &chi);
            vertexProjectionCIC_comm(&Bi);
            compute_vi_rescaled(cosmo, &vi, &source, &Bi, a, a_old);
            a_old = a;
		}
#endif
		


/*#ifdef ANISOTROPIC_EXPANSION
		projection_Tij_project(&pcls_cdm, &Sij, a, &phi, 1., hij_hom);
#else
		projection_Tij_project(&pcls_cdm, &Sij, a, &phi);
#endif
		if (sim.baryon_flag)
#ifdef ANISOTROPIC_EXPANSION
			projection_Tij_project(&pcls_b, &Sij, a, &phi, 1., hij_hom);
#else
			projection_Tij_project(&pcls_b, &Sij, a, &phi);
#endif*/


		//projection_Tij_comm(&Sij);
		
#ifdef BENCHMARK 
		projection_time += MPI_Wtime() - cycle_start_time;
		ref_time = MPI_Wtime();
#endif
		
		nvtxRangePushA("Solve phi");
		if (sim.gr_flag > 0)
		{
			if (dtau_old > 0.)
			{
				nvtxRangePushA("prepareFTsource");
				T00hom = prepareFTsource(phi, chi, source, cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo), source, 3. * Hconf(a, fourpiG, cosmo) * dx * dx / dtau_old, fourpiG * dx * dx / a, 3. * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo) * dx * dx);  // prepare nonlinear source for phi update
				T00hom /= (double) numpts3d;
				nvtxRangePop();
			}
			
			if (cycle % CYCLE_INFO_INTERVAL == 0)
			{
				COUT << " cycle " << cycle << ", background information: z = " << (1./a) - 1. << ", average T00 = " << T00hom << ", background model = " << cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo) << endl;
			}
		}

		if (sim.gr_flag == 0 || dtau_old > 0.)
		{
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			nvtxRangePushA("FFT forward source");
			plan_source.execute(FFT_FORWARD);  // go to k-space
			nvtxRangePop();
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count++;
#endif
		}


		
		nvtxRangePushA("solveModifiedPoissonFT");
		if (sim.gr_flag == 0)
		{
			solveModifiedPoissonFT(scalarFT, scalarFT, fourpiG / a);  // Newton: phi update (k-space)
		}
		else if (dtau_old > 0.)
		{
			solveModifiedPoissonFT(scalarFT, scalarFT, 1. / (dx * dx), 3. * Hconf(a, fourpiG, cosmo) / dtau_old);  // phi update (k-space)
		}
		nvtxRangePop();

	
		if (sim.gr_flag == 0 || dtau_old > 0.)
		{		
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif	
			nvtxRangePushA("FFT backward phi");	
			plan_phi.execute(FFT_BACKWARD);	 // go back to position space
			nvtxRangePop();
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count++;
#endif	

			nvtxRangePushA("Update halo phi");
			phi.updateHalo();  // communicate halo values
			nvtxRangePop();
		}

		nvtxRangePop();

		if (kFT.setCoord(0, 0, 0))
			phi_hom = scalarFT(kFT).real();
		
		nvtxRangePushA("Solve chi");
		nvtxRangePushA("prepareFTsource");
		prepareFTsource(phi, Sij, Sij, 2. * fourpiG * dx * dx / a);  // prepare nonlinear source for additional equations
		nvtxRangePop();

#ifdef BENCHMARK
		ref2_time= MPI_Wtime();
#endif		
		nvtxRangePushA("FFT forward Sij");
		plan_Sij.execute(FFT_FORWARD);  // go to k-space
		nvtxRangePop();
#ifdef BENCHMARK
		fft_time += MPI_Wtime() - ref2_time;
		fft_count += 6;
#endif

		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
			nvtxRangePushA("Zero T0i");
			//projection_init(&Bi);
			thrust::fill_n(thrust::device, Bi.data(), 3*lat.sitesLocalGross(), Real(0));
			nvtxRangePop();
			//projection_T0i_project(&pcls_cdm, &Bi, &phi);
			//if (sim.baryon_flag)
			//	projection_T0i_project(&pcls_b, &Bi, &phi);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (a >= 1. / (sim.z_switch_Bncdm[i] + 1.) && sim.numpcl[1+sim.baryon_flag+i] > 0)
				{
					nvtxRangePushA("T0i projection of ncdm particle species");
					projection_T0i_project(pcls_ncdm+i, &Bi, &phi);
					nvtxRangePop();
				}
			}
			//projection_T0i_comm(&Bi);
			nvtxRangePushA("offload T0i projection to GPU");
			f_params[0] = 1.;
			projection_T0i_project_Async(&pcls_cdm, project_T0i_fields, 2, f_params);
			if (sim.baryon_flag)
				projection_T0i_project_Async(&pcls_b, project_T0i_fields, 2, f_params);
			nvtxRangePop();
		}

		nvtxRangePushA("projectFTscalar");
#ifdef HAVE_CLASS
		if (sim.radiation_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.))
		{
			prepareFTchiLinear(class_background, class_perturbs, scalarFT, sim, ic, cosmo, fourpiG, a, 1., zetaFT);
			projectFTscalar(SijFT, scalarFT, 1);
		}
		else
#endif		
		projectFTscalar(SijFT, scalarFT);  // construct chi by scalar projection (k-space)
		nvtxRangePop();

		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
			nvtxRangePushA("sync and finalize T0i projection");
			auto success_T0i = cudaDeviceSynchronize();

			if (success_T0i != cudaSuccess)
			{
				std::cerr << "CUDA kernel failed: " << cudaGetErrorString(success_T0i) << std::endl;
				throw std::runtime_error("Error in CUDA kernel called via projection_T0i_project_Async");
			}

			projection_T0i_comm(&Bi);
			nvtxRangePop();
		}

#ifdef BENCHMARK
		ref2_time= MPI_Wtime();
#endif	
		nvtxRangePushA("FFT backward chi");
		plan_chi.execute(FFT_BACKWARD);	 // go back to position space
		nvtxRangePop();
#ifdef BENCHMARK
		fft_time += MPI_Wtime() - ref2_time;
		fft_count++;
#endif	
		nvtxRangePushA("Update halo chi");
		chi.updateHalo();  // communicate halo values
		nvtxRangePop();
		nvtxRangePop();

		nvtxRangePushA("Solve B (k-space)");
		if (sim.vector_flag == VECTOR_ELLIPTIC)
		{
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif
			nvtxRangePushA("FFT forward T0i");
			plan_Bi.execute(FFT_FORWARD);
			nvtxRangePop();
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count += 3;
#endif
			nvtxRangePushA("projectFTvector");
			projectFTvector(BiFT, BiFT, fourpiG * dx * dx); // solve B using elliptic constraint (k-space)
			nvtxRangePop();
#ifdef CHECK_B
			nvtxRangePushA("evolveFTvector (check)");
			evolveFTvector(SijFT, BiFT_check, a * a * dtau_old);
			nvtxRangePop();
#endif
		}
		else
		{
			nvtxRangePushA("evolveFTvector");
			evolveFTvector(SijFT, BiFT, a * a * dtau_old);  // evolve B using vector projection (k-space)
			nvtxRangePop();
		}
		nvtxRangePop();

		if (sim.gr_flag > 0)
		{
			nvtxRangePushA("Solve B (position space)");
#ifdef BENCHMARK
			ref2_time= MPI_Wtime();
#endif		
			nvtxRangePushA("FFT backward B");	
			plan_Bi.execute(FFT_BACKWARD);  // go back to position space
			nvtxRangePop();
#ifdef BENCHMARK
			fft_time += MPI_Wtime() - ref2_time;
			fft_count += 3;
#endif
			nvtxRangePushA("Update halo B");
			Bi.updateHalo();  // communicate halo values
			nvtxRangePop();
			nvtxRangePop();

#ifdef TENSOR_EVOLUTION
			nvtxRangePushA("Solve hij (k-space)");
			if (cycle == 0)
				projectFTtensor(SijFT, hijFT);
			else
				evolveFTtensor(SijFT, hijFT, hijprimeFT, Hconf(a, fourpiG, cosmo), dtau, dtau_old);
			nvtxRangePop();
#endif

#ifdef ANISOTROPIC_EXPANSION
			if (kFT.setCoord(0, 0, 0))
			{
#ifdef TENSOR_EVOLUTION
				hij_hom[0] = hijFT(kFT, 0, 0).real();
				hij_hom[1] = hijFT(kFT, 0, 1).real();
				hij_hom[2] = hijFT(kFT, 0, 2).real();
				hij_hom[3] = hijFT(kFT, 1, 1).real();
				hij_hom[4] = hijFT(kFT, 1, 2).real();

				hijprime_hom[0] = hijprimeFT(kFT, 0, 0).real();
				hijprime_hom[1] = hijprimeFT(kFT, 0, 1).real();
				hijprime_hom[2] = hijprimeFT(kFT, 0, 2).real();
				hijprime_hom[3] = hijprimeFT(kFT, 1, 1).real();
				hijprime_hom[4] = hijprimeFT(kFT, 1, 2).real();
#else
				for (int i = 0; i < 5; i++)
					hij_hom[i] += hijprime_hom[i] * dtau_old;
				
				hijprime_hom[0] = ((1. - 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo)) * hijprime_hom[0] + (dtau_old + dtau) * (Real(2) * SijFT(kFT, 0, 0).real() - SijFT(kFT, 1, 1).real() - SijFT(kFT, 2, 2).real()) / Real(3) / sim.numpts) / (1. + 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo));
				hijprime_hom[1] = ((1. - 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo)) * hijprime_hom[1] + (dtau_old + dtau) * SijFT(kFT, 0, 1).real() / sim.numpts) / (1. + 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo));
				hijprime_hom[2] = ((1. - 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo)) * hijprime_hom[2] + (dtau_old + dtau) * SijFT(kFT, 0, 2).real() / sim.numpts) / (1. + 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo));
				hijprime_hom[3] = ((1. - 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo)) * hijprime_hom[3] + (dtau_old + dtau) * (Real(2) * SijFT(kFT, 1, 1).real() - SijFT(kFT, 0, 0).real() - SijFT(kFT, 2, 2).real()) / Real(3) / sim.numpts) / (1. + 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo));
				hijprime_hom[4] = ((1. - 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo)) * hijprime_hom[4] + (dtau_old + dtau) * SijFT(kFT, 1, 2).real() / sim.numpts) / (1. + 0.5 * (dtau_old + dtau) * Hconf(a, fourpiG, cosmo));
#endif
			}
			
			parallel.broadcast(hij_hom, 5, 0);
			parallel.broadcast(hijprime_hom, 5, 0);

			// anisotropic expansion parameters for particle updates
			f_params[2] = hij_hom[0];
			f_params[3] = hij_hom[1];
			f_params[4] = hij_hom[2];
			f_params[5] = hij_hom[3];
			f_params[6] = hij_hom[4];
#endif
		}

#ifdef BENCHMARK 
		gravity_solver_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif

		// record some background data
		if (kFT.setCoord(0, 0, 0))
		{
			sprintf(filename, "%s%s_background.dat", sim.output_path, sim.basename_generic);
			outfile = fopen(filename, "a");
			if (outfile == NULL)
			{
				cout << " error opening file for background output!" << endl;
			}
			else
			{
#ifdef ANISOTROPIC_EXPANSION
				if (cycle == 0)
					fprintf(outfile, "# background statistics\n# cycle   tau/boxsize    a             conformal H/H0  phi(k=0)       T00(k=0)       h00(k=0)       h01(k=0)       h02(k=0)       h11(k=0)       h12(k=0)\n");
				fprintf(outfile, " %6d   %e   %e   %e   %e   %e   %e   %e   %e   %e   %e\n", cycle, tau, a, Hconf(a, fourpiG, cosmo) / Hconf(1., fourpiG, cosmo), phi_hom, T00hom, hij_hom[0], hij_hom[1], hij_hom[2], hij_hom[3], hij_hom[4]);
#else
				//if (cycle == 0)
					//fprintf(outfile, "# background statistics\n# cycle   tau/boxsize    a             conformal H/H0  phi(k=0)       T00(k=0)\n");
				//fprintf(outfile, " %6d   %e   %e   %e   %e   %e\n", cycle, tau, a, Hconf(a, fourpiG, cosmo) / Hconf(1., fourpiG, cosmo), phi_hom, T00hom);

				//Ahmad added thee lines
				
    #ifdef HAVE_HICLASS_BG
            if (cycle == 0)
            {
                // Write multi-line header with fixed-width fields
                fprintf(outfile, "# background statistics\n");
                fprintf(outfile, "# All the values except 'a' and 'T00' are computed in the hiclass.\n");
                fprintf(outfile, "# The quantities rho_x, rho_x_prime are in unit of hiclass, T00 is in the k-evolution code's unit\n");		
				fprintf(outfile, "# Constant values at end of file:\n");
				fprintf(outfile, "# fourpiG   = %24e\n", fourpiG);
				fprintf(outfile, "# H0[1/Mpc] = %24e\n", gsl_spline_eval(cosmo.H_spline, 1, cosmo.acc_H_s));
                
                // Header line with fixed-width fields (25 characters each)
                fprintf(outfile, "\n# %-12s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s %-24s\n",
                    "0:cycle",
                    "1:tau/boxsize",
                    "2:a",
					"3:z",
					"4:H [1/Mpc]",
                    "5:Hconf",
                    "6:Hconf_prime",
                    "7:Hconf_prime_prime",
					"8:rho_cdm",
					"9:rho_b",
					"10:rho_g",
					"11:rho_ur",
					"12:rho_crit",
					"13:rho_smg",
                    "14:p_smg",
                    "15:rho_smg_prime",
                    "16:p_smg_prime",
                    "17:alpha_K",
                    "18:alpha_B",
                    "19:alpha_K_prime",
                    "20:alpha_B_prime",
                    "21:cs2",
					"22:cs2num",
					"23:kin (D)",
					"24:phi(k=0)",
					"25:T00_hom"
                );
            }
    
            // Define a format string with fixed-width fields for alignment (25 characters each)
            // Left-align each field using the '-' flag
            const char* format = " %-15d %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e %-24e\n";
    
            // Write the data with alignment
            /*fprintf(outfile, format,
                    cycle,                                                // 0:cycle (int)
                    tau,                                                  // 1:tau/boxsize (double)
                    a,                                                    // 2:a (double)
					1./a-1.,                                              // 3:z
                    gsl_spline_eval(H_spline, a, acc),                    // 4:H [1/Mpc]
                    Hconf(a, fourpiG, cosmo),                             // 5:Hconf
                    Hconf_prime(a, fourpiG, cosmo),                       // 6:Hconf_prime
					Hconf_prime_prime(a, fourpiG, cosmo),                 // 7:Hconf_prime_prime
                    gsl_spline_eval(rho_cdm_spline, a, acc),              // 8:rho_cdm
                    gsl_spline_eval(rho_b_spline, a, acc),                // 9:rho_b
                    gsl_spline_eval(rho_g_spline, a, acc),                // 10:rho_g
                    gsl_spline_eval(rho_ur_spline, a, acc),               // 11:rho_ur
                    gsl_spline_eval(rho_crit_spline, a, acc),             // 12:rho_crit
                    gsl_spline_eval(rho_smg_spline, a, acc),              // 13:rho_smg
                    gsl_spline_eval(p_smg_spline, a, acc),                // 14:p_smg
                    gsl_spline_eval(rho_smg_prime_spline, a, acc),        // 15:rho_smg_prime
                    gsl_spline_eval(p_smg_prime_spline, a, acc),          // 16:p_smg_prime
                    gsl_spline_eval(alpha_K_spline, a, acc),              // 17:alpha_K
                    gsl_spline_eval(alpha_B_spline, a, acc),              // 18:alpha_B
                    gsl_spline_eval(alpha_K_prime_spline, a, acc),        // 19:alpha_K_prime
                    gsl_spline_eval(alpha_B_prime_spline, a, acc),        // 20:alpha_B_prime
					gsl_spline_eval(cs2_spline, a, acc),                  // 21:cs2
                    gsl_spline_eval(cs2num_spline, a, acc),               // 22:cs2num
                    gsl_spline_eval(kin_D_spline, a, acc),                // 23:kin (D)
					phi_hom,                                              // 24:phi(k=0)
					T00hom                                               // 25:T00hom
				); */

			fprintf(outfile, format,
                    cycle,                                                                     // 0:cycle (int)
                    tau,                                                                       // 1:tau/boxsize (double)
                    a,                                                                         // 2:a (double)
					1./a-1.,                                                                   // 3:z
                    gsl_spline_eval(cosmo.H_spline, a, cosmo.acc_H_s),                         // 4:H [1/Mpc]
                    Hconf(a, fourpiG, cosmo),                                                  // 5:Hconf
                    Hconf_prime(a, fourpiG, cosmo),                                            // 6:Hconf_prime
					Hconf_prime_prime(a, fourpiG, cosmo),                                      // 7:Hconf_prime_prime
                    gsl_spline_eval(cosmo.rho_cdm_spline, a, cosmo.acc_rho_cdm),             // 8:rho_cdm
                    gsl_spline_eval(cosmo.rho_b_spline, a, cosmo.acc_rho_b),                 // 9:rho_b
                    gsl_spline_eval(cosmo.rho_g_spline, a, cosmo.acc_rho_g),                 // 10:rho_g
                    gsl_spline_eval(cosmo.rho_ur_spline, a, cosmo.acc_rho_ur),               // 11:rho_ur
                    gsl_spline_eval(cosmo.rho_crit_spline, a, cosmo.acc_rho_crit),           // 12:rho_crit
                    gsl_spline_eval(cosmo.rho_smg_spline, a, cosmo.acc_rho_smg),              // 13:rho_smg
                    gsl_spline_eval(cosmo.p_smg_spline, a, cosmo.acc_p_smg),                  // 14:p_smg
                    gsl_spline_eval(cosmo.rho_smg_prime_spline, a, cosmo.acc_rho_smg_prime),  // 15:rho_smg_prime
                    gsl_spline_eval(cosmo.p_smg_prime_spline, a, cosmo.acc_p_smg_prime),      // 16:p_smg_prime
                    gsl_spline_eval(cosmo.alpha_K_spline, a, cosmo.acc_alpha_K),              // 17:alpha_K
                    gsl_spline_eval(cosmo.alpha_B_spline, a, cosmo.acc_alpha_B),              // 18:alpha_B
                    gsl_spline_eval(cosmo.alpha_K_prime_spline, a, cosmo.acc_alpha_K_prime),  // 19:alpha_K_prime
                    gsl_spline_eval(cosmo.alpha_B_prime_spline, a, cosmo.acc_alpha_B_prime),  // 20:alpha_B_prime
					gsl_spline_eval(cosmo.cs2_spline, a, cosmo.acc_cs2),                      // 21:cs2
                    gsl_spline_eval(cosmo.cs2num_spline, a, cosmo.acc_cs2num),                // 22:cs2num
                    gsl_spline_eval(cosmo.kin_D_spline, a, cosmo.acc_kin_D),                  // 23:kin (D)
					phi_hom,                                                                    // 24:phi(k=0)
					T00hom                                                                      // 25:T00hom
				); 
        #else
        #endif
				//Ahmad end
#endif
				fclose(outfile);
			}
		}
		// done recording background data

		// lightcone output
		nvtxRangePushA("Lightcone output");
		if (sim.num_lightcone > 0)
			writeLightcones(sim, cosmo, fourpiG, a, tau, dtau, dtau_old, maxvel[0], cycle, h5filename + sim.basename_lightcone, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &Sij, &BiFT, &SijFT, &plan_Bi, &plan_Sij, done_hij, IDbacklog);
		else done_hij = 0;
		nvtxRangePop();

#ifdef BENCHMARK
		lightcone_output_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif 

		// snapshot output
		if (snapcount < sim.num_snapshot && 1. / a < sim.z_snapshot[snapcount] + 1.)
		{
			nvtxRangePushA("Snapshot output");
			COUT << COLORTEXT_CYAN << " writing snapshot" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

			writeSnapshots(sim, cosmo, fourpiG, a, dtau_old, done_hij, snapcount, h5filename + sim.basename_snapshot, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi, &Bi, &source, &Sij, &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_chi, &plan_Bi, &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi
#endif
			);

			snapcount++;
			nvtxRangePop();
		}
		
#ifdef BENCHMARK
		snapshot_output_time += MPI_Wtime() - ref_time;
		ref_time = MPI_Wtime();
#endif
		
		// power spectra
		if (pkcount < sim.num_pk && 1. / a < sim.z_pk[pkcount] + 1.)
		{
			nvtxRangePushA("Power spectrum output");
			COUT << COLORTEXT_CYAN << " writing power spectra" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;

			writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#ifdef HAVE_CLASS
				class_background, class_perturbs, ic,
#endif
#ifdef HAVE_HICLASS_BG
				cosmo.H_spline, cosmo.acc_H_s, gsl_spline_eval(cosmo.rho_smg_spline, a, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
#endif
				&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k, &zeta_half, &chi, &Bi, &T00_kgb, &T0i_kgb, &Tij_kgb, &source, &Sij, zetaFT, &scalarFT, &scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_kgbFT, &T0i_kgbFT, &Tij_kgbFT, &SijFT, &plan_phi, &plan_pi_k , &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_kgb, &plan_T0i_kgb, &plan_Tij_kgb , &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi, &viFT, &plan_vi
#endif
#ifdef TENSOR_EVOLUTION
				, &hijFT, &hijprimeFT
#endif
			);
		//Ahmad added
if (sim.out_pk & MASK_PHI_PRIME)
		{
		writeSpectra_phi_prime(sim, cosmo, fourpiG,  a, pkcount,
		#ifdef HAVE_HICLASS_BG
		cosmo.H_spline, cosmo.acc_H_s,
		#endif
		&phi_prime ,&phi_prime_scalarFT ,  &phi_prime_plan);
			}

		//Ahmad end

			pkcount++;
			nvtxRangePop();
		}

#ifdef EXACT_OUTPUT_REDSHIFTS
		tmp = a;
		rungekutta4bg(tmp, fourpiG, cosmo, 0.5 * dtau);
		rungekutta4bg(tmp, fourpiG, cosmo, 0.5 * dtau);

		if (pkcount < sim.num_pk && 1. / tmp < sim.z_pk[pkcount] + 1.)
		{
			nvtxRangePushA("Power spectrum output (exact redshifts)");
			writeSpectra(sim, cosmo, fourpiG, a, pkcount,
#ifdef HAVE_CLASS
				class_background, class_perturbs, ic,
#endif
#ifdef HAVE_HICLASS_BG
				cosmo.H_spline, cosmo.acc_H_s, gsl_spline_eval(cosmo.rho_smg_spline, a, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
#endif
				&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &pi_k, &zeta_half, &chi, &Bi, &T00_kgb, &T0i_kgb, &Tij_kgb, &source, &Sij, zetaFT, &scalarFT, &scalarFT_pi, &scalarFT_zeta_half, &BiFT, &T00_kgbFT, &T0i_kgbFT, &Tij_kgbFT, &SijFT, &plan_phi, &plan_pi_k , &plan_zeta_half, &plan_chi, &plan_Bi, &plan_T00_kgb, &plan_T0i_kgb, &plan_Tij_kgb , &plan_source, &plan_Sij
#ifdef CHECK_B
				, &Bi_check, &BiFT_check, &plan_Bi_check
#endif
#ifdef VELOCITY
				, &vi, &viFT, &plan_vi
#endif
#ifdef TENSOR_EVOLUTION
				, &hijFT, &hijprimeFT
#endif
			);
// Ahmad added
if (sim.out_pk & MASK_PHI_PRIME)
	{
	writeSpectra_phi_prime(sim, cosmo, fourpiG,  a, pkcount,
	#ifdef HAVE_HICLASS_BG
	 cosmo.H_spline, cosmo.acc_H_s,
	#endif
	&phi_prime ,&phi_prime_scalarFT ,  &phi_prime_plan);
	}
// Ahmad end		
			nvtxRangePop();
		}
#endif // EXACT_OUTPUT_REDSHIFTS
		
#ifdef BENCHMARK
		spectra_output_time += MPI_Wtime() - ref_time;
#endif

		if (pkcount >= sim.num_pk && snapcount >= sim.num_snapshot)
		{
			int i;
			for (i = 0; i < sim.num_lightcone; i++)
			{
				if (sim.lightcone[i].z + 1. < 1. / a)
					i = sim.num_lightcone + 1;
			}
			if (i == sim.num_lightcone) break; // simulation complete
		}
		
		// compute number of step subdivisions for ncdm particle updates
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (dtau * maxvel[i+1+sim.baryon_flag] > dx * sim.movelimit)
				numsteps_ncdm[i] = (int) ceil(dtau * maxvel[i+1+sim.baryon_flag] / dx / sim.movelimit);
			else numsteps_ncdm[i] = 1;
		}
		
		if (cycle % CYCLE_INFO_INTERVAL == 0)
		{
			COUT << " cycle " << cycle << ", time integration information: max |v| = " << maxvel[0] << " (cdm Courant factor = " << maxvel[0] * dtau / dx;
			if (sim.baryon_flag)
			{
				COUT << "), baryon max |v| = " << maxvel[1] << " (Courant factor = " << maxvel[1] * dtau / dx;
			}
			
			COUT << "), time step / Hubble time = " << Hconf(a, fourpiG, cosmo) * dtau;
			
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (i == 0)
				{
					COUT << endl << " time step subdivision for ncdm species: ";
				}
				COUT << numsteps_ncdm[i] << " (max |v| = " << maxvel[i+1+sim.baryon_flag] << ")";
				if (i < cosmo.num_ncdm-1)
				{
					COUT << ", ";
				}
			}
			
			COUT << endl;
		}

		// KGB loop start!
		#ifdef BENCHMARK
			ref_time = MPI_Wtime();
		#endif
		
		nvtxRangePushA("KGB update");
		#ifdef HAVE_HICLASS_BG // If we have BG vlaues from hicalss/CLASS!
		
			derivatives_update(dtau_old, cycle, phi, phi_old, chi, chi_old, phi_prime, psi_prime); // The derivatives of phi and psi computed at step n! At cycle 0 they are 0! We should use dtau not dtau_old to be the derivative at the requested time similar to the way we update the background a_n -> a_n+1 where we use dtau!

			a_kgb = a;
			if(cycle==0)
			{
				update_zeta(-dtau/ (2.0 * sim.n_kgb_numsteps), dx, a_kgb, fourpiG, gsl_spline_eval(cosmo.H_spline, 1., cosmo.acc_H_s), phi, chi, phi_prime, pi_k, zeta_half, deltaPm, 
				Hconf(a_kgb, fourpiG, cosmo), Hconf_prime(a_kgb, fourpiG, cosmo), Hconf_prime_prime(a_kgb, fourpiG, cosmo),
				 gsl_spline_eval(cosmo.rho_smg_spline, a_kgb, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.p_smg_spline, a_kgb, cosmo.acc_p_smg), gsl_spline_eval(cosmo.p_smg_prime_spline, a_kgb, cosmo.acc_p_smg_prime), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
				 gsl_spline_eval(cosmo.alpha_K_spline, a_kgb, cosmo.acc_alpha_K), gsl_spline_eval(cosmo.alpha_B_spline, a_kgb, cosmo.acc_alpha_B), gsl_spline_eval(cosmo.alpha_K_prime_spline, a_kgb, cosmo.acc_alpha_K_prime), 
				 gsl_spline_eval(cosmo.alpha_B_prime_spline, a_kgb, cosmo.acc_alpha_B_prime), sim.NL_kgb);
				zeta_half.updateHalo();
			}
			for (int i=0;i<sim.n_kgb_numsteps;i++)
			{
				update_zeta(dtau/ sim.n_kgb_numsteps, dx, a_kgb, fourpiG, gsl_spline_eval(cosmo.H_spline, 1., cosmo.acc_H_s), phi, chi, phi_prime, pi_k, zeta_half, deltaPm, 
				Hconf(a_kgb, fourpiG, cosmo), Hconf_prime(a_kgb, fourpiG, cosmo), Hconf_prime_prime(a_kgb, fourpiG, cosmo),
				 gsl_spline_eval(cosmo.rho_smg_spline, a_kgb, cosmo.acc_rho_smg), gsl_spline_eval(cosmo.p_smg_spline, a_kgb, cosmo.acc_p_smg), gsl_spline_eval(cosmo.p_smg_prime_spline, a_kgb, cosmo.acc_p_smg_prime), gsl_spline_eval(cosmo.rho_crit_spline, 1., cosmo.acc_rho_crit),
				 gsl_spline_eval(cosmo.alpha_K_spline, a_kgb, cosmo.acc_alpha_K), gsl_spline_eval(cosmo.alpha_B_spline, a_kgb, cosmo.acc_alpha_B), gsl_spline_eval(cosmo.alpha_K_prime_spline, a_kgb, cosmo.acc_alpha_K_prime), 
				 gsl_spline_eval(cosmo.alpha_B_prime_spline, a_kgb, cosmo.acc_alpha_B_prime), sim.NL_kgb);
				zeta_half.updateHalo();

				rungekutta4bg(a_kgb, fourpiG, cosmo, dtau  / sim.n_kgb_numsteps / 2.0);
				update_pi(dtau/ sim.n_kgb_numsteps, dtau,  phi, chi, psi_prime, pi_k, zeta_half, Hconf(a_kgb, fourpiG, cosmo)); // H_old is updated here in the function
				pi_k.updateHalo();
				rungekutta4bg(a_kgb, fourpiG, cosmo, dtau  / sim.n_kgb_numsteps / 2.0);
			}
		#else // If not HAVE_HICLASS_BG We use  KGB-evolution with w, c_s^2 constants.
			derivatives_update(dtau_old, cycle, phi, phi_old, chi, chi_old, phi_prime, psi_prime); // The derivatives of phi and psi computed at step n! At cycle 0 they are 0! We should use dtau not dtau_old to be the derivative at the requested time similar to the way we update the background a_n -> a_n+1 where we use dtau!
			a_kgb = a;
			//First we update zeta_half to have it at -1/2 just in the first loop
			if(cycle==0)
			{
				update_zeta(-dtau/(2.0 * sim.n_kgb_numsteps) , dx, a_kgb, phi, chi, phi_prime, pi_k, zeta_half, cosmo.cs2_kgb, 0., cosmo.w_kgb, Hconf(a_kgb, fourpiG, cosmo), Hconf_prime(a_kgb, fourpiG, cosmo), sim.NL_kgb);
				zeta_half.updateHalo();
			}
			for (int i=0;i<sim.n_kgb_numsteps;i++)
			{
				update_zeta(dtau/ sim.n_kgb_numsteps, dx, a_kgb, phi, chi, phi_prime, pi_k, zeta_half, cosmo.cs2_kgb, 0., cosmo.w_kgb, Hconf(a_kgb, fourpiG, cosmo), Hconf_prime(a_kgb, fourpiG, cosmo), sim.NL_kgb);
				zeta_half.updateHalo();
				rungekutta4bg(a_kgb, fourpiG, cosmo, dtau/sim.n_kgb_numsteps/2.0);
				update_pi(dtau/ sim.n_kgb_numsteps, dtau, phi, chi, psi_prime, pi_k, zeta_half, Hconf(a_kgb, fourpiG, cosmo)); 
				pi_k.updateHalo();
				rungekutta4bg(a_kgb, fourpiG, cosmo, dtau/sim.n_kgb_numsteps/2.0 );
			}
		#endif // KGB - LeapFrog: End
		nvtxRangePop();
       

		#ifdef BENCHMARK
			kgb_update_time += MPI_Wtime() - ref_time;
			ref_time = MPI_Wtime();
		#endif

		nvtxRangePushA("Particle update: ncdm species");
#ifdef BENCHMARK
		ref2_time = MPI_Wtime();
#endif
		for (int i = 0; i < cosmo.num_ncdm; i++) // non-cold DM particle update
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			
			tmp = a;
			
			for (int j = 0; j < numsteps_ncdm[i]; j++)
			{
				f_params[0] = tmp;
				f_params[1] = tmp * tmp * sim.numpts;
				if (sim.gr_flag > 0)
					maxvel[i+1+sim.baryon_flag] = pcls_ncdm[i].updateVel(update_q, (dtau + dtau_old) / 2. / numsteps_ncdm[i], update_ncdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
				else
					maxvel[i+1+sim.baryon_flag] = pcls_ncdm[i].updateVel(update_q_Newton, (dtau + dtau_old) / 2. / numsteps_ncdm[i], update_ncdm_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);

#ifdef BENCHMARK
				update_q_count++;
				update_q_time += MPI_Wtime() - ref2_time;
				ref2_time = MPI_Wtime();
#endif

				rungekutta4bg(tmp, fourpiG, cosmo, 0.5 * dtau / numsteps_ncdm[i]);
				f_params[0] = tmp;
				f_params[1] = tmp * tmp * sim.numpts;
				
				if (sim.gr_flag > 0)
					pcls_ncdm[i].moveParticles(update_pos, dtau / numsteps_ncdm[i], update_ncdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
				else
					pcls_ncdm[i].moveParticles(update_pos_Newton, dtau / numsteps_ncdm[i], NULL, 0, f_params);
#ifdef BENCHMARK
				moveParts_count++;
				moveParts_time += MPI_Wtime() - ref2_time;
				ref2_time = MPI_Wtime();
#endif
				rungekutta4bg(tmp, fourpiG, cosmo, 0.5 * dtau / numsteps_ncdm[i]);
			}
		}
		nvtxRangePop();

		// cdm and baryon particle update
		nvtxRangePushA("Particle update: cdm and baryons, kick step");
		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			maxvel[0] = pcls_cdm.updateVel(update_q_functor(), (dtau + dtau_old) / 2., update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q_functor(), (dtau + dtau_old) / 2., update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 2), f_params);
		}
		else
		{
			maxvel[0] = pcls_cdm.updateVel(update_q_Newton_functor(), (dtau + dtau_old) / 2., update_cdm_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
			if (sim.baryon_flag)
				maxvel[1] = pcls_b.updateVel(update_q_Newton_functor(), (dtau + dtau_old) / 2., update_b_fields, ((sim.radiation_flag + sim.fluid_flag > 0 && a < 1. / (sim.z_switch_linearchi + 1.)) ? 2 : 1), f_params);
		}
		nvtxRangePop();

#ifdef BENCHMARK
		update_q_count++;
		update_q_time += MPI_Wtime() - ref2_time;
		ref2_time = MPI_Wtime();
#endif
				
		rungekutta4bg(a, fourpiG, cosmo, 0.5 * dtau);  // evolve background by half a time step

		nvtxRangePushA("Particle update: cdm and baryons, drift step");
		f_params[0] = a;
		f_params[1] = a * a * sim.numpts;
		if (sim.gr_flag > 0)
		{
			pcls_cdm.moveParticles(update_pos_functor(), dtau, update_cdm_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos_functor(), dtau, update_b_fields, (1. / a < ic.z_relax + 1. ? 3 : 0), f_params);
		}
		else
		{
			pcls_cdm.moveParticles(update_pos_Newton_functor(), dtau, NULL, 0, f_params);
			if (sim.baryon_flag)
				pcls_b.moveParticles(update_pos_Newton_functor(), dtau, NULL, 0, f_params);
		}
		nvtxRangePop();

#ifdef BENCHMARK
		moveParts_count++;
		moveParts_time += MPI_Wtime() - ref2_time;
#endif

		rungekutta4bg(a, fourpiG, cosmo, 0.5 * dtau);  // evolve background by half a time step
		
		parallel.max<double>(maxvel, numspecies);
		
		if (sim.gr_flag > 0)
		{
			for (int i = 0; i < numspecies; i++)
				maxvel[i] /= sqrt(maxvel[i] * maxvel[i] + 1.0);
		}
		// done particle update
		
		tau += dtau;
		
		if (sim.wallclocklimit > 0.)   // check for wallclock time limit
		{
			tmp = MPI_Wtime() - start_time;
			parallel.max(tmp);
			if (tmp > sim.wallclocklimit)   // hibernate
			{
				COUT << COLORTEXT_YELLOW << " reaching hibernation wallclock limit, hibernating..." << COLORTEXT_RESET << endl;
				COUT << COLORTEXT_CYAN << " writing hibernation point" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
				if (sim.vector_flag == VECTOR_PARABOLIC && sim.gr_flag == 0)
					plan_Bi.execute(FFT_BACKWARD);
#ifdef CHECK_B
				if (sim.vector_flag == VECTOR_ELLIPTIC)
				{
					plan_Bi_check.execute(FFT_BACKWARD);
					//hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, chi, Bi_check, a, tau, dtau, cycle); // FIXME
				}
				else
#endif
				//hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, chi, Bi, a, tau, dtau, cycle); // FIXME
				break;
			}
		}
		
		if (restartcount < sim.num_restart && 1. / a < sim.z_restart[restartcount] + 1.)
		{
			COUT << COLORTEXT_CYAN << " writing hibernation point" << COLORTEXT_RESET << " at z = " << ((1./a) - 1.) <<  " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
			if (sim.vector_flag == VECTOR_PARABOLIC && sim.gr_flag == 0)
				plan_Bi.execute(FFT_BACKWARD);
#ifdef CHECK_B
			if (sim.vector_flag == VECTOR_ELLIPTIC)
			{
				plan_Bi_check.execute(FFT_BACKWARD);
				//hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, chi, Bi_check, a, tau, dtau, cycle, restartcount);  // FIXME
			}
			else
#endif
			//hibernate(sim, ic, cosmo, &pcls_cdm, &pcls_b, pcls_ncdm, phi, chi, Bi, a, tau, dtau, cycle, restartcount);  // FIXME
			restartcount++;
		}
		
		dtau_old = dtau;
		
		if (sim.Cf * dx < sim.steplimit / Hconf(a, fourpiG, cosmo))
			dtau = sim.Cf * dx;
		else
			dtau = sim.steplimit / Hconf(a, fourpiG, cosmo);
		   
		cycle++;
		
#ifdef BENCHMARK
		cycle_time += MPI_Wtime()-cycle_start_time;
#endif
	}
	
	COUT << COLORTEXT_GREEN << " simulation complete." << COLORTEXT_RESET << endl;

#ifdef BENCHMARK
		ref_time = MPI_Wtime();
#endif

if (zetaFT != NULL)
	delete[] zetaFT;

for (int i = 0; i < sim.num_IDlogs; i++)
{
	if (IDbacklog[i] != NULL)
		delete[] IDbacklog[i];
}
delete [] IDbacklog;

#ifdef HAVE_CLASS
	if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
		freeCLASSstructures(class_background, class_perturbs);
#endif

#ifdef BENCHMARK
	lightcone_output_time += MPI_Wtime() - ref_time;
	run_time = MPI_Wtime() - start_time;

	parallel.sum(run_time);
	parallel.sum(cycle_time);
	parallel.sum(projection_time);
	parallel.sum(snapshot_output_time);
	parallel.sum(spectra_output_time);
	parallel.sum(lightcone_output_time);
	parallel.sum(gravity_solver_time);
	parallel.sum(fft_time);
	parallel.sum(update_q_time);
	parallel.sum(moveParts_time);
	
	COUT << endl << "BENCHMARK" << endl;   
	COUT << "total execution time  : "<<hourMinSec(run_time) << endl;
	COUT << "total number of cycles: "<< cycle << endl;
	COUT << "time consumption breakdown:" << endl;
	COUT << "initialization   : "  << hourMinSec(initialization_time) << " ; " << 100. * initialization_time/run_time <<"%."<<endl;
	COUT << "main loop        : "  << hourMinSec(cycle_time) << " ; " << 100. * cycle_time/run_time <<"%."<<endl;
	
	COUT << "----------- main loop: components -----------"<<endl;
	COUT << "projections                : "<< hourMinSec(projection_time) << " ; " << 100. * projection_time/cycle_time <<"%."<<endl;
	COUT << "snapshot outputs           : "<< hourMinSec(snapshot_output_time) << " ; " << 100. * snapshot_output_time/cycle_time <<"%."<<endl;
	COUT << "lightcone outputs          : "<< hourMinSec(lightcone_output_time) << " ; " << 100. * lightcone_output_time/cycle_time <<"%."<<endl;
	COUT << "power spectra outputs      : "<< hourMinSec(spectra_output_time) << " ; " << 100. * spectra_output_time/cycle_time <<"%."<<endl;
	COUT << "update momenta (count: "<<update_q_count <<"): "<< hourMinSec(update_q_time) << " ; " << 100. * update_q_time/cycle_time <<"%."<<endl;
	COUT << "move particles (count: "<< moveParts_count <<"): "<< hourMinSec(moveParts_time) << " ; " << 100. * moveParts_time/cycle_time <<"%."<<endl;
	COUT << "gravity solver             : "<< hourMinSec(gravity_solver_time) << " ; " << 100. * gravity_solver_time/cycle_time <<"%."<<endl;
	COUT << "-- thereof Fast Fourier Transforms (count: " << fft_count <<"): "<< hourMinSec(fft_time) << " ; " << 100. * fft_time/gravity_solver_time <<"%."<<endl;
#endif

#ifdef EXTERNAL_IO	
		ioserver.stop();
	}
#endif

	if (cosmo.num_ncdm > 0) delete[] pcls_ncdm;

	parallel.finalize();

	return 0;
}

