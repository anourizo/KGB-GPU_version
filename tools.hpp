//////////////////////////
// tools.hpp
//////////////////////////
// 
// Collection of analysis tools for gevolution
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: February 2025
//
//////////////////////////

#ifndef TOOLS_HEADER
#define TOOLS_HEADER

#include "lattice_loop.hpp"

#ifndef Cplx
#define Cplx Imag
#endif

#define KTYPE_GRID      0
#define KTYPE_LINEAR    1

using namespace std;
using namespace LATfield2;


#ifdef FFT3D
//////////////////////////
// extractCrossSpectrum
//////////////////////////
// Description:
//   generates the cross spectrum for two Fourier images
// 
// Arguments:
//   fld1FT     reference to the first Fourier image for which the cross spectrum should be extracted
//   fld2FT     reference to the second Fourier image for which the cross spectrum should be extracted
//   kbin       allocated array that will contain the central k-value for the bins
//   power      allocated array that will contain the average power in each bin
//   kscatter   allocated array that will contain the k-scatter for each bin
//   pscatter   allocated array that will contain the scatter in power for each bin
//   occupation allocated array that will count the number of grid points contributing to each bin
//   numbin     number of bins (minimum size of all arrays)
//   ktype      flag indicating which definition of momentum to be used
//                  0: grid momentum
//                  1: linear (default)
//   comp1      for component-wise cross spectra, the component for the first field (ignored if negative)
//   comp2      for component-wise cross spectra, the component for the second field (ignored if negative)
//
// Returns:
// 
//////////////////////////

void extractCrossSpectrum(Field<Cplx> & fld1FT, Field<Cplx> & fld2FT, Real * kbin, Real * power, Real * kscatter, Real * pscatter, int * occupation, const int numbins, const bool deconvolve = true, const int ktype = KTYPE_LINEAR, const int comp1 = -1, const int comp2 = -1)
{
	int weight, bin;
	const int linesize = fld1FT.lattice().size(1);
	Real * typek2;
	Real * sinc;
	Real k2max, k2, s;
	rKSite k(fld1FT.lattice());
	Cplx p;
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		typek2 = (Real *) alloca(linesize * sizeof(Real));
		sinc = (Real *) alloca(linesize * sizeof(Real));
	}
	else
	{
		typek2 = (Real *) malloc(linesize * sizeof(Real));
		sinc = (Real *) malloc(linesize * sizeof(Real));
	}
	
	if (ktype == KTYPE_GRID)
	{
#pragma omp parallel for
		for (int i = 0; i < linesize; i++)
		{
			typek2[i] = 2. * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
			typek2[i] *= typek2[i];
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i <= linesize/2; i++)
		{
			typek2[i] = 2. * M_PI * (Real) i;
			typek2[i] *= typek2[i];
		}
#pragma omp parallel for
		for (int i = linesize/2 + 1; i < linesize; i++)
		{
			typek2[i] = 2. * M_PI * (Real) (linesize-i);
			typek2[i] *= typek2[i];
		}
	}
	
	sinc[0] = 1.;
	if (deconvolve)
	{
#pragma omp parallel for
		for (int i = 1; i <= linesize/2; i++)
		{
			sinc[i] = sin(M_PI * (float) i / (float) linesize) * (float) linesize / (M_PI * (float) i);
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 1; i <= linesize/2; i++)
		{
			sinc[i] = 1.;
		}
	}
#pragma omp parallel for
	for (int i = linesize/2 + 1; i < linesize; i++)
	{
		sinc[i] = sinc[linesize-i];
	}
	
	k2max = 3. * typek2[linesize/2];
	
	for (int i = 0; i < numbins; i++)
	{
		kbin[i] = 0.;
		power[i] = 0.;
		kscatter[i] = 0.;
		pscatter[i] = 0.;
		occupation[i] = 0;
	}
	
//	for (k.first(); k.test(); k.next())
//	{
#pragma omp parallel for collapse(2) default(shared) firstprivate(k) private(weight, p, k2, s, bin)
	for (int ii = 0; ii < fld1FT.lattice().sizeLocal(1); ii++)
	{
		for (int jj = 0; jj < fld1FT.lattice().sizeLocal(2); jj++)
		{
			if (!k.setCoord(0, jj + fld1FT.lattice().coordSkip()[0], ii + fld1FT.lattice().coordSkip()[1]))
			{
				throw std::runtime_error("Error in projectFTscalar: Could not set coordinates.");
			}

			for (int z = 0; z < fld1FT.lattice().sizeLocal(0); z++)
			{
				if (k.coord(0) == 0 && k.coord(1) == 0 && k.coord(2) == 0)
				{
					k.next();
					continue;
				}
				else if (k.coord(0) == 0)
					weight = 1;
				else if ((k.coord(0) == linesize/2) && (linesize % 2 == 0))
					weight = 1;
				else
					weight = 2;
					
				k2 = typek2[k.coord(0)] + typek2[k.coord(1)] + typek2[k.coord(2)];
				s = sinc[k.coord(0)] * sinc[k.coord(1)] * sinc[k.coord(2)];
				s *= s;
				
				if (comp1 >= 0 && comp2 >= 0 && comp1 < fld1FT.components() && comp2 < fld2FT.components())
				{
					p = fld1FT(k, comp1) * fld2FT(k, comp2).conj();
				}
				else if (fld1FT.symmetry() == LATfield2::symmetric)
				{
					p = fld1FT(k, 0, 1) * fld2FT(k, 0, 1).conj();
					p += fld1FT(k, 0, 2) * fld2FT(k, 0, 2).conj();
					p += fld1FT(k, 1, 2) * fld2FT(k, 1, 2).conj();
					p *= 2.;
					p += fld1FT(k, 0, 0) * fld2FT(k, 0, 0).conj();
					p += fld1FT(k, 1, 1) * fld2FT(k, 1, 1).conj();
					p += fld1FT(k, 2, 2) * fld2FT(k, 2, 2).conj();
				}
				else
				{
					p = Cplx(0., 0.);
					for (int i = 0; i < fld1FT.components(); i++)
						p += fld1FT(k, i) * fld2FT(k, i).conj();
				}

				bin = (int) floor((double) ((Real) numbins * sqrt(k2 / k2max)));
				if (bin < numbins) 
				{
					#pragma omp critical
					{
						kbin[bin] += weight * sqrt(k2);
						kscatter[bin] += weight * k2;
						power[bin] += weight * p.real() * k2 * sqrt(k2) / s;
						pscatter[bin] += weight * p.real() * p.real() * k2 * k2 * k2 / s / s;
						occupation[bin] += weight;
					}
				}

				k.next();
			}
		}
	}
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(typek2);
		free(sinc);
	}

	if (parallel.isRoot())
	{
#ifdef SINGLE
		MPI_Reduce(MPI_IN_PLACE, (void *) kbin, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) kscatter, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) power, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) pscatter, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
#else
		MPI_Reduce(MPI_IN_PLACE, (void *) kbin, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) kscatter, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) power, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce(MPI_IN_PLACE, (void *) pscatter, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
#endif
		MPI_Reduce(MPI_IN_PLACE, (void *) occupation, numbins, MPI_INT, MPI_SUM, 0, parallel.lat_world_comm());

#pragma omp parallel for
		for (int i = 0; i < numbins; i++)
		{
			if (occupation[i] > 0)
			{
				kscatter[i] = sqrt(kscatter[i] * occupation[i] - kbin[i] * kbin[i]) / occupation[i];
				if (!isfinite(kscatter[i])) kscatter[i] = 0.;
				kbin[i] = kbin[i] / occupation[i];
				power[i] /= occupation[i];
				pscatter[i] = sqrt(pscatter[i] / occupation[i] - power[i] * power[i]);
				if (!isfinite(pscatter[i])) pscatter[i] = 0.;
			}
		}
	}
	else
	{
#ifdef SINGLE
		MPI_Reduce((void *) kbin, NULL, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) kscatter, NULL, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) power, NULL, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) pscatter, NULL, numbins, MPI_FLOAT, MPI_SUM, 0, parallel.lat_world_comm());
#else
		MPI_Reduce((void *) kbin, NULL, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) kscatter, NULL, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) power, NULL, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
		MPI_Reduce((void *) pscatter, NULL, numbins, MPI_DOUBLE, MPI_SUM, 0, parallel.lat_world_comm());
#endif
		MPI_Reduce((void *) occupation, NULL, numbins, MPI_INT, MPI_SUM, 0, parallel.lat_world_comm());
	}
}


//////////////////////////
// extractPowerSpectrum
//////////////////////////
// Description:
//   generates the power spectrum for a Fourier image
// 
// Arguments:
//   fldFT      reference to the Fourier image for which the power spectrum should be extracted
//   kbin       allocated array that will contain the central k-value for the bins
//   power      allocated array that will contain the average power in each bin
//   kscatter   allocated array that will contain the k-scatter for each bin
//   pscatter   allocated array that will contain the scatter in power for each bin
//   occupation allocated array that will count the number of grid points contributing to each bin
//   numbin     number of bins (minimum size of all arrays)
//   ktype      flag indicating which definition of momentum to be used
//                  0: grid momentum
//                  1: linear (default)
//
// Returns:
// 
//////////////////////////

void extractPowerSpectrum(Field<Cplx> & fldFT, Real * kbin, Real * power, Real * kscatter, Real * pscatter, int * occupation, const int numbins, const bool deconvolve = true, const int ktype = KTYPE_LINEAR)
{
	extractCrossSpectrum(fldFT, fldFT, kbin, power, kscatter, pscatter, occupation, numbins, deconvolve, ktype);
}
#endif


//////////////////////////
// writePowerSpectrum
//////////////////////////
// Description:
//   writes power spectra as tabulated data into ASCII file
// 
// Arguments:
//   kbin           array containing the central values of k for each bin
//   power          array containing the central values of P(k) for each bin
//   kscatter       array containing the statistical error on k for each bin
//   pscatter       array containing the statistical error on P(k) for each bin
//   occupation     array containing the number of k-modes contributing to each bin
//   numbins        total number of bins (length of the arrays)
//   rescalek       unit conversion factor for k
//   rescalep       unit conversion factor for P(k)
//   filename       output file name
//   description    descriptive header
//   a              scale factor for this spectrum
//   z_target       target redshift for this output (used only if EXACT_OUTPUT_REDSHIFTS is defined)
//
// Returns:
// 
//////////////////////////

void writePowerSpectrum(Real * kbin, Real * power, Real * kscatter, Real * pscatter, int * occupation, const int numbins, const Real rescalek, const Real rescalep, const char * filename, const char * description, double a, const double z_target = -1)
{
	if (parallel.isRoot())
	{
#ifdef EXACT_OUTPUT_REDSHIFTS
		Real * power2 = (Real *) malloc(numbins * sizeof(Real));

		for (int i = 0; i < numbins; i++)
			power2[i] = power[i]/rescalep;

		if (1. / a < z_target + 1.)
		{
			FILE * infile = fopen(filename, "r");
			double weight = 1.;
			int count = 0;
			if (infile != NULL)
			{
				if (fscanf(infile, "%*[^\n]\n") == EOF || fscanf(infile, "# redshift z=%lf\n", &weight) != 1)
				{
					cout << " error parsing power spectrum file header for interpolation (EXACT_OUTPUT_REDSHIFTS)" << endl;
					weight = 1.;
				}
				else
				{
					weight = (weight - z_target) / (1. + weight - 1./a);
					if (fscanf(infile, "%*[^\n]\n") == EOF)
					{
						cout << " error parsing power spectrum file header for interpolation (EXACT_OUTPUT_REDSHIFTS)" << endl;
					}
					for (int i = 0; i < numbins; i++)
					{
						if (occupation[i] > 0)
						{
#ifdef SINGLE
							if(fscanf(infile, " %*e %e %*e %*e %*d \n", power2+i) != 1)
#else
							if(fscanf(infile, " %*e %le %*e %*e %*d \n", power2+i) != 1)
#endif
							{
								cout << " error parsing power spectrum file data " << i << " for interpolation (EXACT_OUTPUT_REDSHIFTS)" << endl;
								break;
							}
							else count++;
						}
					}
				}
				fclose(infile);

				for (int i = 0; i < numbins; i++)
					power2[i] = (1.-weight)*power2[i] + weight*power[i]/rescalep;

				a = 1. / (z_target + 1.);
			}
		}
#endif // EXACT_OUTPUT_REDSHIFTS
		FILE * outfile = fopen(filename, "w");
		if (outfile == NULL)
		{
			cout << " error opening file for power spectrum output!" << endl;
		}
		else
		{
			fprintf(outfile, "# %s\n", description);
			fprintf(outfile, "# redshift z=%f\n", (1./a)-1.);
			fprintf(outfile, "# k              Pk             sigma(k)       sigma(Pk)      count\n");
			for (int i = 0; i < numbins; i++)
			{
				if (occupation[i] > 0)
#ifdef EXACT_OUTPUT_REDSHIFTS
					fprintf(outfile, "  %e   %e   %e   %e   %d\n", kbin[i]/rescalek, power2[i], kscatter[i]/rescalek, pscatter[i]/rescalep/ sqrt(occupation[i]), occupation[i]);
#else
					fprintf(outfile, "  %e   %e   %e   %e   %d\n", kbin[i]/rescalek, power[i]/rescalep, kscatter[i]/rescalek, pscatter[i]/rescalep/ sqrt(occupation[i]), occupation[i]);
#endif
			}
			fclose(outfile);
		}
#ifdef EXACT_OUTPUT_REDSHIFTS
		free(power2);
#endif
	}
}


//////////////////////////
// computeVectorDiagnostics
//////////////////////////
// Description:
//   computes some diagnostics for the spin-1 perturbation
// 
// Arguments:
//   Bi         reference to the real-space vector field to analyze
//   mdivB      will contain the maximum value of the divergence of Bi
//   mcurlB     will contain the maximum value of the curl of Bi
//
// Returns:
// 
//////////////////////////

__host__ __device__ void computeVectorDiagnostics_kernel(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
{
	outputs[0] = fabs(((*fields[0])(sites[0],0)-(*fields[0])(sites[0]-0,0)) + ((*fields[0])(sites[0],1)-(*fields[0])(sites[0]-1,1)) + ((*fields[0])(sites[0],2)-(*fields[0])(sites[0]-2,2)));
	Real b1 = ((*fields[0])(sites[0],0) + (*fields[0])(sites[0]+0,1) - (*fields[0])(sites[0]+1,0) - (*fields[0])(sites[0],1) + (*fields[0])(sites[0]+2,0) + (*fields[0])(sites[0]+0+2,1) - (*fields[0])(sites[0]+1+2,0) - (*fields[0])(sites[0]+2,1));
	Real b2 = ((*fields[0])(sites[0],0) + (*fields[0])(sites[0]+0,2) - (*fields[0])(sites[0]+2,0) - (*fields[0])(sites[0],2) + (*fields[0])(sites[0]+1,0) + (*fields[0])(sites[0]+0+1,2) - (*fields[0])(sites[0]+2+1,0) - (*fields[0])(sites[0]+1,2));
	Real b3 = ((*fields[0])(sites[0],2) + (*fields[0])(sites[0]+2,1) - (*fields[0])(sites[0]+1,2) - (*fields[0])(sites[0],1) + (*fields[0])(sites[0]+0,2) + (*fields[0])(sites[0]+2+0,1) - (*fields[0])(sites[0]+1+0,2) - (*fields[0])(sites[0]+0,1));
	outputs[1] = 0.5 * sqrt(b1 * b1 + b2 * b2 + b3 * b3);
}

struct computeVectorDiagnostics_functor
{
	__host__ __device__ void operator()(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
	{
		computeVectorDiagnostics_kernel(fields, sites, nfields, params, outputs);
	}
};

void computeVectorDiagnostics(Field<Real> & Bi, Real & mdivB, Real & mcurlB)
{
	//Real b1, b2, b3, b4;
	const Real linesize = (Real) Bi.lattice().sizeLocal(0);
	//Site x(Bi.lattice());
	
	//mdivB = 0.;
	//mcurlB = 0.;

	Field<Real> * fieldptr = &Bi;
	double result[2] = { 0., 0. };
	int reduce[2] = { MAX, MAX };

	int numpts = Bi.lattice().sizeLocal(0);
	int block_x = Bi.lattice().sizeLocal(1);
	int block_y = Bi.lattice().sizeLocal(2);

	lattice_for_each<computeVectorDiagnostics_functor, 2><<<dim3(block_x, block_y), 128>>>(computeVectorDiagnostics_functor(), numpts, &fieldptr, 1, nullptr, result, reduce);

	cudaDeviceSynchronize();

	parallel.max<double>(result, 2);

	mdivB = result[0] * linesize;
	mcurlB = result[1] * linesize;

	/*for (x.first(); x.test(); x.next())
	{
		b1 = fabs((Bi(x,0)-Bi(x-0,0)) + (Bi(x,1)-Bi(x-1,1)) + (Bi(x,2)-Bi(x-2,2))) * linesize;
		if (b1 > mdivB) mdivB = b1;
		b1 = 0.5 * (Bi(x,0) + Bi(x+0,1) - Bi(x+1,0) - Bi(x,1) + Bi(x+2,0) + Bi(x+0+2,1) - Bi(x+1+2,0) - Bi(x+2,1)) * linesize;
		b2 = 0.5 * (Bi(x,0) + Bi(x+0,2) - Bi(x+2,0) - Bi(x,2) + Bi(x+1,0) + Bi(x+0+1,2) - Bi(x+2+1,0) - Bi(x+1,2)) * linesize;
		b3 = 0.5 * (Bi(x,2) + Bi(x+2,1) - Bi(x+1,2) - Bi(x,1) + Bi(x+0,2) + Bi(x+2+0,1) - Bi(x+1+0,2) - Bi(x+0,1)) * linesize;
		b4 = sqrt(b1 * b1 + b2 * b2 + b3 * b3);
		if (b4 > mcurlB) mcurlB = b4;
	}
	
	parallel.max<Real>(mdivB);
	parallel.max<Real>(mcurlB);*/
}


//////////////////////////
// computeTensorDiagnostics
//////////////////////////
// Description:
//   computes some diagnostics for the spin-2 perturbation
// 
// Arguments:
//   hij        reference to the real-space tensor field to analyze
//   mdivh      will contain the maximum value of the divergence of hij
//   mtraceh    will contain the maximum value of the trace of hij
//   mnormh     will contain the maximum value of the norm of hij
//
// Returns:
// 
//////////////////////////

__host__ __device__ void computeTensorDiagnostics_kernel(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
{
	Real d1 = ((*fields[0])(sites[0]+0, 0, 0) - (*fields[0])(sites[0], 0, 0) + (*fields[0])(sites[0], 0, 1) - (*fields[0])(sites[0]-1, 0, 1) + (*fields[0])(sites[0], 0, 2) - (*fields[0])(sites[0]-2, 0, 2));
	Real d2 = ((*fields[0])(sites[0]+1, 1, 1) - (*fields[0])(sites[0], 1, 1) + (*fields[0])(sites[0], 0, 1) - (*fields[0])(sites[0]-0, 0, 1) + (*fields[0])(sites[0], 1, 2) - (*fields[0])(sites[0]-2, 1, 2));
	Real d3 = ((*fields[0])(sites[0]+2, 2, 2) - (*fields[0])(sites[0], 2, 2) + (*fields[0])(sites[0], 0, 2) - (*fields[0])(sites[0]-0, 0, 2) + (*fields[0])(sites[0], 1, 2) - (*fields[0])(sites[0]-1, 1, 2));
	outputs[0] = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
	outputs[1] = fabs((*fields[0])(sites[0], 0, 0) + (*fields[0])(sites[0], 1, 1) + (*fields[0])(sites[0], 2, 2));
	outputs[2] = sqrt((*fields[0])(sites[0], 0, 0) * (*fields[0])(sites[0], 0, 0) + Real(2) * (*fields[0])(sites[0], 0, 1) * (*fields[0])(sites[0], 0, 1) + Real(2) * (*fields[0])(sites[0], 0, 2)* (*fields[0])(sites[0], 0, 2) + (*fields[0])(sites[0], 1, 1) * (*fields[0])(sites[0], 1, 1) + Real(2) * (*fields[0])(sites[0], 1, 2) * (*fields[0])(sites[0], 1, 2) + (*fields[0])(sites[0], 2, 2) * (*fields[0])(sites[0], 2, 2));
}

struct computeTensorDiagnostics_functor
{
	__host__ __device__ void operator()(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
	{
		computeTensorDiagnostics_kernel(fields, sites, nfields, params, outputs);
	}
};

void computeTensorDiagnostics(Field<Real> & hij, Real & mdivh, Real & mtraceh, Real & mnormh)
{
	//Real d1, d2, d3;
	const Real linesize = (Real) hij.lattice().sizeLocal(0);
	//Site x(hij.lattice());
	
	//mdivh = 0.;
	//mtraceh = 0.;
	//mnormh = 0.;

	Field<Real> * fieldptr = &hij;
	double result[3] = { 0., 0., 0. };
	int reduce[3] = { MAX, MAX, MAX };

	int numpts = hij.lattice().sizeLocal(0);
	int block_x = hij.lattice().sizeLocal(1);
	int block_y = hij.lattice().sizeLocal(2);

	lattice_for_each<computeTensorDiagnostics_functor, 3><<<dim3(block_x, block_y), 128>>>(computeTensorDiagnostics_functor(), numpts, &fieldptr, 1, nullptr, result, reduce);

	cudaDeviceSynchronize();

	parallel.max<double>(result, 3);

	mdivh = result[0] * linesize;
	mtraceh = result[1];
	mnormh = result[2];

	/*for (x.first(); x.test(); x.next())
	{
		d1 = (hij(x+0, 0, 0) - hij(x, 0, 0) + hij(x, 0, 1) - hij(x-1, 0, 1) + hij(x, 0, 2) - hij(x-2, 0, 2)) * linesize;
		d2 = (hij(x+1, 1, 1) - hij(x, 1, 1) + hij(x, 0, 1) - hij(x-0, 0, 1) + hij(x, 1, 2) - hij(x-2, 1, 2)) * linesize;
		d3 = (hij(x+2, 2, 2) - hij(x, 2, 2) + hij(x, 0, 2) - hij(x-0, 0, 2) + hij(x, 1, 2) - hij(x-1, 1, 2)) * linesize;
		d1 = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
		if (d1 > mdivh) mdivh = d1;
		d1 = fabs(hij(x, 0, 0) + hij(x, 1, 1) + hij(x, 2, 2));
		if (d1 > mtraceh) mtraceh = d1;
		d1 = sqrt(hij(x, 0, 0) * hij(x, 0, 0) + 2. * hij(x, 0, 1) * hij(x, 0, 1) + 2. * hij(x, 0, 2)* hij(x, 0, 2) + hij(x, 1, 1) * hij(x, 1, 1) + 2. * hij(x, 1, 2) * hij(x, 1, 2) + hij(x, 2, 2) * hij(x, 2, 2));
		if (d1 > mnormh) mnormh = d1;
	}
	
	parallel.max<Real>(mdivh);
	parallel.max<Real>(mtraceh);
	parallel.max<Real>(mnormh);*/
}


//////////////////////////
// findIntersectingLightcones
//////////////////////////
// Description:
//   determines periodic copies of light cone vertex for which the present
//   look-back interval may overlap with a given spatial domain
// 
// Arguments:
//   lightcone  reference to structure describing light cone geometry
//   outer      outer (far) limit of look-back interval
//   inner      inner (close) limit of look-back interval
//   domain     array of domain boundaries
//   vertex     will contain array of relevant vertex locations
//
// Returns:
//   number of vertices found
// 
//////////////////////////

int findIntersectingLightcones(lightcone_geometry & lightcone, double outer, double inner, double * domain, double vertex[MAX_INTERSECTS][3])
{
	int range = (int) ceil(outer) + 1;
	int u, v, w, n = 0;
	double corner[8][3];
	double rdom, dist;

	corner[0][0] = domain[0];
	corner[0][1] = domain[1];
	corner[0][2] = domain[2];

	corner[1][0] = domain[3];
	corner[1][1] = domain[1];
	corner[1][2] = domain[2];

	corner[2][0] = domain[0];
	corner[2][1] = domain[4];
	corner[2][2] = domain[2];

	corner[3][0] = domain[3];
	corner[3][1] = domain[4];
	corner[3][2] = domain[2];

	corner[4][0] = domain[0];
	corner[4][1] = domain[1];
	corner[4][2] = domain[5];

	corner[5][0] = domain[3];
	corner[5][1] = domain[1];
	corner[5][2] = domain[5];

	corner[6][0] = domain[0];
	corner[6][1] = domain[4];
	corner[6][2] = domain[5];

	corner[7][0] = domain[3];
	corner[7][1] = domain[4];
	corner[7][2] = domain[5];

#pragma omp parallel for collapse(3) default(shared) private(rdom, dist)
	for (u = -range; u <= range; u++)
	{
		for (v = -range; v <= range; v++)
		{
			for (w = -range; w <= range; w++)
			{
				double vertex_[3];
				vertex_[0] = lightcone.vertex[0] + u;
				vertex_[1] = lightcone.vertex[1] + v;
				vertex_[2] = lightcone.vertex[2] + w;

				// first, check if domain lies outside outer sphere
				if (vertex_[0] < domain[0])
				{
					if (vertex_[1] < domain[1])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-corner[0][0])*(vertex_[0]-corner[0][0]) + (vertex_[1]-corner[0][1])*(vertex_[1]-corner[0][1]) + (vertex_[2]-corner[0][2])*(vertex_[2]-corner[0][2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-corner[4][0])*(vertex_[0]-corner[4][0]) + (vertex_[1]-corner[4][1])*(vertex_[1]-corner[4][1]) + (vertex_[2]-corner[4][2])*(vertex_[2]-corner[4][2])) > outer) continue;
						}
						else if (sqrt((vertex_[0]-domain[0])*(vertex_[0]-domain[0]) + (vertex_[1]-domain[1])*(vertex_[1]-domain[1])) > outer) continue;
					}
					else if (vertex_[1] > domain[4])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-corner[2][0])*(vertex_[0]-corner[2][0]) + (vertex_[1]-corner[2][1])*(vertex_[1]-corner[2][1]) + (vertex_[2]-corner[2][2])*(vertex_[2]-corner[2][2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-corner[6][0])*(vertex_[0]-corner[6][0]) + (vertex_[1]-corner[6][1])*(vertex_[1]-corner[6][1]) + (vertex_[2]-corner[6][2])*(vertex_[2]-corner[6][2])) > outer) continue;
						}
						else if (sqrt((vertex_[0]-domain[0])*(vertex_[0]-domain[0]) + (vertex_[1]-domain[4])*(vertex_[1]-domain[4])) > outer) continue;
					}
					else
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-domain[0])*(vertex_[0]-domain[0]) + (vertex_[2]-domain[2])*(vertex_[2]-domain[2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-domain[0])*(vertex_[0]-domain[0]) + (vertex_[2]-domain[5])*(vertex_[2]-domain[5])) > outer) continue;
						}
						else if (domain[0]-vertex_[0] > outer) continue;
					}
				}
				else if (vertex_[0] > domain[3])
				{
					if (vertex_[1] < domain[1])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-corner[1][0])*(vertex_[0]-corner[1][0]) + (vertex_[1]-corner[1][1])*(vertex_[1]-corner[1][1]) + (vertex_[2]-corner[1][2])*(vertex_[2]-corner[1][2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-corner[5][0])*(vertex_[0]-corner[5][0]) + (vertex_[1]-corner[5][1])*(vertex_[1]-corner[5][1]) + (vertex_[2]-corner[5][2])*(vertex_[2]-corner[5][2])) > outer) continue;
						}
						else if (sqrt((vertex_[0]-domain[3])*(vertex_[0]-domain[3]) + (vertex_[1]-domain[1])*(vertex_[1]-domain[1])) > outer) continue;
					}
					else if (vertex_[1] > domain[4])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-corner[3][0])*(vertex_[0]-corner[3][0]) + (vertex_[1]-corner[3][1])*(vertex_[1]-corner[3][1]) + (vertex_[2]-corner[3][2])*(vertex_[2]-corner[3][2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-corner[7][0])*(vertex_[0]-corner[7][0]) + (vertex_[1]-corner[7][1])*(vertex_[1]-corner[7][1]) + (vertex_[2]-corner[7][2])*(vertex_[2]-corner[7][2])) > outer) continue;
						}
						else if (sqrt((vertex_[0]-domain[3])*(vertex_[0]-domain[3]) + (vertex_[1]-domain[4])*(vertex_[1]-domain[4])) > outer) continue;
					}
					else
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[0]-domain[3])*(vertex_[0]-domain[3]) + (vertex_[2]-domain[2])*(vertex_[2]-domain[2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[0]-domain[3])*(vertex_[0]-domain[3]) + (vertex_[2]-domain[5])*(vertex_[2]-domain[5])) > outer) continue;
						}
						else if (vertex_[0]-domain[3] > outer) continue;
					}
				}
				else
				{
					if (vertex_[1] < domain[1])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[1]-domain[1])*(vertex_[1]-domain[1]) + (vertex_[2]-domain[2])*(vertex_[2]-domain[2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[1]-domain[1])*(vertex_[1]-domain[1]) + (vertex_[2]-domain[5])*(vertex_[2]-domain[5])) > outer) continue;
						}
						else if (domain[1]-vertex_[1] > outer) continue;
					}
					else if (vertex_[1] > domain[4])
					{
						if (vertex_[2] < domain[2])
						{
							if (sqrt((vertex_[1]-domain[4])*(vertex_[1]-domain[4]) + (vertex_[2]-domain[2])*(vertex_[2]-domain[2])) > outer) continue;
						}
						else if (vertex_[2] > domain[5])
						{
							if (sqrt((vertex_[1]-domain[4])*(vertex_[1]-domain[4]) + (vertex_[2]-domain[5])*(vertex_[2]-domain[5])) > outer) continue;
						}
						else if (vertex_[1]-domain[4] > outer) continue;
					}
					else if (vertex_[2]-domain[5] > outer || domain[2]-vertex_[2] > outer) continue;
				}
				
				if (sqrt((corner[0][0]-vertex_[0])*(corner[0][0]-vertex_[0]) + (corner[0][1]-vertex_[1])*(corner[0][1]-vertex_[1]) + (corner[0][2]-vertex_[2])*(corner[0][2]-vertex_[2])) < inner && sqrt((corner[1][0]-vertex_[0])*(corner[1][0]-vertex_[0]) + (corner[1][1]-vertex_[1])*(corner[1][1]-vertex_[1]) + (corner[1][2]-vertex_[2])*(corner[1][2]-vertex_[2])) < inner && sqrt((corner[2][0]-vertex_[0])*(corner[2][0]-vertex_[0]) + (corner[2][1]-vertex_[1])*(corner[2][1]-vertex_[1]) + (corner[2][2]-vertex_[2])*(corner[2][2]-vertex_[2])) < inner && sqrt((corner[3][0]-vertex_[0])*(corner[3][0]-vertex_[0]) + (corner[3][1]-vertex_[1])*(corner[3][1]-vertex_[1]) + (corner[3][2]-vertex_[2])*(corner[3][2]-vertex_[2])) < inner && sqrt((corner[4][0]-vertex_[0])*(corner[4][0]-vertex_[0]) + (corner[4][1]-vertex_[1])*(corner[4][1]-vertex_[1]) + (corner[4][2]-vertex_[2])*(corner[4][2]-vertex_[2])) < inner && sqrt((corner[5][0]-vertex_[0])*(corner[5][0]-vertex_[0]) + (corner[5][1]-vertex_[1])*(corner[5][1]-vertex_[1]) + (corner[5][2]-vertex_[2])*(corner[5][2]-vertex_[2])) < inner && sqrt((corner[6][0]-vertex_[0])*(corner[6][0]-vertex_[0]) + (corner[6][1]-vertex_[1])*(corner[6][1]-vertex_[1]) + (corner[6][2]-vertex_[2])*(corner[6][2]-vertex_[2])) < inner && sqrt((corner[7][0]-vertex_[0])*(corner[7][0]-vertex_[0]) + (corner[7][1]-vertex_[1])*(corner[7][1]-vertex_[1]) + (corner[7][2]-vertex_[2])*(corner[7][2]-vertex_[2])) < inner) continue; // domain lies within inner sphere

				rdom = 0.5 * sqrt((domain[3]-domain[0])*(domain[3]-domain[0]) + (domain[4]-domain[1])*(domain[4]-domain[1]) + (domain[5]-domain[2])*(domain[5]-domain[2]));
				dist = sqrt((0.5*domain[0]+0.5*domain[3]-vertex_[0])*(0.5*domain[0]+0.5*domain[3]-vertex_[0]) + (0.5*domain[1]+0.5*domain[4]-vertex_[1])*(0.5*domain[1]+0.5*domain[4]-vertex_[1]) + (0.5*domain[2]+0.5*domain[5]-vertex_[2])*(0.5*domain[2]+0.5*domain[5]-vertex_[2]));

				if (dist <= rdom) // vertex lies within domain enclosing sphere
				{
					#pragma omp critical(recordvertex)
					{
						if (n < MAX_INTERSECTS)
						{
							vertex[n][0] = vertex_[0];
							vertex[n][1] = vertex_[1];
							vertex[n][2] = vertex_[2];
						}
						n++;
					}
					continue;
				}

				if (((0.5*domain[0]+0.5*domain[3]-vertex_[0])*lightcone.direction[0] + (0.5*domain[1]+0.5*domain[4]-vertex_[1])*lightcone.direction[1] + (0.5*domain[2]+0.5*domain[5]-vertex_[2])*lightcone.direction[2]) / dist >= lightcone.opening) // center of domain lies within opening
				{
					#pragma omp critical(recordvertex)
					{
						if (n < MAX_INTERSECTS)
						{
							vertex[n][0] = vertex_[0];
							vertex[n][1] = vertex_[1];
							vertex[n][2] = vertex_[2];
						}
						n++;
					}
					continue;
				} 

				if (dist > outer && acos(((0.5*domain[0]+0.5*domain[3]-vertex_[0])*lightcone.direction[0] + (0.5*domain[1]+0.5*domain[4]-vertex_[1])*lightcone.direction[1] + (0.5*domain[2]+0.5*domain[5]-vertex_[2])*lightcone.direction[2]) / dist) - acos(lightcone.opening) <= acos((outer*outer + dist*dist - rdom*rdom) / (2. * outer * dist))) // enclosing sphere within opening
				{
					#pragma omp critical(recordvertex)
					{
						if (n < MAX_INTERSECTS)
						{
							vertex[n][0] = vertex_[0];
							vertex[n][1] = vertex_[1];
							vertex[n][2] = vertex_[2];
						}
						n++;
					}
					continue;
				}
				
				if (dist <= outer && acos(((0.5*domain[0]+0.5*domain[3]-vertex_[0])*lightcone.direction[0] + (0.5*domain[1]+0.5*domain[4]-vertex_[1])*lightcone.direction[1] + (0.5*domain[2]+0.5*domain[5]-vertex_[2])*lightcone.direction[2]) / dist) - acos(lightcone.opening) <= asin(rdom / dist)) // enclosing sphere within opening
				{
					#pragma omp critical(recordvertex)
					{
						if (n < MAX_INTERSECTS)
						{
							vertex[n][0] = vertex_[0];
							vertex[n][1] = vertex_[1];
							vertex[n][2] = vertex_[2];
						}
						n++;
					}
				}
			}
		}
	}

	if (n >= MAX_INTERSECTS)
	{
		cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": maximum number of lightcone intersects exceeds MAX_INTERSECTS = " << MAX_INTERSECTS << " for domain (" << domain[0] << ", " << domain[1] << ", " << domain[2] << ") - (" << domain[3] << ", " << domain[4] << ", " << domain[5] << "); some data may be missing in output!" << endl;
		return MAX_INTERSECTS;
	}

	return n;
}


//////////////////////////
// hourMinSec
//////////////////////////
// Description:
//   generates formatted output for cpu-time: hh..h:mm:ss.s
// 
// Arguments:
//   seconds    number of seconds
//
// Returns:
//   formatted string
// 
//////////////////////////

string hourMinSec(double seconds)
{
	string output;
	char ptr[20];
	int h, m, s, f;

	h = (int) floor(seconds / 3600.);
	seconds -= 3600. * h;
	m = (int) floor(seconds / 60.);
	seconds -= 60. * m;
	s = (int) floor(seconds);
	seconds -= s;
	f = (int) floor(10. * seconds);
	sprintf(ptr, "%d:%02d:%02d.%d", h, m, s, f);

	output.reserve(20);
	output.assign(ptr);

	return output;
}

__host__ __device__ void lattice_add(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
{
	(*fields[0])(sites[0]) += (*params);
}

struct lattice_add_functor
{
	__host__ __device__ void operator()(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
	{
		lattice_add(fields, sites, nfields, params, outputs);
	}
};


#ifdef HAVE_HEALPIX

__device__ int64_t compress_bits64_helper (int64_t v)
{
	const short ctab[]={
		#define Z(a) a,a+1,a+256,a+257
		#define Y(a) Z(a),Z(a+2),Z(a+512),Z(a+514)
		#define X(a) Y(a),Y(a+4),Y(a+1024),Y(a+1028)
		X(0),X(8),X(2048),X(2056)
		#undef X
		#undef Y
		#undef Z
	};
	int64_t raw = v&0x5555555555555555ull;
	raw|=raw>>15;
	return ctab[ raw     &0xff]      | (ctab[(raw>> 8)&0xff]<< 4)
		| (ctab[(raw>>32)&0xff]<<16) | (ctab[(raw>>40)&0xff]<<20);
}

__device__ void nest2xyf64_helper (int64_t nside, int64_t pix, int *ix, int *iy, int *face_num)
{
	int64_t npface_=nside*nside;
	*face_num = pix/npface_;
	pix &= (npface_-1);
	*ix = compress_bits64_helper(pix);
	*iy = compress_bits64_helper(pix>>1);
}


//////////////////////////
// pix2vec_nest64_gpu
//////////////////////////
// Description:
//   converts pixel index to vector on unit sphere (device code)
//
// Arguments:
//   nside      HEALPix resolution parameter
//   ipix       pixel index
//   vec        will contain the vector
//
// Returns:
//
//////////////////////////

__device__ void pix2vec_nest64_gpu(int64_t nside, int64_t pix, double *vec)
{
	double z, phi, s = -5;
	int64_t nl4 = nside*4;
	int64_t npix_=12*nside*nside;
	double fact2_ = 4./npix_;
	int face_num, ix, iy;
	int64_t jr, nr, kshift, jp;
	const int jrll_[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
	const int jpll_[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };

	nest2xyf64_helper(nside,pix,&ix,&iy,&face_num);
	jr = (jrll_[face_num]*nside) - ix - iy - 1;

	if (jr<nside)
	{
		double tmp;
		nr = jr;
		tmp=(nr*nr)*fact2_;
		z = 1 - tmp;
		if (z>0.99) s=sqrt(tmp*(2.-tmp));
		kshift = 0;
	}
	else if (jr > 3*nside)
	{
		double tmp;
		nr = nl4-jr;
		tmp=(nr*nr)*fact2_;
		z = tmp - 1;
		if (z<-0.99) s=sqrt(tmp*(2.-tmp));
		kshift = 0;
	}
	else
	{
		double fact1_ = (nside<<1)*fact2_;
		nr = nside;
		z = (2*nside-jr)*fact1_;
		kshift = (jr-nside)&1;
	}

	jp = (jpll_[face_num]*nr + ix -iy + 1 + kshift) / 2;
	if (jp>nl4) jp-=nl4;
	if (jp<1) jp+=nl4;

	phi = (jp-(kshift+1)*0.5)*(1.570796326794896619231321691639751442099/nr);

	if (s<-2) s=sqrt((1.-z)*(1.+z));

	vec[0]=s*cos(phi);
	vec[1]=s*sin(phi);
	vec[2]=z;
}

#endif

#endif
