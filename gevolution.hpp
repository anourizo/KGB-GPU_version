//////////////////////////
// gevolution.hpp
//////////////////////////
// 
// Geneva algorithms for evolution of metric perturbations
// and relativistic free-streaming particles (gevolution)
//
// 1. Suite of Fourier-based methods for the computation of the
//    relativistic scalar (Phi, Phi-Psi), vector, and tensor modes [see J. Adamek,
//    R. Durrer, and M. Kunz, Class. Quant. Grav. 31, 234006 (2014); J. Adamek,
//    R. Durrer, and M. Kunz, JCAP 1607, 053 (2016); Ø. Christiansen, J. Adamek,
//    F. Hassani, and D. F. Mota, arXiv:2401.02409]
//
// 2. Collection of "update position" and "update velocity/momentum" methods
//    [see J. Adamek, D. Daverio, R. Durrer, and M. Kunz, JCAP 1607, 053 (2016)]
//
// 3. Collection of projection methods for the construction of the
//    stress-energy-tensor
//
// 4. Fourier-space projection methods for the computation of the
//    curl and divergence of the velocity field
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: February 2025
//
//////////////////////////

#ifndef GEVOLUTION_HEADER
#define GEVOLUTION_HEADER

#include "lattice_loop.hpp"

#ifndef Cplx
#define Cplx Imag
#endif

using namespace std;
using namespace LATfield2;


//////////////////////////
// prepareFTsource (1)
//////////////////////////
// Description:
//   construction of real-space source tensor for Fourier-based solvers
// 
// Arguments:
//   phi        reference to field configuration
//   Tij        reference to symmetric tensor field containing the space-space
//              components of the stress-energy tensor (rescaled by a^3)
//   Sij        reference to allocated symmetric tensor field which will contain
//              the source tensor (may be identical to Tji)
//   coeff      scaling coefficient for Tij ("8 pi G dx^2 / a")
//
// Returns:
// 
//////////////////////////

__host__ __device__ void prepareFTsource_Tij(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
{
	// 0-0-component:
	(*fields[0])(sites[0], 0, 0) = 
#ifdef PHINONLINEAR
	Real(0.5) * ((*fields[2])(sites[2]+0) - (*fields[2])(sites[2]-0)) * ((*fields[2])(sites[2]+0) - (*fields[2])(sites[2]-0)) +
#endif
	(*params) * (*fields[1])(sites[1], 0, 0);

	// 1-1-component:
	(*fields[0])(sites[0], 1, 1) =
#ifdef PHINONLINEAR
	Real(0.5) * ((*fields[2])(sites[2]+1) - (*fields[2])(sites[2]-1)) * ((*fields[2])(sites[2]+1) - (*fields[2])(sites[2]-1)) +
#endif
	(*params) * (*fields[1])(sites[1], 1, 1);

	// 2-2-component:
	(*fields[0])(sites[0], 2, 2) =
#ifdef PHINONLINEAR
	Real(0.5) * ((*fields[2])(sites[2]+2) - (*fields[2])(sites[2]-2)) * ((*fields[2])(sites[2]+2) - (*fields[2])(sites[2]-2)) +
#endif
	(*params) * (*fields[1])(sites[1], 2, 2);

	// 0-1-component:
	(*fields[0])(sites[0], 0, 1) =
#ifdef PHINONLINEAR
	(*fields[2])(sites[2]+0) * (*fields[2])(sites[2]+1) - (*fields[2])(sites[2]) * (*fields[2])(sites[2]+0+1) + Real(0.5) * ((*fields[2])(sites[2]) * (*fields[2])(sites[2]) - (*fields[2])(sites[2]+0) * (*fields[2])(sites[2]+0) - (*fields[2])(sites[2]+1) * (*fields[2])(sites[2]+1) + (*fields[2])(sites[2]+0+1) * (*fields[2])(sites[2]+0+1)) +
#endif
	(*params) * (*fields[1])(sites[1], 0, 1);

	// 0-2-component:
	(*fields[0])(sites[0], 0, 2) =
#ifdef PHINONLINEAR
	(*fields[2])(sites[2]+0) * (*fields[2])(sites[2]+2) - (*fields[2])(sites[2]) * (*fields[2])(sites[2]+0+2) + Real(0.5) * ((*fields[2])(sites[2]) * (*fields[2])(sites[2]) - (*fields[2])(sites[2]+0) * (*fields[2])(sites[2]+0) - (*fields[2])(sites[2]+2) * (*fields[2])(sites[2]+2) + (*fields[2])(sites[2]+0+2) * (*fields[2])(sites[2]+0+2)) +
#endif
	(*params) * (*fields[1])(sites[1], 0, 2);

	// 1-2-component:
	(*fields[0])(sites[0], 1, 2) =
#ifdef PHINONLINEAR
	(*fields[2])(sites[2]+1) * (*fields[2])(sites[2]+2) - (*fields[2])(sites[2]) * (*fields[2])(sites[2]+1+2) + Real(0.5) * ((*fields[2])(sites[2]) * (*fields[2])(sites[2]) - (*fields[2])(sites[2]+1) * (*fields[2])(sites[2]+1) - (*fields[2])(sites[2]+2) * (*fields[2])(sites[2]+2) + (*fields[2])(sites[2]+1+2) * (*fields[2])(sites[2]+1+2)) +
#endif
	(*params) * (*fields[1])(sites[1], 1, 2);
}

// callable struct for prepareFTsource_Tij

struct prepareFTsource_Tij_functor
{
	__host__ __device__ void operator()(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
	{
		prepareFTsource_Tij(fields, sites, nfields, params, outputs);
	}
};

void prepareFTsource(Field<Real> & phi, Field<Real> & Tij, Field<Real> & Sij, const double coeff)
{
	Field<Real> * fields[3] = {&Sij, &Tij, &phi};
	double params = coeff;

	int numpts = phi.lattice().sizeLocal(0);
	int block_x = phi.lattice().sizeLocal(1);
	int block_y = phi.lattice().sizeLocal(2);

	lattice_for_each<<<dim3(block_x, block_y), 128>>>(prepareFTsource_Tij_functor(), numpts, fields, 3, &params, nullptr, nullptr);

	cudaDeviceSynchronize();

/*	Site x(phi.lattice());
	
	for (x.first(); x.test(); x.next())
	{
		// 0-0-component:
		Sij(x, 0, 0) = coeff * Tij(x, 0, 0);
#ifdef PHINONLINEAR
#ifdef ORIGINALMETRIC
		Sij(x, 0, 0) -= 4. * phi(x) * (phi(x-0) + phi(x+0) - 2. * phi(x));
		Sij(x, 0, 0) -= 0.5 * (phi(x+0) - phi(x-0)) * (phi(x+0) - phi(x-0));
#else
		Sij(x, 0, 0) += 0.5 * (phi(x+0) - phi(x-0)) * (phi(x+0) - phi(x-0));
#endif
#endif

		// 1-1-component:
		Sij(x, 1, 1) = coeff * Tij(x, 1, 1);
#ifdef PHINONLINEAR
#ifdef ORIGINALMETRIC
		Sij(x, 1, 1) -= 4. * phi(x) * (phi(x-1) + phi(x+1) - 2. * phi(x));
		Sij(x, 1, 1) -= 0.5 * (phi(x+1) - phi(x-1)) * (phi(x+1) - phi(x-1));
#else
		Sij(x, 1, 1) += 0.5 * (phi(x+1) - phi(x-1)) * (phi(x+1) - phi(x-1));
#endif
#endif

		// 2-2-component:
		Sij(x, 2, 2) = coeff * Tij(x, 2, 2);
#ifdef PHINONLINEAR
#ifdef ORIGINALMETRIC
		Sij(x, 2, 2) -= 4. * phi(x) * (phi(x-2) + phi(x+2) - 2. * phi(x));
		Sij(x, 2, 2) -= 0.5 * (phi(x+2) - phi(x-2)) * (phi(x+2) - phi(x-2));
#else
		Sij(x, 2, 2) += 0.5 * (phi(x+2) - phi(x-2)) * (phi(x+2) - phi(x-2));
#endif
#endif

		// 0-1-component:
		Sij(x, 0, 1) = coeff * Tij(x, 0, 1);
#ifdef PHINONLINEAR
		Sij(x, 0, 1) += phi(x+0) * phi(x+1) - phi(x) * phi(x+0+1);
#ifdef ORIGINALMETRIC
		Sij(x, 0, 1) -= 1.5 * phi(x) * phi(x);
		Sij(x, 0, 1) += 1.5 * phi(x+0) * phi(x+0);
		Sij(x, 0, 1) += 1.5 * phi(x+1) * phi(x+1);
		Sij(x, 0, 1) -= 1.5 * phi(x+0+1) * phi(x+0+1);
#else
		Sij(x, 0, 1) += 0.5 * phi(x) * phi(x);
		Sij(x, 0, 1) -= 0.5 * phi(x+0) * phi(x+0);
		Sij(x, 0, 1) -= 0.5 * phi(x+1) * phi(x+1);
		Sij(x, 0, 1) += 0.5 * phi(x+0+1) * phi(x+0+1);
#endif
#endif

		// 0-2-component:
		Sij(x, 0, 2) = coeff * Tij(x, 0, 2);
#ifdef PHINONLINEAR
		Sij(x, 0, 2) += phi(x+0) * phi(x+2) - phi(x) * phi(x+0+2);
#ifdef ORIGINALMETRIC
		Sij(x, 0, 2) -= 1.5 * phi(x) * phi(x);
		Sij(x, 0, 2) += 1.5 * phi(x+0) * phi(x+0);
		Sij(x, 0, 2) += 1.5 * phi(x+2) * phi(x+2);
		Sij(x, 0, 2) -= 1.5 * phi(x+0+2) * phi(x+0+2);
#else
		Sij(x, 0, 2) += 0.5 * phi(x) * phi(x);
		Sij(x, 0, 2) -= 0.5 * phi(x+0) * phi(x+0);
		Sij(x, 0, 2) -= 0.5 * phi(x+2) * phi(x+2);
		Sij(x, 0, 2) += 0.5 * phi(x+0+2) * phi(x+0+2);
#endif
#endif

		// 1-2-component:
		Sij(x, 1, 2) = coeff * Tij(x, 1, 2);
#ifdef PHINONLINEAR
		Sij(x, 1, 2) += phi(x+1) * phi(x+2) - phi(x) * phi(x+1+2);
#ifdef ORIGINALMETRIC
		Sij(x, 1, 2) -= 1.5 * phi(x) * phi(x);
		Sij(x, 1, 2) += 1.5 * phi(x+1) * phi(x+1);
		Sij(x, 1, 2) += 1.5 * phi(x+2) * phi(x+2);
		Sij(x, 1, 2) -= 1.5 * phi(x+1+2) * phi(x+1+2);
#else
		Sij(x, 1, 2) += 0.5 * phi(x) * phi(x);
		Sij(x, 1, 2) -= 0.5 * phi(x+1) * phi(x+1);
		Sij(x, 1, 2) -= 0.5 * phi(x+2) * phi(x+2);
		Sij(x, 1, 2) += 0.5 * phi(x+1+2) * phi(x+1+2);
#endif
#endif
	}*/
}

//////////////////////////
// prepareFTsource (2)
//////////////////////////
// Description:
//   construction of real-space source field for Fourier-based solvers
// 
// Arguments:
//   phi        reference to field configuration (first Bardeen potential)
//   chi        reference to field configuration (difference between Bardeen potentials, phi-psi)
//   source     reference to fully dressed source field (rescaled by a^3)
//   bgmodel    background model of the source (rescaled by a^3) to be subtracted
//   result     reference to allocated field which will contain the result (may be identical to source)
//   coeff      diffusion coefficient ("3 H_conformal dx^2 / dtau")
//   coeff2     scaling coefficient for the source ("4 pi G dx^2 / a")
//   coeff3     scaling coefficient for the psi-term ("3 H_conformal^2 dx^2")
//
// Returns:
//   sum of the source field
//////////////////////////

__host__ __device__ void prepareFTsource_T00(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
{
	outputs[0] = (*fields[1])(sites[1]);
	(*fields[0])(sites[0]) = params[2] * ((*fields[1])(sites[1]) - params[0]);
#ifdef PHINONLINEAR
	(*fields[0])(sites[0]) *= Real(1) - Real(2) * (*fields[2])(sites[2]) * (Real(1) - (*fields[2])(sites[2]));
	(*fields[0])(sites[0]) += Real(0.125) * ((*fields[2])(sites[2]-0) - (*fields[2])(sites[2]+0)) * ((*fields[2])(sites[2]-0) - (*fields[2])(sites[2]+0));
	(*fields[0])(sites[0]) += Real(0.125) * ((*fields[2])(sites[2]-1) - (*fields[2])(sites[2]+1)) * ((*fields[2])(sites[2]-1) - (*fields[2])(sites[2]+1));
	(*fields[0])(sites[0]) += Real(0.125) * ((*fields[2])(sites[2]-2) - (*fields[2])(sites[2]+2)) * ((*fields[2])(sites[2]-2) - (*fields[2])(sites[2]+2));
#endif
	(*fields[0])(sites[0]) += (params[3] * (Real(1) - Real(3) * (*fields[2])(sites[2])) - params[1]) * (*fields[2])(sites[2]) - params[3] * (*fields[3])(sites[3]);
}

// callable struct for prepareFTsource_T00

struct prepareFTsource_T00_functor
{
	__host__ __device__ void operator()(Field<Real> * fields[], Site * sites, int nfields, double * params, double * outputs)
	{
		prepareFTsource_T00(fields, sites, nfields, params, outputs);
	}
};

double prepareFTsource(Field<Real> & phi, Field<Real> & chi, Field<Real> & source, const double bgmodel, Field<Real> & result, const double coeff, const double coeff2, const double coeff3)
{
	Field<Real> * fields[4] = {&result, &source, &phi, &chi};
	double params[4] = {bgmodel, coeff, coeff2, coeff3};
	double sum = 0.;
	int reduce = SUM;

	double * d_params;
	double * d_sum;
	int * d_reduce;

	cudaMalloc(&d_params, 4 * sizeof(double));
	cudaMalloc(&d_sum, sizeof(double));
	cudaMalloc(&d_reduce, sizeof(int));
	cudaMemcpy(d_params, params, 4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_reduce, &reduce, sizeof(int), cudaMemcpyHostToDevice);

	int numpts = result.lattice().sizeLocal(0);
	int block_x = result.lattice().sizeLocal(1);
	int block_y = result.lattice().sizeLocal(2);

	lattice_for_each<prepareFTsource_T00_functor, 1><<<dim3(block_x, block_y), 128>>>(prepareFTsource_T00_functor(), numpts, fields, 4, d_params, d_sum, d_reduce);

	cudaDeviceSynchronize();

	cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_params);
	cudaFree(d_sum);
	cudaFree(d_reduce);

	parallel.sum<double>(sum);

	return sum;

/*	Site x(phi.lattice());
	
	for (x.first(); x.test(); x.next())
	{
		result(x) = coeff2 * (source(x) - bgmodel);
#ifdef PHINONLINEAR
#ifdef ORIGINALMETRIC
		result(x) *= 1. - 4. * phi(x);
		result(x) -= 0.375 * (phi(x-0) - phi(x+0)) * (phi(x-0) - phi(x+0));
		result(x) -= 0.375 * (phi(x-1) - phi(x+1)) * (phi(x-1) - phi(x+1));
		result(x) -= 0.375 * (phi(x-2) - phi(x+2)) * (phi(x-2) - phi(x+2));
#else
		result(x) *= 1. - 2. * phi(x) * (1. - phi(x));
		result(x) += 0.125 * (phi(x-0) - phi(x+0)) * (phi(x-0) - phi(x+0));
		result(x) += 0.125 * (phi(x-1) - phi(x+1)) * (phi(x-1) - phi(x+1));
		result(x) += 0.125 * (phi(x-2) - phi(x+2)) * (phi(x-2) - phi(x+2));
#endif
#endif
		result(x) += (coeff3 * (1. - 3. * phi(x)) - coeff) * phi(x) - coeff3 * chi(x);
	}*/
}


/*template <class FieldType>
void prepareFTsource(Field<FieldType> & T0i, Field<FieldType> & phi, const double coeff)
{
	Site x(phi.lattice());
	
	for (x.first(); x.test(); x.next())
	{
		T0i(x,0) *= (1. - 0.5 * (phi(x) + phi(x+0)));
		T0i(x,1) *= (1. - 0.5 * (phi(x) + phi(x+1)));
		T0i(x,2) *= (1. - 0.5 * (phi(x) + phi(x+2)));
	}
}*/

#ifdef FFT3D
//////////////////////////
// projectFTscalar
//////////////////////////
// Description:
//   projection of the Fourier image of a tensor field on the trace-free
//   longitudinal (scalar) component
// 
// Arguments:
//   SijFT      reference to the Fourier image of the input tensor field
//   chiFT      reference to allocated field which will contain the Fourier
//              image of the trace-free longitudinal (scalar) component
//
// Returns:
// 
//////////////////////////

void projectFTscalar(Field<Cplx> & SijFT, Field<Cplx> & chiFT, const int add = 0)
{
	const int linesize = chiFT.lattice().size(1);
	Real * gridk2;
	Cplx * kshift;
	rKSite k(chiFT.lattice());
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
		kshift = (Cplx *) alloca(linesize * sizeof(Cplx));
	}
	else
	{
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
		kshift = (Cplx *) malloc(linesize * sizeof(Cplx));
	}
	
#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		kshift[i] = gridk2[i] * Cplx(cos(M_PI * (Real) i / (Real) linesize), -sin(M_PI * (Real) i / (Real) linesize));
		gridk2[i] *= gridk2[i];
	}

	/*k.first();
	if (k.coord(0) == 0 && k.coord(1) == 0 && k.coord(2) == 0)
	{
		chiFT(k) = Cplx(0,0);
		k.next();
	}*/
	
	if (add)
	{
#pragma omp parallel for collapse(2) default(shared) firstprivate(k)
		for (int i = 0; i < chiFT.lattice().sizeLocal(1); i++)
		{
			for (int j = 0; j < chiFT.lattice().sizeLocal(2); j++)
			{
				if (!k.setCoord(0, j + chiFT.lattice().coordSkip()[0], i + chiFT.lattice().coordSkip()[1]))
				{
					throw std::runtime_error("Error in projectFTscalar: Could not set coordinates.");
				}

				for (int z = 0; z < chiFT.lattice().sizeLocal(0); z++)
				{
					chiFT(k) += ((gridk2[k.coord(1)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(0)]) * SijFT(k, 0, 0) +
						(gridk2[k.coord(0)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(1)]) * SijFT(k, 1, 1) +
						(gridk2[k.coord(0)] + gridk2[k.coord(1)] - Real(2) * gridk2[k.coord(2)]) * SijFT(k, 2, 2) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2) -
						Real(6) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) /
						(Real(2) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * linesize);
					k.next();
				}
			}
		}
		/*for (; k.test(); k.next())
		{
			chiFT(k) += ((gridk2[k.coord(1)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(0)]) * SijFT(k, 0, 0) +
						(gridk2[k.coord(0)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(1)]) * SijFT(k, 1, 1) +
						(gridk2[k.coord(0)] + gridk2[k.coord(1)] - Real(2) * gridk2[k.coord(2)]) * SijFT(k, 2, 2) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2) -
						Real(6) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) /
						(Real(2) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * linesize);
		}*/
	}
	else
	{
#pragma omp parallel for collapse(2) default(shared) firstprivate(k)
		for (int i = 0; i < chiFT.lattice().sizeLocal(1); i++)
		{
			for (int j = 0; j < chiFT.lattice().sizeLocal(2); j++)
			{
				if (!k.setCoord(0, j + chiFT.lattice().coordSkip()[0], i + chiFT.lattice().coordSkip()[1]))
				{
					throw std::runtime_error("Error in projectFTscalar: Could not set coordinates.");
				}

				for (int z = 0; z < chiFT.lattice().sizeLocal(0); z++)
				{
					chiFT(k) = ((gridk2[k.coord(1)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(0)]) * SijFT(k, 0, 0) +
						(gridk2[k.coord(0)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(1)]) * SijFT(k, 1, 1) +
						(gridk2[k.coord(0)] + gridk2[k.coord(1)] - Real(2) * gridk2[k.coord(2)]) * SijFT(k, 2, 2) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2) -
						Real(6) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) /
						(Real(2) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * linesize);
					k.next();
				}
			}
		}
		/*for (; k.test(); k.next())
		{
			chiFT(k) = ((gridk2[k.coord(1)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(0)]) * SijFT(k, 0, 0) +
						(gridk2[k.coord(0)] + gridk2[k.coord(2)] - Real(2) * gridk2[k.coord(1)]) * SijFT(k, 1, 1) +
						(gridk2[k.coord(0)] + gridk2[k.coord(1)] - Real(2) * gridk2[k.coord(2)]) * SijFT(k, 2, 2) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1) -
						Real(6) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2) -
						Real(6) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) /
						(Real(2) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]) * linesize);
		}*/
	}

	if (k.setCoord(0, 0, 0))
	{
		chiFT(k) = Cplx(0,0);
	}
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(gridk2);
		free(kshift);
	}
}


//////////////////////////
// evolveFTvector
//////////////////////////
// Description:
//   projects the Fourier image of a tensor field on the spin-1 component
//   used as a source for the evolution of the vector perturbation
// 
// Arguments:
//   SijFT      reference to the Fourier image of the input tensor field
//   BiFT       reference to the Fourier image of the vector perturbation
//   a2dtau     conformal time step times scale factor squared (a^2 * dtau)
//
// Returns:
// 
//////////////////////////

void evolveFTvector(Field<Cplx> & SijFT, Field<Cplx> & BiFT, const Real a2dtau)
{
	const int linesize = BiFT.lattice().size(1);
	Real * gridk2;
	Cplx * kshift;
	rKSite k(BiFT.lattice());
	Real k4;
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
		kshift = (Cplx *) alloca(linesize * sizeof(Cplx));
	}
	else
	{
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
		kshift = (Cplx *) malloc(linesize * sizeof(Cplx));
	}

#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		kshift[i] = gridk2[i] * Cplx(cos(M_PI * (Real) i / (Real) linesize), -sin(M_PI * (Real) i / (Real) linesize));
		gridk2[i] *= gridk2[i];
	}

#pragma omp parallel for collapse(2) default(shared) firstprivate(k) private(k4)
	for (int i = 0; i < BiFT.lattice().sizeLocal(1); i++)
	{
		for (int j = 0; j < BiFT.lattice().sizeLocal(2); j++)
		{
			if (!k.setCoord(0, j + BiFT.lattice().coordSkip()[0], i + BiFT.lattice().coordSkip()[1]))
			{
				std::cerr << "proc#" << parallel.rank() << ": Error in projectFTvector! Could not set coordinates at k=(0, " << j + BiFT.lattice().coordSkip()[0] << ", " << i + BiFT.lattice().coordSkip()[1] << ")" << std::endl;
				throw std::runtime_error("Error in projectFTvector: Could not set coordinates.");
			}

			for (int z = 0; z < BiFT.lattice().sizeLocal(0); z++)
			{
				k4 = gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)];
				k4 *= k4;
				
				BiFT(k, 0) += Cplx(0,Real(-2)*a2dtau/k4) * (kshift[k.coord(0)].conj() * ((gridk2[k.coord(1)] + gridk2[k.coord(2)]) * SijFT(k, 0, 0)
						- gridk2[k.coord(1)] * SijFT(k, 1, 1) - gridk2[k.coord(2)] * SijFT(k, 2, 2) - Real(2) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2))
						+ (gridk2[k.coord(1)] + gridk2[k.coord(2)] - gridk2[k.coord(0)]) * (kshift[k.coord(1)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 0, 2)));
				BiFT(k, 1) += Cplx(0,Real(-2)*a2dtau/k4) * (kshift[k.coord(1)].conj() * ((gridk2[k.coord(0)] + gridk2[k.coord(2)]) * SijFT(k, 1, 1)
						- gridk2[k.coord(0)] * SijFT(k, 0, 0) - gridk2[k.coord(2)] * SijFT(k, 2, 2) - Real(2) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2))
						+ (gridk2[k.coord(0)] + gridk2[k.coord(2)] - gridk2[k.coord(1)]) * (kshift[k.coord(0)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 1, 2)));
				BiFT(k, 2) += Cplx(0,Real(-2)*a2dtau/k4) * (kshift[k.coord(2)].conj() * ((gridk2[k.coord(0)] + gridk2[k.coord(1)]) * SijFT(k, 2, 2)
						- gridk2[k.coord(0)] * SijFT(k, 0, 0) - gridk2[k.coord(1)] * SijFT(k, 1, 1) - Real(2) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1))
						+ (gridk2[k.coord(0)] + gridk2[k.coord(1)] - gridk2[k.coord(2)]) * (kshift[k.coord(0)] * SijFT(k, 0, 2) + kshift[k.coord(1)] * SijFT(k, 1, 2)));

				k.next();
			}
		}
	}
	
	if (k.setCoord(0, 0, 0))
	{
		BiFT(k, 0) = Cplx(0,0);
		BiFT(k, 1) = Cplx(0,0);
		BiFT(k, 2) = Cplx(0,0);
	}
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(gridk2);
		free(kshift);
	}
}


//////////////////////////
// projectFTvector
//////////////////////////
// Description:
//   projects the Fourier image of a vector field on the transverse component
//   and solves the constraint equation for the vector perturbation
// 
// Arguments:
//   SiFT       reference to the Fourier image of the input vector field
//   BiFT       reference to the Fourier image of the vector perturbation (can be identical to input)
//   coeff      rescaling coefficient (default 1)
//   modif      modification k^2 -> k^2 + modif (default 0)
//
// Returns:
// 
//////////////////////////

void projectFTvector(Field<Cplx> & SiFT, Field<Cplx> & BiFT, const Real coeff = 1, const Real modif = 0)
{
	const int linesize = BiFT.lattice().size(1);
	Real * gridk2;
	Cplx * kshift;
	rKSite k(BiFT.lattice());
	Real k2;
	Cplx tmp(0, 0);
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
		kshift = (Cplx *) alloca(linesize * sizeof(Cplx));
	}
	else
	{
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
		kshift = (Cplx *) malloc(linesize * sizeof(Cplx));
	}

#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		kshift[i] = gridk2[i] * Cplx(cos(M_PI * (Real) i / (Real) linesize), -sin(M_PI * (Real) i / (Real) linesize));
		gridk2[i] *= gridk2[i];
	}

#pragma omp parallel for collapse(2) default(shared) firstprivate(k) private(k2, tmp)
	for (int i = 0; i < BiFT.lattice().sizeLocal(1); i++)
	{
		for (int j = 0; j < BiFT.lattice().sizeLocal(2); j++)
		{
			if (!k.setCoord(0, j + BiFT.lattice().coordSkip()[0], i + BiFT.lattice().coordSkip()[1]))
			{
				std::cerr << "proc#" << parallel.rank() << ": Error in projectFTvector! Could not set coordinates at k=(0, " << j + BiFT.lattice().coordSkip()[0] << ", " << i + BiFT.lattice().coordSkip()[1] << ")" << std::endl;
				throw std::runtime_error("Error in projectFTvector: Could not set coordinates.");
			}

			for (int z = 0; z < BiFT.lattice().sizeLocal(0); z++)
			{
				k2 = gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)];
		
				tmp = (kshift[k.coord(0)] * SiFT(k, 0) + kshift[k.coord(1)] * SiFT(k, 1) + kshift[k.coord(2)] * SiFT(k, 2)) / k2;
				
				BiFT(k, 0) = (SiFT(k, 0) - kshift[k.coord(0)].conj() * tmp) * Real(4) * coeff / (k2 + modif);
				BiFT(k, 1) = (SiFT(k, 1) - kshift[k.coord(1)].conj() * tmp) * Real(4) * coeff / (k2 + modif);
				BiFT(k, 2) = (SiFT(k, 2) - kshift[k.coord(2)].conj() * tmp) * Real(4) * coeff / (k2 + modif);

				k.next();
			}
		}
	}
	
	if (k.setCoord(0, 0, 0))
	{
		BiFT(k, 0) = Cplx(0,0);
		BiFT(k, 1) = Cplx(0,0);
		BiFT(k, 2) = Cplx(0,0);
	}
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(gridk2);
		free(kshift);
	}
}


//////////////////////////
// evolveFTtensor
//////////////////////////
// Description:
//   projection of the Fourier image of a tensor field on the transverse
//   trace-free tensor component
// 
// Arguments:
//   SijFT      reference to the Fourier image of the input tensor field
//   hijFT      reference to the Fourier image of the current hij
//   hijprimeFT reference to the Fourier image of hij' that will be updated
//   hubble     conformal Hubble rate
//   dtau       next time step
//   dtau_old   previous time step
//
// Returns:
// 
//////////////////////////

void evolveFTtensor(Field<Cplx> & SijFT, Field<Cplx> & hijFT, Field<Cplx> & hijprimeFT, const double hubble, const double dtau, const double dtau_old)
{
	const int linesize = hijFT.lattice().size(1);
	Real * gridk2;
	Cplx * kshift;
	rKSite k(hijFT.lattice());
	Real k2, k4;
	Real dtau_mean = 0.5 * (dtau_old + dtau);
	Real one_plus_hubble_dtau = 1. + hubble * dtau_mean;
	Real one_minus_hubble_dtau = 1. - hubble * dtau_mean;

	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
		kshift = (Cplx *) alloca(linesize * sizeof(Cplx));
	}
	else
	{
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
		kshift = (Cplx *) malloc(linesize * sizeof(Cplx));
	}

#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		kshift[i] = gridk2[i] * Cplx(cos(M_PI * (Real) i / (Real) linesize), -sin(M_PI * (Real) i / (Real) linesize));
		gridk2[i] *= gridk2[i];
	}

#pragma omp parallel for collapse(2) default(shared) firstprivate(k) private(k2, k4)
	for (int i = 0; i < hijFT.lattice().sizeLocal(1); i++)
	{
		for (int j = 0; j < hijFT.lattice().sizeLocal(2); j++)
		{
			if (!k.setCoord(0, j + hijFT.lattice().coordSkip()[0], i + hijFT.lattice().coordSkip()[1]))
			{
				std::cerr << "proc#" << parallel.rank() << ": Error in evolveFTtensor! Could not set coordinates at k=(0, " << j + hijFT.lattice().coordSkip()[0] << ", " << i + hijFT.lattice().coordSkip()[1] << ")" << std::endl;
				throw std::runtime_error("Error in evolveFTtensor: Could not set coordinates.");
			}

			for (int z = 0; z < hijFT.lattice().sizeLocal(0); z++)
			{
				k2 = gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)];
				k4 = k2 * k2 * linesize;
				
				if (k2 * dtau_mean * dtau_mean < Real(1))
				{
					for (int c = 0; c < hijprimeFT.components(); c++)
						hijFT(k, c) += hijprimeFT(k, c) * dtau_old;
				
					if (k2 > 0)
					{
						hijprimeFT(k, 0, 0) = (one_minus_hubble_dtau * hijprimeFT(k, 0, 0) + (((gridk2[k.coord(0)] - k2) * ((gridk2[k.coord(0)] - k2) * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(0)] * (kshift[k.coord(1)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 0, 2)))
								+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SijFT(k, 1, 1)
								+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SijFT(k, 2, 2)
								+ Real(2) * (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) / k4 - k2 * hijFT(k, 0, 0)) * dtau_mean) / one_plus_hubble_dtau;
				
						hijprimeFT(k, 0, 1) = (one_minus_hubble_dtau * hijprimeFT(k, 0, 1) + ((Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(1)] - k2) * SijFT(k, 0, 1) + (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(1)].conj() * SijFT(k, 2, 2)
								+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(0)].conj() * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(2)] * SijFT(k, 0, 2))
								+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(1)].conj() * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(2)] * SijFT(k, 1, 2))) / k4 - k2 * hijFT(k, 0, 1)) * dtau_mean) / one_plus_hubble_dtau;
				
						hijprimeFT(k, 0, 2) = (one_minus_hubble_dtau * hijprimeFT(k, 0, 2) + ((Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(2)] - k2) * SijFT(k, 0, 2) + (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(2)].conj() * SijFT(k, 1, 1)
								+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(0)].conj() * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(1)] * SijFT(k, 0, 1))
								+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(2)].conj() * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(1)] * SijFT(k, 1, 2))) / k4 - k2 * hijFT(k, 0, 2)) * dtau_mean) / one_plus_hubble_dtau;
				
						hijprimeFT(k, 1, 1) = (one_minus_hubble_dtau * hijprimeFT(k, 1, 1) + (((gridk2[k.coord(1)] - k2) * ((gridk2[k.coord(1)] - k2) * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(1)] * (kshift[k.coord(0)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 1, 2)))
								+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SijFT(k, 0, 0)
								+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SijFT(k, 2, 2)
								+ Real(2) * (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2)) / k4 - k2 * hijFT(k, 1, 1)) * dtau_mean) / one_plus_hubble_dtau;
				
						hijprimeFT(k, 1, 2) = (one_minus_hubble_dtau * hijprimeFT(k, 1, 2) + ((Real(2) * (gridk2[k.coord(1)] - k2) * (gridk2[k.coord(2)] - k2) * SijFT(k, 1, 2) + (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)].conj() * kshift[k.coord(2)].conj() * SijFT(k, 0, 0)
								+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(1)].conj() * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(0)] * SijFT(k, 0, 1))
								+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(2)].conj() * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(0)] * SijFT(k, 0, 2))) / k4 - k2 * hijFT(k, 1, 2)) * dtau_mean) / one_plus_hubble_dtau;
				
						hijprimeFT(k, 2, 2) = (one_minus_hubble_dtau * hijprimeFT(k, 2, 2) + (((gridk2[k.coord(2)] - k2) * ((gridk2[k.coord(2)] - k2) * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(2)] * (kshift[k.coord(0)] * SijFT(k, 0, 2) + kshift[k.coord(1)] * SijFT(k, 1, 2)))
								+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SijFT(k, 0, 0)
								+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SijFT(k, 1, 1)
								+ Real(2) * (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1)) / k4 - k2 * hijFT(k, 2, 2)) * dtau_mean) / one_plus_hubble_dtau;
					}
					else
					{
						Cplx trS = SijFT(k, 0, 0) + SijFT(k, 1, 1) + SijFT(k, 2, 2);

						SijFT(k, 0, 0) -= trS / Real(3);
						SijFT(k, 1, 1) -= trS / Real(3);
						SijFT(k, 2, 2) -= trS / Real(3);

						for (int c = 0; c < hijprimeFT.components(); c++)
						{
							hijprimeFT(k, c) = (one_minus_hubble_dtau * hijprimeFT(k, c) + Real(2) * SijFT(k, c) * dtau_mean / linesize) / one_plus_hubble_dtau;
						}
					}
				}
				else
				{
					for (int c = 0; c < hijprimeFT.components(); c++)
						hijprimeFT(k, c) = Cplx(0.,0.);
				
					hijFT(k, 0, 0) = (((gridk2[k.coord(0)] - k2) * ((gridk2[k.coord(0)] - k2) * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(0)] * (kshift[k.coord(1)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 0, 2)))
							+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SijFT(k, 1, 1)
							+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SijFT(k, 2, 2)
							+ Real(2) * (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)] * kshift[k.coord(2)] * SijFT(k, 1, 2)) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 0, 0)) / (k2 * dtau_old + Real(2) * hubble);
				
					hijFT(k, 0, 1) = ((Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(1)] - k2) * SijFT(k, 0, 1) + (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(1)].conj() * SijFT(k, 2, 2)
							+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(0)].conj() * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(2)] * SijFT(k, 0, 2))
							+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(1)].conj() * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(2)] * SijFT(k, 1, 2))) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 0, 1)) / (k2 * dtau_old + Real(2) * hubble);
				
					hijFT(k, 0, 2) = ((Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(2)] - k2) * SijFT(k, 0, 2) + (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(2)].conj() * SijFT(k, 1, 1)
							+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(0)].conj() * SijFT(k, 0, 0) + Real(2) * kshift[k.coord(1)] * SijFT(k, 0, 1))
							+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(2)].conj() * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(1)] * SijFT(k, 1, 2))) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 0, 2)) / (k2 * dtau_old + Real(2) * hubble);
				
					hijFT(k, 1, 1) = (((gridk2[k.coord(1)] - k2) * ((gridk2[k.coord(1)] - k2) * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(1)] * (kshift[k.coord(0)] * SijFT(k, 0, 1) + kshift[k.coord(2)] * SijFT(k, 1, 2)))
							+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SijFT(k, 0, 0)
							+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SijFT(k, 2, 2)
							+ Real(2) * (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)] * kshift[k.coord(2)] * SijFT(k, 0, 2)) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 1, 1)) / (k2 * dtau_old + Real(2) * hubble);
				
					hijFT(k, 1, 2) = ((Real(2) * (gridk2[k.coord(1)] - k2) * (gridk2[k.coord(2)] - k2) * SijFT(k, 1, 2) + (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)].conj() * kshift[k.coord(2)].conj() * SijFT(k, 0, 0)
							+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(1)].conj() * SijFT(k, 1, 1) + Real(2) * kshift[k.coord(0)] * SijFT(k, 0, 1))
							+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(2)].conj() * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(0)] * SijFT(k, 0, 2))) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 1, 2)) / (k2 * dtau_old + Real(2) * hubble);
				
					hijFT(k, 2, 2) = (((gridk2[k.coord(2)] - k2) * ((gridk2[k.coord(2)] - k2) * SijFT(k, 2, 2) + Real(2) * kshift[k.coord(2)] * (kshift[k.coord(0)] * SijFT(k, 0, 2) + kshift[k.coord(1)] * SijFT(k, 1, 2)))
							+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SijFT(k, 0, 0)
							+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SijFT(k, 1, 1)
							+ Real(2) * (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)] * kshift[k.coord(1)] * SijFT(k, 0, 1)) * dtau_old / k4 + Real(2) * hubble * hijFT(k, 2, 2)) / (k2 * dtau_old + Real(2) * hubble);
				}

				k.next();
			}
		}
	}
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(gridk2);
		free(kshift);
	}
}


//////////////////////////
// projectFTtensor
//////////////////////////
// Description:
//   projection of the Fourier image of a tensor field on the transverse
//   trace-free tensor component
// 
// Arguments:
//   SijFT      reference to the Fourier image of the input tensor field
//   hijFT      reference to allocated field which will contain the Fourier
//              image of the transverse trace-free tensor component
//
// Returns:
// 
//////////////////////////

void projectFTtensor(Field<Cplx> & SijFT, Field<Cplx> & hijFT)
{
	const int linesize = hijFT.lattice().size(1);
	Real * gridk2;
	Cplx * kshift;
	rKSite k(hijFT.lattice());
	Cplx SxxFT, SxyFT, SxzFT, SyyFT, SyzFT, SzzFT;
	Real k2, k6;
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
	{
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
		kshift = (Cplx *) alloca(linesize * sizeof(Cplx));
	}
	else
	{
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
		kshift = (Cplx *) malloc(linesize * sizeof(Cplx));
	}

#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		kshift[i] = gridk2[i] * Cplx(cos(M_PI * (Real) i / (Real) linesize), -sin(M_PI * (Real) i / (Real) linesize));
		gridk2[i] *= gridk2[i];
	}

#pragma omp parallel for collapse(2) default(shared) firstprivate(k) private(SxxFT, SxyFT, SxzFT, SyyFT, SyzFT, SzzFT, k2, k6)
	for (int i = 0; i < hijFT.lattice().sizeLocal(1); i++)
	{
		for (int j = 0; j < hijFT.lattice().sizeLocal(2); j++)
		{
			if (!k.setCoord(0, j + hijFT.lattice().coordSkip()[0], i + hijFT.lattice().coordSkip()[1]))
			{
				std::cerr << "proc#" << parallel.rank() << ": Error in projectFTtensor! Could not set coordinates at k=(0, " << j + hijFT.lattice().coordSkip()[0] << ", " << i + hijFT.lattice().coordSkip()[1] << ")" << std::endl;
				throw std::runtime_error("Error in projectFTtensor: Could not set coordinates.");
			}

			for (int z = 0; z < hijFT.lattice().sizeLocal(0); z++)
			{
				SxxFT = SijFT(k, 0, 0);
				SxyFT = SijFT(k, 0, 1);
				SxzFT = SijFT(k, 0, 2);
				SyyFT = SijFT(k, 1, 1);
				SyzFT = SijFT(k, 1, 2);
				SzzFT = SijFT(k, 2, 2);
				
				k2 = gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)];
				k6 = k2 * k2 * k2 * linesize;
				
				hijFT(k, 0, 0) = ((gridk2[k.coord(0)] - k2) * ((gridk2[k.coord(0)] - k2) * SxxFT + Real(2) * kshift[k.coord(0)] * (kshift[k.coord(1)] * SxyFT + kshift[k.coord(2)] * SxzFT))
						+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SyyFT
						+ ((gridk2[k.coord(0)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SzzFT
						+ Real(2) * (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)] * kshift[k.coord(2)] * SyzFT) / k6;
				
				hijFT(k, 0, 1) = (Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(1)] - k2) * SxyFT + (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(1)].conj() * SzzFT
						+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(0)].conj() * SxxFT + Real(2) * kshift[k.coord(2)] * SxzFT)
						+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(1)].conj() * SyyFT + Real(2) * kshift[k.coord(2)] * SyzFT)) / k6;
				
				hijFT(k, 0, 2) = (Real(2) * (gridk2[k.coord(0)] - k2) * (gridk2[k.coord(2)] - k2) * SxzFT + (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)].conj() * kshift[k.coord(2)].conj() * SyyFT
						+ (gridk2[k.coord(0)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(0)].conj() * SxxFT + Real(2) * kshift[k.coord(1)] * SxyFT)
						+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(0)].conj() * (kshift[k.coord(2)].conj() * SzzFT + Real(2) * kshift[k.coord(1)] * SyzFT)) / k6;
				
				hijFT(k, 1, 1) = ((gridk2[k.coord(1)] - k2) * ((gridk2[k.coord(1)] - k2) * SyyFT + Real(2) * kshift[k.coord(1)] * (kshift[k.coord(0)] * SxyFT + kshift[k.coord(2)] * SyzFT))
						+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SxxFT
						+ ((gridk2[k.coord(1)] + k2) * (gridk2[k.coord(2)] + k2) - Real(2) * k2 * k2) * SzzFT
						+ Real(2) * (gridk2[k.coord(1)] + k2) * kshift[k.coord(0)] * kshift[k.coord(2)] * SxzFT) / k6;
				
				hijFT(k, 1, 2) = (Real(2) * (gridk2[k.coord(1)] - k2) * (gridk2[k.coord(2)] - k2) * SyzFT + (gridk2[k.coord(0)] + k2) * kshift[k.coord(1)].conj() * kshift[k.coord(2)].conj() * SxxFT
						+ (gridk2[k.coord(1)] - k2) * kshift[k.coord(2)].conj() * (kshift[k.coord(1)].conj() * SyyFT + Real(2) * kshift[k.coord(0)] * SxyFT)
						+ (gridk2[k.coord(2)] - k2) * kshift[k.coord(1)].conj() * (kshift[k.coord(2)].conj() * SzzFT + Real(2) * kshift[k.coord(0)] * SxzFT)) / k6;
				
				hijFT(k, 2, 2) = ((gridk2[k.coord(2)] - k2) * ((gridk2[k.coord(2)] - k2) * SzzFT + Real(2) * kshift[k.coord(2)] * (kshift[k.coord(0)] * SxzFT + kshift[k.coord(1)] * SyzFT))
						+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(0)] + k2) - Real(2) * k2 * k2) * SxxFT
						+ ((gridk2[k.coord(2)] + k2) * (gridk2[k.coord(1)] + k2) - Real(2) * k2 * k2) * SyyFT
						+ Real(2) * (gridk2[k.coord(2)] + k2) * kshift[k.coord(0)] * kshift[k.coord(1)] * SxyFT) / k6;

				k.next();
			}
		}
	}

	if (k.setCoord(0, 0, 0))
	{
		for (int i = 0; i < hijFT.components(); i++)
			hijFT(k, i) = Cplx(0,0);
	}
	
	/*k.first();
	if (k.coord(0) == 0 && k.coord(1) == 0 && k.coord(2) == 0)
	{
		for (int i = 0; i < hijFT.components(); i++)
			hijFT(k, i) = Cplx(0,0);
			
		k.next();
	}
	
	for (; k.test(); k.next())
	{
		
	}*/
	
	if (linesize > STACK_ALLOCATION_LIMIT)
	{
		free(gridk2);
		free(kshift);
	}
}


//////////////////////////
// solveModifiedPoissonFT
//////////////////////////
// Description:
//   Modified Poisson solver using the standard Fourier method
// 
// Arguments:
//   sourceFT   reference to the Fourier image of the source field
//   potFT      reference to the Fourier image of the potential
//   coeff      coefficient applied to the source ("4 pi G / a")
//   modif      modification k^2 -> k^2 + modif (default 0 gives standard Poisson equation)
//
// Returns:
// 
//////////////////////////

void solveModifiedPoissonFT(Field<Cplx> & sourceFT, Field<Cplx> & potFT, Real coeff, const Real modif = 0)
{
	const int linesize = potFT.lattice().size(1);
	Real * gridk2;
	rKSite k(potFT.lattice());
	
	if (linesize <= STACK_ALLOCATION_LIMIT)
		gridk2 = (Real *) alloca(linesize * sizeof(Real));
	else
		gridk2 = (Real *) malloc(linesize * sizeof(Real));
	
	coeff /= -((long) linesize * (long) linesize * (long) linesize);

#pragma omp parallel for
	for (int i = 0; i < linesize; i++)
	{
		gridk2[i] = Real(2) * (Real) linesize * sin(M_PI * (Real) i / (Real) linesize);
		gridk2[i] *= gridk2[i];
	}

#pragma omp parallel for collapse(2) default(shared) firstprivate(k)
	for (int i = 0; i < potFT.lattice().sizeLocal(1); i++)
	{
		for (int j = 0; j < potFT.lattice().sizeLocal(2); j++)
		{
			if (!k.setCoord(0, j + potFT.lattice().coordSkip()[0], i + potFT.lattice().coordSkip()[1]))
			{
				std::cerr << "proc#" << parallel.rank() << ": Error in solveModifiedPoissonFT! Could not set coordinates at k=(0, " << j + potFT.lattice().coordSkip()[0] << ", " << i + potFT.lattice().coordSkip()[1] << ")" << std::endl;
				throw std::runtime_error("Error in solveModifiedPoissonFT: Could not set coordinates.");
			}

			for (int z = 0; z < potFT.lattice().sizeLocal(0); z++)
			{
				potFT(k) = sourceFT(k) * coeff / (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)] + modif);
				k.next();
			}
		}
	}

	if (modif == 0 && k.setCoord(0, 0, 0))
	{
		potFT(k) = Cplx(0,0);
	}
	
	/*k.first();
	if (k.coord(0) == 0 && k.coord(1) == 0 && k.coord(2) == 0)
	{
		if (modif == 0)
			potFT(k) = Cplx(0,0);
		else
			potFT(k) = sourceFT(k) * coeff / modif;
		k.next();
	}
	
	for (; k.test(); k.next())
	{
		potFT(k) = sourceFT(k) * coeff / (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)] + modif);
	}*/
	
	if (linesize > STACK_ALLOCATION_LIMIT)
		free(gridk2);
}
#endif


//////////////////////////
// update_q
//////////////////////////
// Description:
//   Update momentum method (arbitrary momentum)
//   Note that vel[3] in the particle structure is used to store q[3] in units
//   of the particle mass, such that as q^2 << m^2 a^2 the meaning of vel[3]
//   is ~ v*a.
// 
// Arguments:
//   dtau       time step
//   dx         lattice unit  
//   part       pointer to particle structure
//   ref_dist   distance vector to reference point
//   partInfo   global particle properties (unused)
//   fields     array of pointers to fields appearing in geodesic equation
//              fields[0] = phi
//              fields[1] = chi
//              fields[2] = Bi
//   sites      array of sites on the respective lattices
//   nfield     number of fields
//   params     array of additional parameters
//              params[0] = a
//              params[1] = scaling coefficient for Bi
//              params[2] - params[6] = hij_hom[0] - hij_hom[4] for anisotropic expansion
//   outputs    array of reduction variables
//   noutputs   number of reduction variables
//
// Returns: squared velocity of particle after update
// 
//////////////////////////

__host__ __device__ Real update_q(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
#define phi (*fields[0])
#define chi (*fields[1])
#define Bi (*fields[2])
#define xphi (sites[0])
#define xchi (sites[1])
#define xB (sites[2])
	
	Real gradphi[3]={0,0,0};
	Real pgradB[3]={0,0,0};
	Real v2 = (*part).vel[0] * (*part).vel[0] + (*part).vel[1] * (*part).vel[1] + (*part).vel[2] * (*part).vel[2];
#ifdef ANISOTROPIC_EXPANSION
	v2 -= (*part).vel[0] * (*part).vel[0] * params[2] + (*part).vel[1] * (*part).vel[1] * params[5] - (*part).vel[2] * (*part).vel[2] * (params[2] + params[5]) + Real(2) * (*part).vel[0] * (*part).vel[1] * params[3] + Real(2) * (*part).vel[0] * (*part).vel[2] * params[4] + Real(2) * (*part).vel[1] * (*part).vel[2] * params[6];
#endif
	Real e2 = v2 + params[0] * params[0];

	Real phiint = (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]) * phi(xphi);
	phiint += ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]) * phi(xphi+0);
	phiint += (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]) * phi(xphi+1);
	phiint += ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]) * phi(xphi+1+0);
	phiint += (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2] * phi(xphi+2);
	phiint += ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2] * phi(xphi+2+0);
	phiint += (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2] * phi(xphi+2+1);
	phiint += ref_dist[0] * ref_dist[1] * ref_dist[2] * phi(xphi+2+1+0);
	
	phiint /= e2;
	phiint = 4. * phiint * v2 + (1. + params[0] * params[0] * phiint) * (v2 + e2) / e2;

#if GRADIENT_ORDER == 1	
	gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * (phi(xphi+0) - phi(xphi));
	gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * (phi(xphi+1) - phi(xphi));
	gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * (phi(xphi+2) - phi(xphi));
	gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * (phi(xphi+1+0) - phi(xphi+1));
	gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * (phi(xphi+1+0) - phi(xphi+0));
	gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * (phi(xphi+2+0) - phi(xphi+0));
	gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * (phi(xphi+2+0) - phi(xphi+2));
	gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * (phi(xphi+2+1) - phi(xphi+2));
	gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * (phi(xphi+2+1) - phi(xphi+1));
	gradphi[0] += ref_dist[1] * ref_dist[2] * (phi(xphi+2+1+0) - phi(xphi+2+1));
	gradphi[1] += ref_dist[0] * ref_dist[2] * (phi(xphi+2+1+0) - phi(xphi+2+0));
	gradphi[2] += ref_dist[0] * ref_dist[1] * (phi(xphi+2+1+0) - phi(xphi+1+0));
#elif GRADIENT_ORDER == 2
	for (int i=0; i<3; i++)
	{
		gradphi[i] = 0.5 * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (phi(xphi+i) - phi(xphi-i));
		gradphi[i] += 0.5 * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (phi(xphi+i+0) - phi(xphi-i+0));
		gradphi[i] += 0.5 * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]) * (phi(xphi+i+1) - phi(xphi-i+1));
		gradphi[i] += 0.5 * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]) * (phi(xphi+i+1+0) - phi(xphi-i+1+0));
		gradphi[i] += 0.5 * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2] * (phi(xphi+2+i) - phi(xphi+2-i));
		gradphi[i] += 0.5 * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2] * (phi(xphi+2+i+0) - phi(xphi+2-i+0));
		gradphi[i] += 0.5 * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2] * (phi(xphi+2+i+1) - phi(xphi+2-i+1));
		gradphi[i] += 0.5 * ref_dist[0] * ref_dist[1] * ref_dist[2] * (phi(xphi+2+i+1+0) - phi(xphi+2-i+1+0));
	}
#else
#error GRADIENT_ORDER must be set to 1 or 2
#endif

	gradphi[0] *= phiint;
	gradphi[1] *= phiint;
	gradphi[2] *= phiint;
	
	if (nfield>=2 && fields[1] != NULL)
	{
		gradphi[0] -= (1.-ref_dist[1]) * (1.-ref_dist[2]) * (chi(xchi+0) - chi(xchi));
		gradphi[1] -= (1.-ref_dist[0]) * (1.-ref_dist[2]) * (chi(xchi+1) - chi(xchi));
		gradphi[2] -= (1.-ref_dist[0]) * (1.-ref_dist[1]) * (chi(xchi+2) - chi(xchi));
		gradphi[0] -= ref_dist[1] * (1.-ref_dist[2]) * (chi(xchi+1+0) - chi(xchi+1));
		gradphi[1] -= ref_dist[0] * (1.-ref_dist[2]) * (chi(xchi+1+0) - chi(xchi+0));
		gradphi[2] -= ref_dist[0] * (1.-ref_dist[1]) * (chi(xchi+2+0) - chi(xchi+0));
		gradphi[0] -= (1.-ref_dist[1]) * ref_dist[2] * (chi(xchi+2+0) - chi(xchi+2));
		gradphi[1] -= (1.-ref_dist[0]) * ref_dist[2] * (chi(xchi+2+1) - chi(xchi+2));
		gradphi[2] -= (1.-ref_dist[0]) * ref_dist[1] * (chi(xchi+2+1) - chi(xchi+1));
		gradphi[0] -= ref_dist[1] * ref_dist[2] * (chi(xchi+2+1+0) - chi(xchi+2+1));
		gradphi[1] -= ref_dist[0] * ref_dist[2] * (chi(xchi+2+1+0) - chi(xchi+2+0));
		gradphi[2] -= ref_dist[0] * ref_dist[1] * (chi(xchi+2+1+0) - chi(xchi+1+0));
	}
	
	e2 = sqrt(e2);

	if (nfield>=3 && fields[2] != NULL)
	{
		pgradB[0] = ((1.-ref_dist[2]) * (Bi(xB+0,1) - Bi(xB,1)) + ref_dist[2] * (Bi(xB+2+0,1) - Bi(xB+2,1))) * (*part).vel[1];
		pgradB[0] += ((1.-ref_dist[1]) * (Bi(xB+0,2) - Bi(xB,2)) + ref_dist[1] * (Bi(xB+1+0,2) - Bi(xB+1,2))) * (*part).vel[2];
		pgradB[0] += (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((ref_dist[0]-1.) * Bi(xB-0,0) + (1.-2.*ref_dist[0]) * Bi(xB,0) + ref_dist[0] * Bi(xB+0,0)) * (*part).vel[0];
		pgradB[0] += ref_dist[1] * (1.-ref_dist[2]) * ((ref_dist[0]-1.) * Bi(xB+1-0,0) + (1.-2.*ref_dist[0]) * Bi(xB+1,0) + ref_dist[0] * Bi(xB+1+0,0)) * (*part).vel[0];
		pgradB[0] += (1.-ref_dist[1]) * ref_dist[2] * ((ref_dist[0]-1.) * Bi(xB+2-0,0) + (1.-2.*ref_dist[0]) * Bi(xB+2,0) + ref_dist[0] * Bi(xB+2+0,0)) * (*part).vel[0];
		pgradB[0] += ref_dist[1] * ref_dist[2] * ((ref_dist[0]-1.) * Bi(xB+2+1-0,0) + (1.-2.*ref_dist[0]) * Bi(xB+2+1,0) + ref_dist[0] * Bi(xB+2+1+0,0)) * (*part).vel[0];
		
		pgradB[1] = ((1.-ref_dist[0]) * (Bi(xB+1,2) - Bi(xB,2)) + ref_dist[0] * (Bi(xB+1+0,2) - Bi(xB+0,2))) * (*part).vel[2];
		pgradB[1] += ((1.-ref_dist[2]) * (Bi(xB+1,0) - Bi(xB,0)) + ref_dist[2] * (Bi(xB+1+2,0) - Bi(xB+2,0))) * (*part).vel[0];
		pgradB[1] += (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((ref_dist[1]-1.) * Bi(xB-1,1) + (1.-2.*ref_dist[1]) * Bi(xB,1) + ref_dist[1] * Bi(xB+1,1)) * (*part).vel[1];
		pgradB[1] += ref_dist[0] * (1.-ref_dist[2]) * ((ref_dist[1]-1.) * Bi(xB+0-1,1) + (1.-2.*ref_dist[1]) * Bi(xB+0,1) + ref_dist[1] * Bi(xB+0+1,1)) * (*part).vel[1];
		pgradB[1] += (1.-ref_dist[0]) * ref_dist[2] * ((ref_dist[1]-1.) * Bi(xB+2-1,1) + (1.-2.*ref_dist[1]) * Bi(xB+2,1) + ref_dist[1] * Bi(xB+2+1,1)) * (*part).vel[1];
		pgradB[1] += ref_dist[0] * ref_dist[2] * ((ref_dist[1]-1.) * Bi(xB+2+0-1,1) + (1.-2.*ref_dist[1]) * Bi(xB+2+0,1) + ref_dist[1] * Bi(xB+2+0+1,1)) * (*part).vel[1];
		
		pgradB[2] = ((1.-ref_dist[1]) * (Bi(xB+2,0) - Bi(xB,0)) + ref_dist[1] * (Bi(xB+2+1,0) - Bi(xB+1,0))) * (*part).vel[0];
		pgradB[2] += ((1.-ref_dist[0]) * (Bi(xB+2,1) - Bi(xB,1)) + ref_dist[0] * (Bi(xB+2+0,1) - Bi(xB+0,1))) * (*part).vel[1];
		pgradB[2] += (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((ref_dist[2]-1.) * Bi(xB-2,2) + (1.-2.*ref_dist[2]) * Bi(xB,2) + ref_dist[2] * Bi(xB+2,2)) * (*part).vel[2];
		pgradB[2] += ref_dist[0] * (1.-ref_dist[1]) * ((ref_dist[2]-1.) * Bi(xB+0-2,2) + (1.-2.*ref_dist[2]) * Bi(xB+0,2) + ref_dist[2] * Bi(xB+0+2,2)) * (*part).vel[2];
		pgradB[2] += (1.-ref_dist[0]) * ref_dist[1] * ((ref_dist[2]-1.) * Bi(xB+1-2,2) + (1.-2.*ref_dist[2]) * Bi(xB+1,2) + ref_dist[2] * Bi(xB+2+1,2)) * (*part).vel[2];
		pgradB[2] += ref_dist[0] * ref_dist[1] * ((ref_dist[2]-1.) * Bi(xB+1+0-2,2) + (1.-2.*ref_dist[2]) * Bi(xB+1+0,2) + ref_dist[2] * Bi(xB+1+0+2,2)) * (*part).vel[2];
		
		gradphi[0] += pgradB[0] / params[1] / e2;
		gradphi[1] += pgradB[1] / params[1] / e2;
		gradphi[2] += pgradB[2] / params[1] / e2;
	}
	
	v2 = 0.;
	for (int i=0;i<3;i++)
	{
		(*part).vel[i] -= dtau * e2 * gradphi[i] / dx;
		v2 += (*part).vel[i] * (*part).vel[i];
	}
	
	return v2 / params[0] / params[0];
	
#undef phi
#undef chi
#undef Bi
#undef xphi
#undef xchi
#undef xB
}

// callable struct for update_q
struct update_q_functor
{
	__host__ __device__ Real operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		return update_q(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// update_q_Newton
//////////////////////////
// Description:
//   Update momentum method (Newtonian version)
//   Note that vel[3] in the particle structure is used to store q[3] in units
//   of the particle mass, such that the meaning of vel[3] is v*a.
// 
// Arguments:
//   dtau       time step
//   dx         lattice unit
//   part       pointer to particle structure
//   ref_dist   distance vector to reference point
//   partInfo   global particle properties (unused)
//   fields     array of pointers to fields appearing in geodesic equation
//              fields[0] = psi
//              fields[1] = chi
//   sites      array of sites on the respective lattices
//   nfield     number of fields (should be 1)
//   params     array of additional parameters
//              params[0] = a
//   outputs    array of reduction variables
//   noutputs   number of reduction variables
//
// Returns: squared velocity of particle after update
// 
//////////////////////////

__host__ __device__ Real update_q_Newton(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
#define psi (*fields[0])
#define xpsi (sites[0])
#define chi (*fields[1])
#define xchi (sites[1])
	
	Real gradpsi[3]={0,0,0};

#if GRADIENT_ORDER == 1	
	gradpsi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * (psi(xpsi+0) - psi(xpsi));
	gradpsi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * (psi(xpsi+1) - psi(xpsi));
	gradpsi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * (psi(xpsi+2) - psi(xpsi));
	gradpsi[0] += ref_dist[1] * (1.-ref_dist[2]) * (psi(xpsi+1+0) - psi(xpsi+1));
	gradpsi[1] += ref_dist[0] * (1.-ref_dist[2]) * (psi(xpsi+1+0) - psi(xpsi+0));
	gradpsi[2] += ref_dist[0] * (1.-ref_dist[1]) * (psi(xpsi+2+0) - psi(xpsi+0));
	gradpsi[0] += (1.-ref_dist[1]) * ref_dist[2] * (psi(xpsi+2+0) - psi(xpsi+2));
	gradpsi[1] += (1.-ref_dist[0]) * ref_dist[2] * (psi(xpsi+2+1) - psi(xpsi+2));
	gradpsi[2] += (1.-ref_dist[0]) * ref_dist[1] * (psi(xpsi+2+1) - psi(xpsi+1));
	gradpsi[0] += ref_dist[1] * ref_dist[2] * (psi(xpsi+2+1+0) - psi(xpsi+2+1));
	gradpsi[1] += ref_dist[0] * ref_dist[2] * (psi(xpsi+2+1+0) - psi(xpsi+2+0));
	gradpsi[2] += ref_dist[0] * ref_dist[1] * (psi(xpsi+2+1+0) - psi(xpsi+1+0));
#elif GRADIENT_ORDER == 2
	for (int i=0; i<3; i++)
	{
		gradpsi[i] = 0.5 * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (psi(xpsi+i) - psi(xpsi-i));
		gradpsi[i] += 0.5 * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (psi(xpsi+i+0) - psi(xpsi-i+0));
		gradpsi[i] += 0.5 * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]) * (psi(xpsi+i+1) - psi(xpsi-i+1));
		gradpsi[i] += 0.5 * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]) * (psi(xpsi+i+1+0) - psi(xpsi-i+1+0));
		gradpsi[i] += 0.5 * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2] * (psi(xpsi+2+i) - psi(xpsi+2-i));
		gradpsi[i] += 0.5 * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2] * (psi(xpsi+2+i+0) - psi(xpsi+2-i+0));
		gradpsi[i] += 0.5 * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2] * (psi(xpsi+2+i+1) - psi(xpsi+2-i+1));
		gradpsi[i] += 0.5 * ref_dist[0] * ref_dist[1] * ref_dist[2] * (psi(xpsi+2+i+1+0) - psi(xpsi+2-i+1+0));
	}
#else
#error GRADIENT_ORDER must be set to 1 or 2
#endif

	if (nfield>=2 && fields[1] != NULL)
	{
		gradpsi[0] -= (1.-ref_dist[1]) * (1.-ref_dist[2]) * (chi(xchi+0) - chi(xchi));
		gradpsi[1] -= (1.-ref_dist[0]) * (1.-ref_dist[2]) * (chi(xchi+1) - chi(xchi));
		gradpsi[2] -= (1.-ref_dist[0]) * (1.-ref_dist[1]) * (chi(xchi+2) - chi(xchi));
		gradpsi[0] -= ref_dist[1] * (1.-ref_dist[2]) * (chi(xchi+1+0) - chi(xchi+1));
		gradpsi[1] -= ref_dist[0] * (1.-ref_dist[2]) * (chi(xchi+1+0) - chi(xchi+0));
		gradpsi[2] -= ref_dist[0] * (1.-ref_dist[1]) * (chi(xchi+2+0) - chi(xchi+0));
		gradpsi[0] -= (1.-ref_dist[1]) * ref_dist[2] * (chi(xchi+2+0) - chi(xchi+2));
		gradpsi[1] -= (1.-ref_dist[0]) * ref_dist[2] * (chi(xchi+2+1) - chi(xchi+2));
		gradpsi[2] -= (1.-ref_dist[0]) * ref_dist[1] * (chi(xchi+2+1) - chi(xchi+1));
		gradpsi[0] -= ref_dist[1] * ref_dist[2] * (chi(xchi+2+1+0) - chi(xchi+2+1));
		gradpsi[1] -= ref_dist[0] * ref_dist[2] * (chi(xchi+2+1+0) - chi(xchi+2+0));
		gradpsi[2] -= ref_dist[0] * ref_dist[1] * (chi(xchi+2+1+0) - chi(xchi+1+0));
	}
	
	Real v2 = 0.;
	for (int i=0;i<3;i++)
	{
		(*part).vel[i] -= dtau * params[0] * gradpsi[i] / dx;
		v2 += (*part).vel[i] * (*part).vel[i];
	}
	
	return v2 / params[0] / params[0];
	
#undef psi
#undef xpsi
#undef chi
#undef xchi
}

// callable struct for update_q_Newton
struct update_q_Newton_functor
{
	__host__ __device__ Real operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		return update_q_Newton(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// update_pos
//////////////////////////
// Description:
//   Update position method (arbitrary momentum)
//   Note that vel[3] in the particle structure is used to store q[3] in units
//   of the particle mass, such that as q^2 << m^2 a^2 the meaning of vel[3]
//   is ~ v*a.
// 
// Arguments:
//   dtau       time step
//   dx         lattice unit  
//   part       pointer to particle structure
//   ref_dist   distance vector to reference point
//   partInfo   global particle properties (unused)
//   fields     array of pointers to fields appearing in geodesic equation
//              fields[0] = phi
//              fields[1] = chi
//              fields[2] = Bi
//   sites      array of sites on the respective lattices
//   nfield     number of fields
//   params     array of additional parameters
//              params[0] = a
//              params[1] = scaling coefficient for Bi
//              params[2] - params[6] = hij_hom[0] - hij_hom[4] for anisotropic expansion
//   outputs    array of reduction variables
//   noutputs   number of reduction variables
//
// Returns:
// 
//////////////////////////

__host__ __device__ void update_pos(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
	Real v[3];
	Real v2 = (*part).vel[0] * (*part).vel[0] + (*part).vel[1] * (*part).vel[1] + (*part).vel[2] * (*part).vel[2];
#ifdef ANISOTROPIC_EXPANSION
	v2 -= (*part).vel[0] * (*part).vel[0] * params[2] + (*part).vel[1] * (*part).vel[1] * params[5] - (*part).vel[2] * (*part).vel[2] * (params[2] + params[5]) + Real(2) * (*part).vel[0] * (*part).vel[1] * params[3] + Real(2) * (*part).vel[0] * (*part).vel[2] * params[4] + Real(2) * (*part).vel[1] * (*part).vel[2] * params[6];
#endif
	Real e2 = v2 + params[0] * params[0];
	Real phi = 0;
	Real chi = 0;
	
	if (nfield >= 1)
	{
		phi = (*fields[0])(sites[0]) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]);
		phi += (*fields[0])(sites[0]+0) * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]);
		phi += (*fields[0])(sites[0]+1) * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]);
		phi += (*fields[0])(sites[0]+0+1) * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]);
		phi += (*fields[0])(sites[0]+2) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2];
		phi += (*fields[0])(sites[0]+0+2) * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2];
		phi += (*fields[0])(sites[0]+1+2) * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2];
		phi += (*fields[0])(sites[0]+0+1+2) * ref_dist[0] * ref_dist[1] * ref_dist[2];
	}
	
	if (nfield >= 2)
	{
		chi = (*fields[1])(sites[1]) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]);
		chi += (*fields[1])(sites[1]+0) * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]);
		chi += (*fields[1])(sites[1]+1) * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]);
		chi += (*fields[1])(sites[1]+0+1) * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]);
		chi += (*fields[1])(sites[1]+2) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2];
		chi += (*fields[1])(sites[1]+0+2) * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2];
		chi += (*fields[1])(sites[1]+1+2) * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2];
		chi += (*fields[1])(sites[1]+0+1+2) * ref_dist[0] * ref_dist[1] * ref_dist[2];
	}
	
	v2 = (1. + (3. - v2 / e2) * phi - chi) / sqrt(e2);

#ifdef ANISOTROPIC_EXPANSION
	v[0] = ((*part).vel[0] * (1. - params[2]) - (*part).vel[1] * params[3] - (*part).vel[2] * params[4]) * v2;
	v[1] = ((*part).vel[1] * (1. - params[5]) - (*part).vel[0] * params[3] - (*part).vel[2] * params[6]) * v2;
	v[2] = ((*part).vel[2] * (1. + params[2] + params[5]) - (*part).vel[0] * params[4] - (*part).vel[1] * params[6]) * v2;
#else	
	v[0] = (*part).vel[0] * v2;
	v[1] = (*part).vel[1] * v2;
	v[2] = (*part).vel[2] * v2;
#endif
	  
	if (nfield >= 3)
	{   
		Real b[3];
		
		b[0] = (*fields[2])(sites[2], 0) * (1.-ref_dist[1]) * (1.-ref_dist[2]);
		b[1] = (*fields[2])(sites[2], 1) * (1.-ref_dist[0]) * (1.-ref_dist[2]);
		b[2] = (*fields[2])(sites[2], 2) * (1.-ref_dist[0]) * (1.-ref_dist[1]);
		b[1] += (*fields[2])(sites[2]+0, 1) * ref_dist[0] * (1.-ref_dist[2]);
		b[2] += (*fields[2])(sites[2]+0, 2) * ref_dist[0] * (1.-ref_dist[1]);
		b[0] += (*fields[2])(sites[2]+1, 0) * ref_dist[1] * (1.-ref_dist[2]);
		b[2] += (*fields[2])(sites[2]+1, 2) * (1.-ref_dist[0]) * ref_dist[1];
		b[0] += (*fields[2])(sites[2]+2, 0) * (1.-ref_dist[1]) * ref_dist[2];
		b[1] += (*fields[2])(sites[2]+2, 1) * (1.-ref_dist[0]) * ref_dist[2];
		b[1] += (*fields[2])(sites[2]+2+0, 1) * ref_dist[0] * ref_dist[2];
		b[0] += (*fields[2])(sites[2]+2+1, 0) * ref_dist[1] * ref_dist[2];
		b[2] += (*fields[2])(sites[2]+1+0, 2) * ref_dist[0] * ref_dist[1];

		for (int l=0;l<3;l++) (*part).pos[l] += dtau*(v[l] + b[l] / params[1]);
	}
	else
	{
		for (int l=0;l<3;l++) (*part).pos[l] += dtau*v[l];
	}   
}

// callable struct for update_pos
struct update_pos_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		update_pos(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// update_pos_Newton
//////////////////////////
// Description:
//   Update position method (Newtonian version)
//   Note that vel[3] in the particle structure is used to store q[3] in units
//   of the particle mass, such that the meaning of vel[3] is v*a.
// 
// Arguments:
//   dtau       time step
//   dx         lattice unit (unused)
//   part       pointer to particle structure
//   ref_dist   distance vector to reference point (unused)
//   partInfo   global particle properties (unused)
//   fields     array of pointers to fields appearing in geodesic equation (unused)
//   sites      array of sites on the respective lattices (unused)
//   nfield     number of fields (unused)
//   params     array of additional parameters
//              params[0] = a
//   outputs    array of reduction variables (unused)
//   noutputs   number of reduction variables (unused)
//
// Returns:
// 
//////////////////////////

__host__ __device__ void update_pos_Newton(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{	  
	for (int l=0;l<3;l++) (*part).pos[l] += dtau * (*part).vel[l] / params[0];   
}

// callable struct for update_pos_Newton
struct update_pos_Newton_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		update_pos_Newton(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// projection_T00_project (1)
//////////////////////////
// Description:
//   Particle-mesh projection for T00, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   T00        pointer to target field
//   a          scale factor at projection (needed in order to convert
//              canonical momenta to energies)
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
// 
//////////////////////////

template<typename part, typename part_info, typename part_dataType>
void projection_T00_project(Particles<part, part_info, part_dataType> * pcls, Field<Real> * T00, double a = 1., Field<Real> * phi = NULL, double coeff = 1.)
{	
	if (T00->lattice().halo() == 0)
	{
		cout<< "projection_T00_project: target field needs halo > 0" << endl;
		exit(-1);
	}
	
	Site xPart(pcls->lattice());
	Site xField(T00->lattice());
	
	Real referPos[3];
	Real weightScalarGridUp[3];
	Real weightScalarGridDown[3];
	Real dx = pcls->res();
	
	double mass = coeff / (dx*dx*dx);
	mass *= *(double*)((char*)pcls->parts_info() + pcls->mass_offset());
	mass /= a;
	
	Real e = a, f = 0., f2 = 0.;
	Real * q;
	size_t offset_q = offsetof(part,vel);
	
	Real localCube[8]; // XYZ = 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111
	Real localCubePhi[8];
	
	for (int i=0; i<8; i++) localCubePhi[i] = 0.0;
	   
	for (xPart.first(),xField.first(); xPart.test(); xPart.next(),xField.next())
	{			  
		if (!pcls->field()(xPart).parts.empty())
		{
			for(int i=0; i<3; i++) referPos[i] = xPart.coord(i)*dx;
			for(int i=0; i<8; i++) localCube[i] = 0.0;
			
			if (phi != NULL)
			{
				localCubePhi[0] = (*phi)(xField);
				localCubePhi[1] = (*phi)(xField+2);
				localCubePhi[2] = (*phi)(xField+1);
				localCubePhi[3] = (*phi)(xField+1+2);
				localCubePhi[4] = (*phi)(xField+0);
				localCubePhi[5] = (*phi)(xField+0+2);
				localCubePhi[6] = (*phi)(xField+0+1);
				localCubePhi[7] = (*phi)(xField+0+1+2);
			}
			
			for (auto it=(pcls->field())(xPart).parts.begin(); it != (pcls->field())(xPart).parts.end(); ++it)
			{
				for (int i=0; i<3; i++)
				{
					weightScalarGridUp[i] = ((*it).pos[i] - referPos[i]) / dx;
					weightScalarGridDown[i] = 1.0l - weightScalarGridUp[i];
				}
				
				if (phi != NULL)
				{
					q = (Real*)((char*)&(*it)+offset_q);
				
					f2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
					e = sqrt(f2 + a * a);
					f = 3. * e + f2 / e;
					f2 = (0.5 * f * f + f2 / (1. + f2 / a / a)) / e;
				}
				
				//000
				localCube[0] += weightScalarGridDown[0]*weightScalarGridDown[1]*weightScalarGridDown[2]*(e+f*localCubePhi[0]+f2*localCubePhi[0]*localCubePhi[0]);
				//001
				localCube[1] += weightScalarGridDown[0]*weightScalarGridDown[1]*weightScalarGridUp[2]*(e+f*localCubePhi[1]+f2*localCubePhi[1]*localCubePhi[1]);
				//010
				localCube[2] += weightScalarGridDown[0]*weightScalarGridUp[1]*weightScalarGridDown[2]*(e+f*localCubePhi[2]+f2*localCubePhi[2]*localCubePhi[2]);
				//011
				localCube[3] += weightScalarGridDown[0]*weightScalarGridUp[1]*weightScalarGridUp[2]*(e+f*localCubePhi[3]+f2*localCubePhi[3]*localCubePhi[3]);
				//100
				localCube[4] += weightScalarGridUp[0]*weightScalarGridDown[1]*weightScalarGridDown[2]*(e+f*localCubePhi[4]+f2*localCubePhi[4]*localCubePhi[4]);
				//101
				localCube[5] += weightScalarGridUp[0]*weightScalarGridDown[1]*weightScalarGridUp[2]*(e+f*localCubePhi[5]+f2*localCubePhi[5]*localCubePhi[5]);
				//110
				localCube[6] += weightScalarGridUp[0]*weightScalarGridUp[1]*weightScalarGridDown[2]*(e+f*localCubePhi[6]+f2*localCubePhi[6]*localCubePhi[6]);
				//111
				localCube[7] += weightScalarGridUp[0]*weightScalarGridUp[1]*weightScalarGridUp[2]*(e+f*localCubePhi[7]+f2*localCubePhi[7]*localCubePhi[7]);
			}
			
			(*T00)(xField)	   += localCube[0] * mass;
			(*T00)(xField+2)	 += localCube[1] * mass;
			(*T00)(xField+1)	 += localCube[2] * mass;
			(*T00)(xField+1+2)   += localCube[3] * mass;
			(*T00)(xField+0)	 += localCube[4] * mass;
			(*T00)(xField+0+2)   += localCube[5] * mass;
			(*T00)(xField+0+1)   += localCube[6] * mass;
			(*T00)(xField+0+1+2) += localCube[7] * mass;
		}
	}  
}

__global__ void projection_comm1_localhalo(Field<Real> * field, long sizeLocalGross[3], int halo)
{
	long k = blockIdx.x + halo - 1;
	int comp = field->components();
	
	for (long j = threadIdx.x + halo - 1; j < sizeLocalGross[1] - halo + 1; j += 128)
	{
		for (int c = 0; c < comp; c++)
		{
			(*field)(static_cast<long>(halo) + sizeLocalGross[0] * (j + sizeLocalGross[1] * k), c) += (*field)(static_cast<long>(-halo) + sizeLocalGross[0] * (1L + j + sizeLocalGross[1] * k), c);
		}		
	}
}

__global__ void projection_comm2_localhalo(Field<Real> * field, long sizeLocalGross[3], int halo)
{
	long k = blockIdx.x + halo - 1;
	
	for (long j = threadIdx.x + halo - 1; j < sizeLocalGross[1] - halo + 1; j += 128)
	{
		(*field)(static_cast<long>(-halo) - 1L + sizeLocalGross[0] * (j + 1L + sizeLocalGross[1] * k), 0, 1) += (*field)(static_cast<long>(halo) - 1L + sizeLocalGross[0] * (j + sizeLocalGross[1] * k), 0, 1);
		(*field)(static_cast<long>(-halo) - 1L + sizeLocalGross[0] * (j + 1L + sizeLocalGross[1] * k), 0, 2) += (*field)(static_cast<long>(halo) - 1L + sizeLocalGross[0] * (j + sizeLocalGross[1] * k), 0, 2);
		(*field)(static_cast<long>(-halo) - 1L + sizeLocalGross[0] * (j + 1L + sizeLocalGross[1] * k), 1, 2) += (*field)(static_cast<long>(halo) - 1L + sizeLocalGross[0] * (j + sizeLocalGross[1] * k), 1, 2);
	}
}

__global__ void projection_comm1_pack_y(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long k = blockIdx.x;
	int comp = field->components();

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < comp; c++)
		{
			buffer[comp * (k * (sizeLocalGross[0] - 2*halo) + i) + c] = (*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo + 1) - halo), c);
		}
	}	
}

__global__ void projection_comm2_pack_y(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer, Real * buffer2)
{
	long k = blockIdx.x;

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < 6; c++)
		{
			buffer[6L * (k * (sizeLocalGross[0] - 2*halo) + i) + c] = (*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo) - halo), c);
		}

		buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i)] = (*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo - 1) + halo - 1), 0, 1);
		buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i) + 1] = (*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo - 1) + halo - 1), 0, 2);
		buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i) + 2] = (*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo - 1) + halo - 1), 1, 2);
	}	
}

__global__ void projection_comm1_unpack_y(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long k = blockIdx.x;
	int comp = field->components();

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < comp; c++)
		{
			(*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo) + halo), c) += buffer[comp * (k * (sizeLocalGross[0] - 2*halo) + i) + c];
		}
	}
}

__global__ void projection_comm2_unpack_y(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer, Real * buffer2)
{
	long k = blockIdx.x;

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < 6; c++)
		{
			(*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo - 1) + halo), c) += buffer[6L * (k * (sizeLocalGross[0] - 2*halo) + i) + c];
		}

		(*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo) - halo - 1), 0, 1) += buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i)];
		(*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo) - halo - 1), 0, 2) += buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i) + 1];
		(*field)(halo + i + sizeLocalGross[0] * (sizeLocalGross[1] * (k + halo) - halo - 1), 1, 2) += buffer2[3L * (k * (sizeLocalGross[0] - 2*halo) + i) + 2];
	}
}

__global__ void projection_comm1_pack_z(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long j = blockIdx.x;
	int comp = field->components();

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < comp; c++)
		{
			buffer[comp * (j * (sizeLocalGross[0] - 2*halo) + i) + c] = (*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (sizeLocalGross[2] - halo)), c);
		}
	}
}

__global__ void projection_comm2_pack_z(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long j = blockIdx.x;

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i)] = (*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (halo - 1)), 0, 1);
		buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i) + 1] = (*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (halo - 1)), 0, 2);
		buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i) + 2] = (*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (halo - 1)), 1, 2);
	}
}

__global__ void projection_comm1_unpack_z(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long j = blockIdx.x;
	int comp = field->components();

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		for (int c = 0; c < comp; c++)
		{
			(*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * halo), c) += buffer[comp * (j * (sizeLocalGross[0] - 2*halo) + i) + c];
		}
	}
}

__global__ void projection_comm2_unpack_z(Field<Real> * field, long sizeLocalGross[3], int halo, Real * buffer)
{
	long j = blockIdx.x;

	for (long i = threadIdx.x; i < sizeLocalGross[0] - 2*halo; i += 128)
	{
		
		(*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (sizeLocalGross[2] - halo - 1)), 0, 1) += buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i)];
		(*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (sizeLocalGross[2] - halo - 1)), 0, 2) += buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i) + 1];
		(*field)(halo + i + sizeLocalGross[0] * (j + halo + sizeLocalGross[1] * (sizeLocalGross[2] - halo - 1)), 1, 2) += buffer[3L * (j * (sizeLocalGross[0] - 2*halo) + i) + 2];
	}
}

void projection_comm1(Field<Real> * field)
{
	int sizeLocal[3] = {field->lattice().sizeLocal(0), field->lattice().sizeLocal(1), field->lattice().sizeLocal(2)};
	int halo = field->lattice().halo();
	long sizeLocalGross[3] = {sizeLocal[0]+2*halo, sizeLocal[1]+2*halo, sizeLocal[2]+2*halo};
	
	long buffer_size_y = static_cast<long>(sizeLocal[2]+1) * static_cast<long>(sizeLocal[0]) * static_cast<long>(field->components());
	long buffer_size_z = static_cast<long>(sizeLocal[1]) * static_cast<long>(sizeLocal[0]) * static_cast<long>(field->components());
	long buffer_size = buffer_size_y>buffer_size_z ? buffer_size_y : buffer_size_z;

	Real * buffer = (Real*) malloc(2*sizeof(Real)*buffer_size);
	Real * rec_buffer = buffer + buffer_size;

	projection_comm1_localhalo<<<sizeLocal[2]+2, 128>>>(field, sizeLocalGross, halo);

	auto success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm1_localhalo: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm1_localhalo");
	}

	projection_comm1_pack_y<<<sizeLocal[2]+1, 128>>>(field, sizeLocalGross, halo, buffer);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm1_pack_y: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm1_pack_y");
	}

	parallel.sendUp_dim1(buffer, rec_buffer, buffer_size_y);

	projection_comm1_unpack_y<<<sizeLocal[2]+1, 128>>>(field, sizeLocalGross, halo, rec_buffer);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm1_unpack_y: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm1_unpack_y");
	}

	projection_comm1_pack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, buffer);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm1_pack_z: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm1_pack_z");
	}

	parallel.sendUp_dim0(buffer, rec_buffer, buffer_size_z);

	projection_comm1_unpack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, rec_buffer);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm1_unpack_z: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm1_unpack_z");
	}

	free(buffer);
}

void projection_Tij_comm2(Field<Real> * field)
{
	int sizeLocal[3] = {field->lattice().sizeLocal(0), field->lattice().sizeLocal(1), field->lattice().sizeLocal(2)};
	int halo = field->lattice().halo();
	long sizeLocalGross[3] = {sizeLocal[0]+2*halo, sizeLocal[1]+2*halo, sizeLocal[2]+2*halo};
	
	long buffer_size_y = static_cast<long>(sizeLocal[2]+2) * static_cast<long>(sizeLocal[0]) * 6L;
	long buffer_size_z = static_cast<long>(sizeLocal[1]) * static_cast<long>(sizeLocal[0]) * 6L;
	long buffer_size = buffer_size_y>buffer_size_z ? buffer_size_y : buffer_size_z;

	Real * buffer = (Real*) malloc(3*sizeof(Real)*buffer_size);
	Real * rec_buffer = buffer + buffer_size;
	Real * buffer2 = rec_buffer + buffer_size;
	Real * rec_buffer2 = buffer2 + buffer_size/2;

	projection_comm1_localhalo<<<sizeLocal[2]+2, 128>>>(field, sizeLocalGross, halo);

	projection_comm2_localhalo<<<sizeLocal[2]+2, 128>>>(field, sizeLocalGross, halo);

	auto success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm[1/2]_localhalo: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm[1/2]_localhalo");
	}

	projection_comm2_pack_y<<<sizeLocal[2]+2, 128>>>(field, sizeLocalGross, halo, buffer, buffer2);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm2_pack_y: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm2_pack_y");
	}

	parallel.sendUpDown_dim1(buffer, rec_buffer, buffer_size_y, buffer2, rec_buffer2, buffer_size_y/2);

	projection_comm2_unpack_y<<<sizeLocal[2]+2, 128>>>(field, sizeLocalGross, halo, rec_buffer, rec_buffer2);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm2_unpack_y: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm2_unpack_y");
	}

	projection_comm1_pack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, buffer);

	projection_comm2_pack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, buffer2);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm[1/2]_pack_z: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm[1/2]_pack_z");
	}

	parallel.sendUpDown_dim0(buffer, rec_buffer, buffer_size_z, buffer2, rec_buffer2, buffer_size_z/2);

	projection_comm1_unpack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, rec_buffer);

	projection_comm2_unpack_z<<<sizeLocal[1], 128>>>(field, sizeLocalGross, halo, rec_buffer2);

	success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		std::cerr << "Error in projection_comm[1/2]_unpack_z: " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("Error in projection_comm[1/2]_unpack_z");
	}

	free(buffer);
}

#define projection_T00_comm projection_comm1


__host__ __device__ void particle_T00_project(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{	
	Real mass = partInfo.mass * params[1] / (dx*dx*dx);

	if (nfield > 1)
	{
		Real f2 = (*part).vel[0] * (*part).vel[0] + (*part).vel[1] * (*part).vel[1] + (*part).vel[2] * (*part).vel[2];
		Real e = sqrt(f2 + params[0] * params[0]);
		Real f = Real(3) * e + f2 / e;
		f2 = (Real(0.5) * f * f + f2 / (Real(1) + f2 / params[0] / params[0])) / e;
		mass /= params[0];

		Real phi[8];

		phi[0] = (*fields[1])(sites[1]);
		phi[1] = (*fields[1])(sites[1]+2);
		phi[2] = (*fields[1])(sites[1]+1);
		phi[3] = (*fields[1])(sites[1]+1+2);
		phi[4] = (*fields[1])(sites[1]+0);
		phi[5] = (*fields[1])(sites[1]+0+2);
		phi[6] = (*fields[1])(sites[1]+0+1);
		phi[7] = (*fields[1])(sites[1]+0+1+2);

		#ifdef __CUDA_ARCH__
		atomicAdd(&(*fields[0])(sites[0]), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (e + f * phi[0] + f2 * phi[0] * phi[0]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+2), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * (e + f * phi[1] + f2 * phi[1] * phi[1]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+1), (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (e + f * phi[2] + f2 * phi[2] * phi[2]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+1+2), (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * (e + f * phi[3] + f2 * phi[3] * phi[3]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (e + f * phi[4] + f2 * phi[4] * phi[4]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+2), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * (e + f * phi[5] + f2 * phi[5] * phi[5]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+1), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (e + f * phi[6] + f2 * phi[6] * phi[6]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+1+2), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * (e + f * phi[7] + f2 * phi[7] * phi[7]) * mass);
		#else
		(*fields[0])(sites[0]) += (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (e + f * phi[0] + f2 * phi[0] * phi[0]) * mass;
		(*fields[0])(sites[0]+2) += (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2] * (e + f * phi[1] + f2 * phi[1] * phi[1]) * mass;
		(*fields[0])(sites[0]+1) += (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]) * (e + f * phi[2] + f2 * phi[2] * phi[2]) * mass;
		(*fields[0])(sites[0]+1+2) += (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2] * (e + f * phi[3] + f2 * phi[3] * phi[3]) * mass;
		(*fields[0])(sites[0]+0) += ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]) * (e + f * phi[4] + f2 * phi[4] * phi[4]) * mass;
		(*fields[0])(sites[0]+0+2) += ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2] * (e + f * phi[5] + f2 * phi[5] * phi[5]) * mass;
		(*fields[0])(sites[0]+0+1) += ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]) * (e + f * phi[6] + f2 * phi[6] * phi[6]) * mass;
		(*fields[0])(sites[0]+0+1+2) += ref_dist[0] * ref_dist[1] * ref_dist[2] * (e + f * phi[7] + f2 * phi[7] * phi[7]) * mass;
		#endif
	}
	else
	{
		#ifdef __CUDA_ARCH__
		atomicAdd(&(*fields[0])(sites[0]), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass);
		atomicAdd(&(*fields[0])(sites[0]+2), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+1), (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass);
		atomicAdd(&(*fields[0])(sites[0]+1+2), (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+2), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+1), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass);
		atomicAdd(&(*fields[0])(sites[0]+0+1+2), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass);
		#else
		(*fields[0])(sites[0]) += (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]) * mass;
		(*fields[0])(sites[0]+2) += (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2] * mass;
		(*fields[0])(sites[0]+1) += (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]) * mass;
		(*fields[0])(sites[0]+1+2) += (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2] * mass;
		(*fields[0])(sites[0]+0) += ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]) * mass;
		(*fields[0])(sites[0]+0+2) += ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2] * mass;
		(*fields[0])(sites[0]+0+1) += ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]) * mass;
		(*fields[0])(sites[0]+0+1+2) += ref_dist[0] * ref_dist[1] * ref_dist[2] * mass;
		#endif
	}
}

// callable struct for particle_T00_project

struct particle_T00_project_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		particle_T00_project(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// projection_T00_project (2)
//////////////////////////
// Description:
//   Particle-mesh projection for T00, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   T00        pointer to target field
//   a          scale factor at projection (needed in order to convert
//              canonical momenta to energies)
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              nullptr which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
// 
//////////////////////////

void projection_T00_project(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> * T00, double a = 1., Field<Real> * phi = nullptr, double coeff = 1.)
{
	Field<Real> * fields[2];
	fields[0] = T00;
	fields[1] = phi;

	double params[2];
	params[0] = a;
	params[1] = coeff;

	pcls->projectParticles(particle_T00_project_functor(), fields, (phi == nullptr ? 1 : 2), params);
}


void scalarProjectionCIC_project(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> * field)
{
	pcls->meshprojection_project(field);
}


//////////////////////////
// projection_T0i_project (1)
//////////////////////////
// Description:
//   Particle-mesh projection for T0i, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   T0i        pointer to target field
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
// 
//////////////////////////

template<typename part, typename part_info, typename part_dataType>
void projection_T0i_project(Particles<part,part_info,part_dataType> * pcls, Field<Real> * T0i, Field<Real> * phi = NULL, double coeff = 1.)
{
	if (T0i->lattice().halo() == 0)
	{
		cout<< "projection_T0i_project: target field needs halo > 0" << endl;
		exit(-1);
	}
	
	Site xPart(pcls->lattice());
	Site xT0i(T0i->lattice());
    
	Real referPos[3];
	Real weightScalarGridDown[3];
	Real weightScalarGridUp[3];
	Real dx = pcls->res();
	
	double mass = coeff / (dx*dx*dx);
	mass *= *(double*)((char*)pcls->parts_info() + pcls->mass_offset());
    
    Real w;
	Real * q;
	size_t offset_q = offsetof(part,vel);
	
	Real  qi[12];
	Real  localCubePhi[8];
	
	for (int i=0; i<8; i++) localCubePhi[i] = 0;
    
	for(xPart.first(),xT0i.first();xPart.test();xPart.next(),xT0i.next())
	{
		if(!pcls->field()(xPart).parts.empty())
        {
        	for(int i=0; i<3; i++)
        		referPos[i] = xPart.coord(i)*dx;
        		
            for(int i=0; i<12; i++) qi[i]=0.0;
            
			if (phi != NULL)
			{
				localCubePhi[0] = (*phi)(xT0i);
				localCubePhi[1] = (*phi)(xT0i+2);
				localCubePhi[2] = (*phi)(xT0i+1);
				localCubePhi[3] = (*phi)(xT0i+1+2);
				localCubePhi[4] = (*phi)(xT0i+0);
				localCubePhi[5] = (*phi)(xT0i+0+2);
				localCubePhi[6] = (*phi)(xT0i+0+1);
				localCubePhi[7] = (*phi)(xT0i+0+1+2);
			}

			for (auto it=(pcls->field())(xPart).parts.begin(); it != (pcls->field())(xPart).parts.end(); ++it)
			{
				for (int i =0; i<3; i++)
				{
					weightScalarGridUp[i] = ((*it).pos[i] - referPos[i]) / dx;
					weightScalarGridDown[i] = 1.0l - weightScalarGridUp[i];
				}
                
				q = (Real*)((char*)&(*it)+offset_q);
                
				w = mass * q[0];
                
				qi[0] +=  w * weightScalarGridDown[1] * weightScalarGridDown[2];
				qi[1] +=  w * weightScalarGridUp[1]   * weightScalarGridDown[2];
				qi[2] +=  w * weightScalarGridDown[1] * weightScalarGridUp[2];
				qi[3] +=  w * weightScalarGridUp[1]   * weightScalarGridUp[2];
                
				w = mass * q[1];
                
				qi[4] +=  w * weightScalarGridDown[0] * weightScalarGridDown[2];
				qi[5] +=  w * weightScalarGridUp[0]   * weightScalarGridDown[2];
				qi[6] +=  w * weightScalarGridDown[0] * weightScalarGridUp[2];
				qi[7] +=  w * weightScalarGridUp[0]   * weightScalarGridUp[2];
                
                w = mass * q[2];
                
				qi[8] +=  w * weightScalarGridDown[0] * weightScalarGridDown[1];
				qi[9] +=  w * weightScalarGridUp[0]   * weightScalarGridDown[1];
				qi[10]+=  w * weightScalarGridDown[0] * weightScalarGridUp[1];
				qi[11]+=  w * weightScalarGridUp[0]   * weightScalarGridUp[1];
			}
            
			(*T0i)(xT0i,0) += qi[0] * (1. + localCubePhi[0] + localCubePhi[4]);
			(*T0i)(xT0i,1) += qi[4] * (1. + localCubePhi[0] + localCubePhi[2]);
			(*T0i)(xT0i,2) += qi[8] * (1. + localCubePhi[0] + localCubePhi[1]);
            
            (*T0i)(xT0i+0,1) += qi[5] * (1. + localCubePhi[4] + localCubePhi[6]);
            (*T0i)(xT0i+0,2) += qi[9] * (1. + localCubePhi[4] + localCubePhi[5]);
            
            (*T0i)(xT0i+1,0) += qi[1] * (1. + localCubePhi[2] + localCubePhi[6]);
            (*T0i)(xT0i+1,2) += qi[10] * (1. + localCubePhi[2] + localCubePhi[3]);
            
            (*T0i)(xT0i+2,0) += qi[2] * (1. + localCubePhi[1] + localCubePhi[5]);
            (*T0i)(xT0i+2,1) += qi[6] * (1. + localCubePhi[1] + localCubePhi[3]);
            
            (*T0i)(xT0i+1+2,0) += qi[3] * (1. + localCubePhi[3] + localCubePhi[7]);
            (*T0i)(xT0i+0+2,1) += qi[7] * (1. + localCubePhi[5] + localCubePhi[7]);
            (*T0i)(xT0i+0+1,2) += qi[11] * (1. + localCubePhi[6] + localCubePhi[7]);
		}
	}
}

#define projection_T0i_comm projection_comm1


__host__ __device__ void particle_T0i_project(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{	
	Real mass = partInfo.mass * (*params) / (dx*dx*dx);

	if (nfield > 1)
	{
		Real phi[8];

		phi[0] = (*fields[1])(sites[1]);
		phi[1] = (*fields[1])(sites[1]+2);
		phi[2] = (*fields[1])(sites[1]+1);
		phi[3] = (*fields[1])(sites[1]+1+2);
		phi[4] = (*fields[1])(sites[1]+0);
		phi[5] = (*fields[1])(sites[1]+0+2);
		phi[6] = (*fields[1])(sites[1]+0+1);
		phi[7] = (*fields[1])(sites[1]+0+1+2);

		#ifdef __CUDA_ARCH__
		atomicAdd(&(*fields[0])(sites[0],0), (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[0] + phi[4]) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0],1), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[0] + phi[2]) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0],2), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[0] + phi[1]) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+0,1), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[4] + phi[6]) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0]+0,2), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[4] + phi[5]) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+1,0), static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[2] + phi[6]) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+1,2), static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1) + phi[2] + phi[3]) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+2,0), static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[1] + phi[5]) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+2,1), static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1) + phi[1] + phi[3]) * mass * (*part).vel[1]);

		atomicAdd(&(*fields[0])(sites[0]+1+2,0), static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * (Real(1) + phi[3] + phi[7]) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+0+2,1), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[2]) * (Real(1) + phi[5] + phi[7]) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0]+0+1,2), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1) + phi[6] + phi[7]) * mass * (*part).vel[2]);
		#else
		(*fields[0])(sites[0],0) += (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[0] + phi[4]) * mass * (*part).vel[0];
		(*fields[0])(sites[0],1) += (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[0] + phi[2]) * mass * (*part).vel[1];
		(*fields[0])(sites[0],2) += (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[0] + phi[1]) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+0,1) += static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[4] + phi[6]) * mass * (*part).vel[1];
		(*fields[0])(sites[0]+0,2) += static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[4] + phi[5]) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+1,0) += static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * (Real(1) + phi[2] + phi[6]) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+1,2) += static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1) + phi[2] + phi[3]) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+2,0) += static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1) + phi[1] + phi[5]) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+2,1) += static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1) + phi[1] + phi[3]) * mass * (*part).vel[1];

		(*fields[0])(sites[0]+1+2,0) += static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * (Real(1) + phi[3] + phi[7]) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+0+2,1) += static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[2]) * (Real(1) + phi[5] + phi[7]) * mass * (*part).vel[1];
		(*fields[0])(sites[0]+0+1,2) += static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1) + phi[6] + phi[7]) * mass * (*part).vel[2];
		#endif
	}
	else
	{
		#ifdef __CUDA_ARCH__
		atomicAdd(&(*fields[0])(sites[0],0), (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0],1), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0],2), (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+0,1), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0]+0,2), static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+1,0), static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+1,2), static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[0])) * mass * (*part).vel[2]);

		atomicAdd(&(*fields[0])(sites[0]+2,0), static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+2,1), static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[0])) * mass * (*part).vel[1]);

		atomicAdd(&(*fields[0])(sites[0]+1+2,0), static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass * (*part).vel[0]);
		atomicAdd(&(*fields[0])(sites[0]+0+2,1), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[2]) * mass * (*part).vel[1]);
		atomicAdd(&(*fields[0])(sites[0]+0+1,2), static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * mass * (*part).vel[2]);
		#else
		(*fields[0])(sites[0],0) += (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[0];
		(*fields[0])(sites[0],1) += (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[1];
		(*fields[0])(sites[0],2) += (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+0,1) += static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[1];
		(*fields[0])(sites[0]+0,2) += static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+1,0) += static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+1,2) += static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[0])) * mass * (*part).vel[2];

		(*fields[0])(sites[0]+2,0) += static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[1])) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+2,1) += static_cast<Real>(ref_dist[2]) * (Real(1)-static_cast<Real>(ref_dist[0])) * mass * (*part).vel[1];

		(*fields[0])(sites[0]+1+2,0) += static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass * (*part).vel[0];
		(*fields[0])(sites[0]+0+2,1) += static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[2]) * mass * (*part).vel[1];
		(*fields[0])(sites[0]+0+1,2) += static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * mass * (*part).vel[2];
		#endif
	}
}

// callable struct for particle_T0i_project

struct particle_T0i_project_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		particle_T0i_project(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// projection_T0i_project (2)
//////////////////////////
// Description:
//   Particle-mesh projection for T0i, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   T0i        pointer to target field
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
// 
//////////////////////////

void projection_T0i_project(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> * T0i, Field<Real> * phi = nullptr, double coeff = 1.)
{
	Field<Real> * fields[2];
	fields[0] = T0i;
	fields[1] = phi;

	pcls->projectParticles(particle_T0i_project_functor(), fields, (phi == nullptr ? 1 : 2), &coeff);
}

void projection_T0i_project_Async(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> ** fields, int nfield, double * params)
{
	pcls->projectParticles_Async(particle_T0i_project_functor(), fields, nfield, params);
}


//////////////////////////
// projection_Tij_project (1)
//////////////////////////
// Description:
//   Particle-mesh projection for Tij, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   Tij        pointer to target field
//   a          scale factor at projection (needed in order to convert
//              canonical momenta to energies)
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//   hij_hom    pointer to the homogeneous part of the spatial metric (default NULL)
//
// Returns:
// 
//////////////////////////

template<typename part, typename part_info, typename part_dataType>
void projection_Tij_project(Particles<part, part_info, part_dataType> * pcls, Field<Real> * Tij, double a = 1., Field<Real> * phi = NULL, double coeff = 1., Real * hij_hom = nullptr)
{	
	if (Tij->lattice().halo() == 0)
	{
		cout<< "projection_Tij_project: target field needs halo > 0" << endl;
		exit(-1);
	}
	
	Site xPart(pcls->lattice());
	Site xTij(Tij->lattice());
	
	Real referPos[3];
	Real weightScalarGridDown[3];
	Real weightScalarGridUp[3];
	Real dx = pcls->res();
	
	double mass = coeff / (dx*dx*dx);
	mass *= *(double*)((char*)pcls->parts_info() + pcls->mass_offset());
	mass /= a;
	
	Real e, f, w;
	Real * q;
	size_t offset_q = offsetof(part,vel);

#ifdef CIC_PROJECT_TIJ
	Real  tij[54];          // local cube
	Real  weightTensorGrid01[9];
	Real  weightTensorGrid02[9];
	Real  weightTensorGrid12[9];
#else	
	Real  tij[6];           // local cube
#endif
	Real  tii[24];          // local cube
	Real  localCubePhi[8];
	
	for (int i=0; i<8; i++) localCubePhi[i] = 0;

	for (xPart.first(),xTij.first(); xPart.test(); xPart.next(),xTij.next())
	{
		if (!pcls->field()(xPart).parts.empty())
		{
			for (int i=0; i<3; i++)
				referPos[i] = (double)xPart.coord(i)*dx;
			
#ifdef CIC_PROJECT_TIJ
			for (int i=0; i<54; i++) tij[i]=0.0;
#else
			for (int i=0; i<6; i++)  tij[i]=0.0;
#endif
			for (int i=0; i<24; i++) tii[i]=0.0;
			
			if (phi != NULL)
			{
				localCubePhi[0] = (*phi)(xTij);
				localCubePhi[1] = (*phi)(xTij+2);
				localCubePhi[2] = (*phi)(xTij+1);
				localCubePhi[3] = (*phi)(xTij+1+2);
				localCubePhi[4] = (*phi)(xTij+0);
				localCubePhi[5] = (*phi)(xTij+0+2);
				localCubePhi[6] = (*phi)(xTij+0+1);
				localCubePhi[7] = (*phi)(xTij+0+1+2);
			}
			
			for (auto it=(pcls->field())(xPart).parts.begin(); it != (pcls->field())(xPart).parts.end(); ++it)
			{
				for (int i=0; i<3; i++)
				{
					weightScalarGridUp[i] = ((*it).pos[i] - referPos[i]) / dx;
					weightScalarGridDown[i] = 1.0l - weightScalarGridUp[i];
				}

#ifdef CIC_PROJECT_TIJ
				if (weightScalarGridDown[0] < 0.5)
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid01[j*3] = 0.;
						weightTensorGrid01[j*3+1] = weightScalarGridDown[0] + 0.5;
						weightTensorGrid01[j*3+2] = 0.5 - weightScalarGridDown[0];

						weightTensorGrid02[j*3] = 0.;
						weightTensorGrid02[j*3+1] = weightScalarGridDown[0] + 0.5;
						weightTensorGrid02[j*3+2] = 0.5 - weightScalarGridDown[0];
					}
				}
				else
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid01[j*3] = weightScalarGridDown[0] - 0.5;
						weightTensorGrid01[j*3+1] = 1.5 - weightScalarGridDown[0];
						weightTensorGrid01[j*3+2] = 0.;

						weightTensorGrid02[j*3] = weightScalarGridDown[0] - 0.5;
						weightTensorGrid02[j*3+1] = 1.5 - weightScalarGridDown[0];
						weightTensorGrid02[j*3+2] = 0.;
					}
				}

				if (weightScalarGridDown[1] < 0.5)
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid01[j] = 0.;
						weightTensorGrid01[j+3] *= weightScalarGridDown[1] + 0.5;
						weightTensorGrid01[j+6] *= 0.5 - weightScalarGridDown[1];

						weightTensorGrid12[j*3] = 0.;
						weightTensorGrid12[j*3+1] = weightScalarGridDown[1] + 0.5;
						weightTensorGrid12[j*3+2] = 0.5 - weightScalarGridDown[1];
					}
				}
				else
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid01[j] *= weightScalarGridDown[1] - 0.5;
						weightTensorGrid01[j+3] *= 1.5 - weightScalarGridDown[1];
						weightTensorGrid01[j+6] = 0.;

						weightTensorGrid12[j*3] = weightScalarGridDown[1] - 0.5;
						weightTensorGrid12[j*3+1] = 1.5 - weightScalarGridDown[1];
						weightTensorGrid12[j*3+2] = 0.;
					}
				}

				if (weightScalarGridDown[2] < 0.5)
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid02[j] = 0.;
						weightTensorGrid02[j+3] *= weightScalarGridDown[2] + 0.5;
						weightTensorGrid02[j+6] *= 0.5 - weightScalarGridDown[2];

						weightTensorGrid12[j] = 0.;
						weightTensorGrid12[j+3] *= weightScalarGridDown[2] + 0.5;
						weightTensorGrid12[j+6] *= 0.5 - weightScalarGridDown[2];
					}
				}
				else
				{
					for (int j=0; j<3; j++)
					{
						weightTensorGrid02[j] *= weightScalarGridDown[2] - 0.5;
						weightTensorGrid02[j+3] *= 1.5 - weightScalarGridDown[2];
						weightTensorGrid02[j+6] = 0.;

						weightTensorGrid12[j] *= weightScalarGridDown[2] - 0.5;
						weightTensorGrid12[j+3] *= 1.5 - weightScalarGridDown[2];
						weightTensorGrid12[j+6] = 0.;
					}
				}
#endif
								
				q = (Real*)((char*)&(*it)+offset_q);
				f = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
#ifdef ANISOTROPIC_EXPANSION
				if (hij_hom != nullptr)
				{
					f -= q[0] * q[0] * hij_hom[0] + q[1] * q[1] * hij_hom[3] - q[2] * q[2] * (hij_hom[0] + hij_hom[3]) + Real(2) * q[0] * q[1] * hij_hom[1] + Real(2) * q[0] * q[2] * hij_hom[2] + Real(2) * q[1] * q[2] * hij_hom[4];
				}
#endif
				e = sqrt(f + a * a);
				f = 4. + a * a / (f + a * a);
				f *= weightScalarGridDown[0] * (weightScalarGridDown[1] * (weightScalarGridDown[2] * localCubePhi[0] + weightScalarGridUp[2] * localCubePhi[1]) + weightScalarGridUp[1] * (weightScalarGridDown[2] * localCubePhi[2] + weightScalarGridUp[2] * localCubePhi[3])) + weightScalarGridUp[0] * (weightScalarGridDown[1] * (weightScalarGridDown[2] * localCubePhi[4] + weightScalarGridUp[2] * localCubePhi[5]) + weightScalarGridUp[1] * (weightScalarGridDown[2] * localCubePhi[6] + weightScalarGridUp[2] * localCubePhi[7]));
				f += 1.;
								
				// diagonal components				
				for (int i=0; i<3; i++)
				{
					w = mass * q[i] * q[i] / e;
					//000
					tii[0+i*8] += w * weightScalarGridDown[0] * weightScalarGridDown[1] * weightScalarGridDown[2] * f;
					//001
					tii[1+i*8] += w * weightScalarGridDown[0] * weightScalarGridDown[1] * weightScalarGridUp[2]   * f; 
					//010
					tii[2+i*8] += w * weightScalarGridDown[0] * weightScalarGridUp[1]   * weightScalarGridDown[2] * f;
					//011
					tii[3+i*8] += w * weightScalarGridDown[0] * weightScalarGridUp[1]   * weightScalarGridUp[2]   * f;
					//100
					tii[4+i*8] += w * weightScalarGridUp[0]   * weightScalarGridDown[1] * weightScalarGridDown[2] * f;
					//101
					tii[5+i*8] += w * weightScalarGridUp[0]   * weightScalarGridDown[1] * weightScalarGridUp[2]   * f;
					//110
					tii[6+i*8] += w * weightScalarGridUp[0]   * weightScalarGridUp[1]   * weightScalarGridDown[2] * f;
					//111
					tii[7+i*8] += w * weightScalarGridUp[0]   * weightScalarGridUp[1]   * weightScalarGridUp[2]   * f;
				}
				
				w = mass * q[0] * q[1] / e;
#ifdef CIC_PROJECT_TIJ
				for (int i = 0; i < 9; i++)
				{
					tij[i] += w * weightScalarGridDown[2] * weightTensorGrid01[i] * f;
					tij[i+9] += w * weightScalarGridUp[2] * weightTensorGrid01[i] * f;
				}
#else
				tij[0] +=  w * weightScalarGridDown[2] * f;
				tij[1] +=  w * weightScalarGridUp[2] * f;
#endif
				
				w = mass * q[0] * q[2] / e;
#ifdef CIC_PROJECT_TIJ
				for (int i = 0; i < 9; i++)
				{
					tij[i+18] += w * weightScalarGridDown[1] * weightTensorGrid02[i] * f;
					tij[i+27] += w * weightScalarGridUp[1] * weightTensorGrid02[i] * f;
				}
#else
				tij[2] +=  w * weightScalarGridDown[1] * f;
				tij[3] +=  w * weightScalarGridUp[1] * f;
#endif
				
				w = mass * q[1] * q[2] / e;
#ifdef CIC_PROJECT_TIJ
				for (int i = 0; i < 9; i++)
				{
					tij[i+36] += w * weightScalarGridDown[0] * weightTensorGrid12[i] * f;
					tij[i+45] += w * weightScalarGridUp[0] * weightTensorGrid12[i] * f;
				}
#else
				tij[4] +=  w * weightScalarGridDown[0] * f;
				tij[5] +=  w * weightScalarGridUp[0] * f;
#endif		
			}
			
			
			for (int i=0; i<3; i++) (*Tij)(xTij,i,i) += tii[8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+0,i,i) += tii[4+8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+1,i,i) += tii[2+8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+2,i,i) += tii[1+8*i];	
			for (int i=0; i<3; i++) (*Tij)(xTij+0+1,i,i) += tii[6+8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+0+2,i,i) += tii[5+8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+1+2,i,i) += tii[3+8*i];
			for (int i=0; i<3; i++) (*Tij)(xTij+0+1+2,i,i) += tii[7+8*i];

#ifdef CIC_PROJECT_TIJ
			(*Tij)(xTij,0,1) += tij[4];
			(*Tij)(xTij,0,2) += tij[22];
			(*Tij)(xTij,1,2) += tij[40];
			(*Tij)(xTij-0-1,0,1) += tij[0];
			(*Tij)(xTij-0-1+2,0,1) += tij[9];
			(*Tij)(xTij-0+1,0,1) += tij[6];
			(*Tij)(xTij-0+1,0,2) += tij[30];
			(*Tij)(xTij-0,0,1) += tij[3];
			(*Tij)(xTij-0,0,2) += tij[21];
			(*Tij)(xTij-0-2,0,2) += tij[18];
			(*Tij)(xTij-0+1-2,0,2) += tij[27];
			(*Tij)(xTij-0+2,0,1) += tij[12];
			(*Tij)(xTij-0+2,0,2) += tij[24];
			(*Tij)(xTij-1,0,1) += tij[1];
			(*Tij)(xTij-1,1,2) += tij[39];
			(*Tij)(xTij-1-2,1,2) += tij[36];
			(*Tij)(xTij+0-1-2,1,2) += tij[45];
			(*Tij)(xTij+0-1,0,1) += tij[2];
			(*Tij)(xTij+0-1,1,2) += tij[48];
			(*Tij)(xTij-1+2,0,1) += tij[10];
			(*Tij)(xTij-1+2,1,2) += tij[42];
			(*Tij)(xTij-2,0,2) += tij[19];
			(*Tij)(xTij-2,1,2) += tij[37];
			(*Tij)(xTij+0-2,0,2) += tij[20];
			(*Tij)(xTij+0-2,1,2) += tij[46];
			(*Tij)(xTij+1-2,0,2) += tij[28];
			(*Tij)(xTij+1-2,1,2) += tij[38];
			(*Tij)(xTij+0+1-2,0,2) += tij[29];
			(*Tij)(xTij+0+1-2,1,2) += tij[47];
			(*Tij)(xTij+0-1+2,0,1) += tij[11];
			(*Tij)(xTij+0-1+2,1,2) += tij[51];
			(*Tij)(xTij-0+1+2,0,1) += tij[15];
			(*Tij)(xTij-0+1+2,0,2) += tij[33];
			(*Tij)(xTij+0,0,1) += tij[5];
			(*Tij)(xTij+0,0,2) += tij[23];
			(*Tij)(xTij+0,1,2) += tij[49];
			(*Tij)(xTij+1,0,1) += tij[7];
			(*Tij)(xTij+1,0,2) += tij[31];
			(*Tij)(xTij+1,1,2) += tij[41];
			(*Tij)(xTij+0+1,0,1) += tij[8];
			(*Tij)(xTij+0+1,0,2) += tij[32];
			(*Tij)(xTij+0+1,1,2) += tij[50];
			(*Tij)(xTij+2,0,1) += tij[13];
			(*Tij)(xTij+2,0,2) += tij[25];
			(*Tij)(xTij+2,1,2) += tij[43];
			(*Tij)(xTij+0+2,0,1) += tij[14];
			(*Tij)(xTij+0+2,0,2) += tij[26];
			(*Tij)(xTij+0+2,1,2) += tij[52];
			(*Tij)(xTij+1+2,0,1) += tij[16];
			(*Tij)(xTij+1+2,0,2) += tij[34];
			(*Tij)(xTij+1+2,1,2) += tij[44];
			(*Tij)(xTij+0+1+2,0,1) += tij[17];
			(*Tij)(xTij+0+1+2,0,2) += tij[35];
			(*Tij)(xTij+0+1+2,1,2) += tij[53];
#else
			(*Tij)(xTij,0,1) += tij[0];
			(*Tij)(xTij,0,2) += tij[2];
			(*Tij)(xTij,1,2) += tij[4];	
			(*Tij)(xTij+0,1,2) += tij[5];
			(*Tij)(xTij+1,0,2) += tij[3];
			(*Tij)(xTij+2,0,1) += tij[1];
#endif
		}
	}  
}

#ifndef projection_Tij_comm
#ifdef CIC_PROJECT_TIJ
#define projection_Tij_comm projection_Tij_comm2
#else
#define projection_Tij_comm projection_comm1
#endif
#endif


__host__ __device__ void particle_Tij_project(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
	Real mass = partInfo.mass * params[1] / (dx*dx*dx) / params[0];
	Real w;

#ifdef CIC_PROJECT_TIJ
	Real  tij[54];
	Real  weightTensorGrid01[9];
	Real  weightTensorGrid02[9];
	Real  weightTensorGrid12[9];

	if (ref_dist[0] > 0.5)
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid01[j*3] = Real(0);
			weightTensorGrid01[j*3+1] = 1.5 - ref_dist[0]; // weightScalarGridDown[0] + 0.5;
			weightTensorGrid01[j*3+2] = ref_dist[0] - 0.5; // 0.5 - weightScalarGridDown[0];

			weightTensorGrid02[j*3] = Real(0);
			weightTensorGrid02[j*3+1] = 1.5 - ref_dist[0]; // weightScalarGridDown[0] + 0.5;
			weightTensorGrid02[j*3+2] = ref_dist[0] - 0.5; // 0.5 - weightScalarGridDown[0];
		}
	}
	else
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid01[j*3] = 0.5 - ref_dist[0]; // weightScalarGridDown[0] - 0.5;
			weightTensorGrid01[j*3+1] = 0.5 + ref_dist[0]; // 1.5 - weightScalarGridDown[0];
			weightTensorGrid01[j*3+2] = Real(0);

			weightTensorGrid02[j*3] = 0.5 - ref_dist[0]; // weightScalarGridDown[0] - 0.5;
			weightTensorGrid02[j*3+1] = 0.5 + ref_dist[0]; // 1.5 - weightScalarGridDown[0];
			weightTensorGrid02[j*3+2] = Real(0);
		}
	}

	if (ref_dist[1] > 0.5)
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid01[j] = Real(0);
			weightTensorGrid01[j+3] *= 1.5 - ref_dist[1]; // weightScalarGridDown[1] + 0.5;
			weightTensorGrid01[j+6] *= ref_dist[1] - 0.5; // 0.5 - weightScalarGridDown[1];

			weightTensorGrid12[j*3] = Real(0);
			weightTensorGrid12[j*3+1] = 1.5 - ref_dist[1]; // weightScalarGridDown[1] + 0.5;
			weightTensorGrid12[j*3+2] = ref_dist[1] - 0.5; // 0.5 - weightScalarGridDown[1];
		}
	}
	else
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid01[j] *= 0.5 - ref_dist[1]; // weightScalarGridDown[1] - 0.5;
			weightTensorGrid01[j+3] *= 0.5 + ref_dist[1]; // 1.5 - weightScalarGridDown[1];
			weightTensorGrid01[j+6] = Real(0);

			weightTensorGrid12[j*3] = 0.5 - ref_dist[1]; // weightScalarGridDown[1] - 0.5;
			weightTensorGrid12[j*3+1] = 0.5 + ref_dist[1]; // 1.5 - weightScalarGridDown[1];
			weightTensorGrid12[j*3+2] = Real(0);
		}
	}

	if (ref_dist[2] > 0.5)
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid02[j] = Real(0);
			weightTensorGrid02[j+3] *= 1.5 - ref_dist[2]; // weightScalarGridDown[2] + 0.5;
			weightTensorGrid02[j+6] *= ref_dist[2] - 0.5; // 0.5 - weightScalarGridDown[2];

			weightTensorGrid12[j] = Real(0);
			weightTensorGrid12[j+3] *= 1.5 - ref_dist[2]; // weightScalarGridDown[2] + 0.5;
			weightTensorGrid12[j+6] *= ref_dist[2] - 0.5; // 0.5 - weightScalarGridDown[2];
		}
	}
	else
	{
		for (int j=0; j<3; j++)
		{
			weightTensorGrid02[j] *= 0.5 - ref_dist[2]; // weightScalarGridDown[2] - 0.5;
			weightTensorGrid02[j+3] *= 0.5 + ref_dist[2]; // 1.5 - weightScalarGridDown[2];
			weightTensorGrid02[j+6] = Real(0);

			weightTensorGrid12[j] *= 0.5 - ref_dist[2]; // weightScalarGridDown[2] - 0.5;
			weightTensorGrid12[j+3] *= 0.5 + ref_dist[2]; // 1.5 - weightScalarGridDown[2];
			weightTensorGrid12[j+6] = Real(0);
		}
	}
#else	
	Real  tij[24];
#endif

	Real f = (*part).vel[0] * (*part).vel[0] + (*part).vel[1] * (*part).vel[1] + (*part).vel[2] * (*part).vel[2];
#ifdef ANISOTROPIC_EXPANSION
	f -= (*part).vel[0] * (*part).vel[0] * static_cast<Real>(params[2]) + (*part).vel[1] * (*part).vel[1] * static_cast<Real>(params[5]) - (*part).vel[2] * (*part).vel[2] * (static_cast<Real>(params[2]) + static_cast<Real>(params[5])) + Real(2) * (*part).vel[0] * (*part).vel[1] * static_cast<Real>(params[3]) + Real(2) * (*part).vel[0] * (*part).vel[2] * static_cast<Real>(params[4]) + Real(2) * (*part).vel[1] * (*part).vel[2] * static_cast<Real>(params[6]);
#endif
	Real e = sqrt(f + static_cast<Real>(params[0]) * static_cast<Real>(params[0]));

	if (nfield > 1)
	{
		f = Real(4) + static_cast<Real>(params[0]) * static_cast<Real>(params[0]) / (f + static_cast<Real>(params[0]) * static_cast<Real>(params[0]));
		f *= (Real(1)-static_cast<Real>(ref_dist[0])) * ((Real(1)-static_cast<Real>(ref_dist[1])) * ((Real(1)-static_cast<Real>(ref_dist[2])) * (*fields[1])(sites[1]) + static_cast<Real>(ref_dist[2]) * (*fields[1])(sites[1]+2)) + static_cast<Real>(ref_dist[1]) * ((Real(1)-static_cast<Real>(ref_dist[2])) * (*fields[1])(sites[1]+1) + static_cast<Real>(ref_dist[2]) * (*fields[1])(sites[1]+1+2))) + static_cast<Real>(ref_dist[0]) * ((Real(1)-static_cast<Real>(ref_dist[1])) * ((Real(1)-static_cast<Real>(ref_dist[2])) * (*fields[1])(sites[1]+0) + static_cast<Real>(ref_dist[2]) * (*fields[1])(sites[1]+0+2)) + static_cast<Real>(ref_dist[1]) * ((Real(1)-static_cast<Real>(ref_dist[2])) * (*fields[1])(sites[1]+0+1) + static_cast<Real>(ref_dist[2]) * (*fields[1])(sites[1]+0+1+2)));
		f += Real(1);
	}
	else
	{
		f = Real(1);
	}

	// diagonal components				
	for (int i=0; i<3; i++)
	{
		w = f * mass * (*part).vel[i] * (*part).vel[i] / e;
		//000
		tij[0+i*8] = w * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2]));
		//001
		tij[1+i*8] = w * (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]); 
		//010
		tij[2+i*8] = w * (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2]));
		//011
		tij[3+i*8] = w * (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]);
		//100
		tij[4+i*8] = w * static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2]));
		//101
		tij[5+i*8] = w * static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]);
		//110
		tij[6+i*8] = w * static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2]));
		//111
		tij[7+i*8] = w * static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]);
	}

	#ifdef __CUDA_ARCH__
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0],i,i), tij[8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+0,i,i), tij[4+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+1,i,i), tij[2+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+2,i,i), tij[1+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+0+1,i,i), tij[6+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+0+2,i,i), tij[5+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+1+2,i,i), tij[3+8*i]);
	for (int i=0; i<3; i++) atomicAdd(&(*fields[0])(sites[0]+0+1+2,i,i), tij[7+8*i]);
	#else
	for (int i=0; i<3; i++) (*fields[0])(sites[0],i,i) += tij[8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+0,i,i) += tij[4+8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+1,i,i) += tij[2+8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+2,i,i) += tij[1+8*i];	
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+0+1,i,i) += tij[6+8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+0+2,i,i) += tij[5+8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+1+2,i,i) += tij[3+8*i];
	for (int i=0; i<3; i++) (*fields[0])(sites[0]+0+1+2,i,i) += tij[7+8*i];
	#endif
	
	w = f * mass * (*part).vel[0] * (*part).vel[1] / e;
#ifdef CIC_PROJECT_TIJ
	for (int i = 0; i < 9; i++)
	{
		tij[i] = w * (Real(1)-static_cast<Real>(ref_dist[2])) * weightTensorGrid01[i];
		tij[i+9] = w * static_cast<Real>(ref_dist[2]) * weightTensorGrid01[i];
	}
#else
	tij[0] =  w * (Real(1)-static_cast<Real>(ref_dist[2]));
	tij[1] =  w * static_cast<Real>(ref_dist[2]);
#endif
	
	w = f * mass * (*part).vel[0] * (*part).vel[2] / e;
#ifdef CIC_PROJECT_TIJ
	for (int i = 0; i < 9; i++)
	{
		tij[i+18] = w * (Real(1)-static_cast<Real>(ref_dist[1])) * weightTensorGrid02[i];
		tij[i+27] = w * static_cast<Real>(ref_dist[1]) * weightTensorGrid02[i];
	}
#else
	tij[2] =  w * (Real(1)-static_cast<Real>(ref_dist[1]));
	tij[3] =  w * static_cast<Real>(ref_dist[1]);
#endif
	
	w = f * mass * (*part).vel[1] * (*part).vel[2] / e;
#ifdef CIC_PROJECT_TIJ
	for (int i = 0; i < 9; i++)
	{
		tij[i+36] = w * (Real(1)-static_cast<Real>(ref_dist[0])) * weightTensorGrid12[i];
		tij[i+45] = w * static_cast<Real>(ref_dist[0]) * weightTensorGrid12[i];
	}
#else
	tij[4] =  w * (Real(1)-static_cast<Real>(ref_dist[0]));
	tij[5] =  w * static_cast<Real>(ref_dist[0]);
#endif

#ifdef __CUDA_ARCH__
#ifdef CIC_PROJECT_TIJ
	atomicAdd(&(*fields[0])(sites[0],0,1), tij[4]);
	atomicAdd(&(*fields[0])(sites[0],0,2), tij[22]);
	atomicAdd(&(*fields[0])(sites[0],1,2), tij[40]);
	if (tij[0] != 0) atomicAdd(&(*fields[0])(sites[0]-0-1,0,1), tij[0]);
	if (tij[9] != 0) atomicAdd(&(*fields[0])(sites[0]-0-1+2,0,1), tij[9]);
	if (tij[6] != 0) atomicAdd(&(*fields[0])(sites[0]-0+1,0,1), tij[6]);
	if (tij[30] != 0) atomicAdd(&(*fields[0])(sites[0]-0+1,0,2), tij[30]);
	if (tij[3] != 0) atomicAdd(&(*fields[0])(sites[0]-0,0,1), tij[3]);
	if (tij[21] != 0) atomicAdd(&(*fields[0])(sites[0]-0,0,2), tij[21]);
	if (tij[18] != 0) atomicAdd(&(*fields[0])(sites[0]-0-2,0,2), tij[18]);
	if (tij[27] != 0) atomicAdd(&(*fields[0])(sites[0]-0+1-2,0,2), tij[27]);
	if (tij[12] != 0) atomicAdd(&(*fields[0])(sites[0]-0+2,0,1), tij[12]);
	if (tij[24] != 0) atomicAdd(&(*fields[0])(sites[0]-0+2,0,2), tij[24]);
	if (tij[1] != 0) atomicAdd(&(*fields[0])(sites[0]-1,0,1), tij[1]);
	if (tij[39] != 0) atomicAdd(&(*fields[0])(sites[0]-1,1,2), tij[39]);
	if (tij[36] != 0) atomicAdd(&(*fields[0])(sites[0]-1-2,1,2), tij[36]);
	if (tij[45] != 0) atomicAdd(&(*fields[0])(sites[0]+0-1-2,1,2), tij[45]);
	if (tij[2] != 0) atomicAdd(&(*fields[0])(sites[0]+0-1,0,1), tij[2]);
	if (tij[48] != 0) atomicAdd(&(*fields[0])(sites[0]+0-1,1,2), tij[48]);
	if (tij[10] != 0) atomicAdd(&(*fields[0])(sites[0]-1+2,0,1), tij[10]);
	if (tij[42] != 0) atomicAdd(&(*fields[0])(sites[0]-1+2,1,2), tij[42]);
	if (tij[19] != 0) atomicAdd(&(*fields[0])(sites[0]-2,0,2), tij[19]);
	if (tij[37] != 0) atomicAdd(&(*fields[0])(sites[0]-2,1,2), tij[37]);
	if (tij[20] != 0) atomicAdd(&(*fields[0])(sites[0]+0-2,0,2), tij[20]);
	if (tij[46] != 0) atomicAdd(&(*fields[0])(sites[0]+0-2,1,2), tij[46]);
	if (tij[28] != 0) atomicAdd(&(*fields[0])(sites[0]+1-2,0,2), tij[28]);
	if (tij[38] != 0) atomicAdd(&(*fields[0])(sites[0]+1-2,1,2), tij[38]);
	if (tij[29] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1-2,0,2), tij[29]);
	if (tij[47] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1-2,1,2), tij[47]);
	if (tij[11] != 0) atomicAdd(&(*fields[0])(sites[0]+0-1+2,0,1), tij[11]);
	if (tij[51] != 0) atomicAdd(&(*fields[0])(sites[0]+0-1+2,1,2), tij[51]);
	if (tij[15] != 0) atomicAdd(&(*fields[0])(sites[0]-0+1+2,0,1), tij[15]);
	if (tij[33] != 0) atomicAdd(&(*fields[0])(sites[0]-0+1+2,0,2), tij[33]);
	if (tij[5] != 0) atomicAdd(&(*fields[0])(sites[0]+0,0,1), tij[5]);
	if (tij[23] != 0) atomicAdd(&(*fields[0])(sites[0]+0,0,2), tij[23]);
	atomicAdd(&(*fields[0])(sites[0]+0,1,2), tij[49]);
	if (tij[7] != 0) atomicAdd(&(*fields[0])(sites[0]+1,0,1), tij[7]);
	atomicAdd(&(*fields[0])(sites[0]+1,0,2), tij[31]);
	if (tij[41] != 0)  atomicAdd(&(*fields[0])(sites[0]+1,1,2), tij[41]);
	if (tij[8] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1,0,1), tij[8]);
	if (tij[32] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1,0,2), tij[32]);
	if (tij[50] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1,1,2), tij[50]);
	atomicAdd(&(*fields[0])(sites[0]+2,0,1), tij[13]);
	if (tij[25] != 0) atomicAdd(&(*fields[0])(sites[0]+2,0,2), tij[25]);
	if (tij[43] != 0) atomicAdd(&(*fields[0])(sites[0]+2,1,2), tij[43]);
	if (tij[14] != 0) atomicAdd(&(*fields[0])(sites[0]+0+2,0,1), tij[14]);
	if (tij[26] != 0) atomicAdd(&(*fields[0])(sites[0]+0+2,0,2), tij[26]);
	if (tij[52] != 0) atomicAdd(&(*fields[0])(sites[0]+0+2,1,2), tij[52]);
	if (tij[16] != 0) atomicAdd(&(*fields[0])(sites[0]+1+2,0,1), tij[16]);
	if (tij[34] != 0) atomicAdd(&(*fields[0])(sites[0]+1+2,0,2), tij[34]);
	if (tij[44] != 0) atomicAdd(&(*fields[0])(sites[0]+1+2,1,2), tij[44]);
	if (tij[17] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1+2,0,1), tij[17]);
	if (tij[35] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1+2,0,2), tij[35]);
	if (tij[53] != 0) atomicAdd(&(*fields[0])(sites[0]+0+1+2,1,2), tij[53]);
#else
	atomicAdd(&(*fields[0])(sites[0],0,1), tij[0]);
	atomicAdd(&(*fields[0])(sites[0],0,2), tij[2]);
	atomicAdd(&(*fields[0])(sites[0],1,2), tij[4]);
	atomicAdd(&(*fields[0])(sites[0]+0,1,2), tij[5]);
	atomicAdd(&(*fields[0])(sites[0]+1,0,2), tij[3]);
	atomicAdd(&(*fields[0])(sites[0]+2,0,1), tij[1]);
#endif
#else
#ifdef CIC_PROJECT_TIJ
	(*fields[0])(sites[0],0,1) += tij[4];
	(*fields[0])(sites[0],0,2) += tij[22];
	(*fields[0])(sites[0],1,2) += tij[40];
	(*fields[0])(sites[0]-0-1,0,1) += tij[0];
	(*fields[0])(sites[0]-0-1+2,0,1) += tij[9];
	(*fields[0])(sites[0]-0+1,0,1) += tij[6];
	(*fields[0])(sites[0]-0+1,0,2) += tij[30];
	(*fields[0])(sites[0]-0,0,1) += tij[3];
	(*fields[0])(sites[0]-0,0,2) += tij[21];
	(*fields[0])(sites[0]-0-2,0,2) += tij[18];
	(*fields[0])(sites[0]-0+1-2,0,2) += tij[27];
	(*fields[0])(sites[0]-0+2,0,1) += tij[12];
	(*fields[0])(sites[0]-0+2,0,2) += tij[24];
	(*fields[0])(sites[0]-1,0,1) += tij[1];
	(*fields[0])(sites[0]-1,1,2) += tij[39];
	(*fields[0])(sites[0]-1-2,1,2) += tij[36];
	(*fields[0])(sites[0]+0-1-2,1,2) += tij[45];
	(*fields[0])(sites[0]+0-1,0,1) += tij[2];
	(*fields[0])(sites[0]+0-1,1,2) += tij[48];
	(*fields[0])(sites[0]-1+2,0,1) += tij[10];
	(*fields[0])(sites[0]-1+2,1,2) += tij[42];
	(*fields[0])(sites[0]-2,0,2) += tij[19];
	(*fields[0])(sites[0]-2,1,2) += tij[37];
	(*fields[0])(sites[0]+0-2,0,2) += tij[20];
	(*fields[0])(sites[0]+0-2,1,2) += tij[46];
	(*fields[0])(sites[0]+1-2,0,2) += tij[28];
	(*fields[0])(sites[0]+1-2,1,2) += tij[38];
	(*fields[0])(sites[0]+0+1-2,0,2) += tij[29];
	(*fields[0])(sites[0]+0+1-2,1,2) += tij[47];
	(*fields[0])(sites[0]+0-1+2,0,1) += tij[11];
	(*fields[0])(sites[0]+0-1+2,1,2) += tij[51];
	(*fields[0])(sites[0]-0+1+2,0,1) += tij[15];
	(*fields[0])(sites[0]-0+1+2,0,2) += tij[33];
	(*fields[0])(sites[0]+0,0,1) += tij[5];
	(*fields[0])(sites[0]+0,0,2) += tij[23];
	(*fields[0])(sites[0]+0,1,2) += tij[49];
	(*fields[0])(sites[0]+1,0,1) += tij[7];
	(*fields[0])(sites[0]+1,0,2) += tij[31];
	(*fields[0])(sites[0]+1,1,2) += tij[41];
	(*fields[0])(sites[0]+0+1,0,1) += tij[8];
	(*fields[0])(sites[0]+0+1,0,2) += tij[32];
	(*fields[0])(sites[0]+0+1,1,2) += tij[50];
	(*fields[0])(sites[0]+2,0,1) += tij[13];
	(*fields[0])(sites[0]+2,0,2) += tij[25];
	(*fields[0])(sites[0]+2,1,2) += tij[43];
	(*fields[0])(sites[0]+0+2,0,1) += tij[14];
	(*fields[0])(sites[0]+0+2,0,2) += tij[26];
	(*fields[0])(sites[0]+0+2,1,2) += tij[52];
	(*fields[0])(sites[0]+1+2,0,1) += tij[16];
	(*fields[0])(sites[0]+1+2,0,2) += tij[34];
	(*fields[0])(sites[0]+1+2,1,2) += tij[44];
	(*fields[0])(sites[0]+0+1+2,0,1) += tij[17];
	(*fields[0])(sites[0]+0+1+2,0,2) += tij[35];
	(*fields[0])(sites[0]+0+1+2,1,2) += tij[53];
#else
	(*fields[0])(sites[0],0,1) += tij[0];
	(*fields[0])(sites[0],0,2) += tij[2];
	(*fields[0])(sites[0],1,2) += tij[4];	
	(*fields[0])(sites[0]+0,1,2) += tij[5];
	(*fields[0])(sites[0]+1,0,2) += tij[3];
	(*fields[0])(sites[0]+2,0,1) += tij[1];
#endif
#endif
}

// callable struct for particle_Tij_project

struct particle_Tij_project_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		particle_Tij_project(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// projection_Tij_project (2)
//////////////////////////
// Description:
//   Particle-mesh projection for Tij, including geometric corrections
// 
// Arguments:
//   pcls       pointer to particle handler
//   Tij        pointer to target field
//   a          scale factor at projection (needed in order to convert
//              canonical momenta to energies)
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//   hij_hom    pointer to the homogeneous part of the spatial metric (default NULL)
//
// Returns:
// 
//////////////////////////

void projection_Tij_project(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> * Tij, double a = 1., Field<Real> * phi = nullptr, double coeff = 1., Real * hij_hom = nullptr)
{
	Field<Real> * fields[2];
	fields[0] = Tij;
	fields[1] = phi;

	double params[7];
	params[0] = a;
	params[1] = coeff;

	if (hij_hom != nullptr)
	{
		params[2] = hij_hom[0];
		params[3] = hij_hom[1];
		params[4] = hij_hom[2];
		params[5] = hij_hom[3];
		params[6] = hij_hom[4];
	}
	else
	{
		params[2] = 0.;
		params[3] = 0.;
		params[4] = 0.;
		params[5] = 0.;
		params[6] = 0.;
	}

	pcls->projectParticles(particle_Tij_project_functor(), fields, (phi == nullptr ? 1 : 2), params);
}

void projection_Tij_project_Async(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> ** fields, int nfield, double * params)
{
	pcls->projectParticles_Async(particle_Tij_project_functor(), fields, nfield, params);
}


//////////////////////////
// projection_Ti0_project (1)
//////////////////////////
// Description:
//   Particle-mesh projection for Ti0, including geometric corrections
//
// Arguments:
//   pcls       pointer to particle handler
//   Ti0        pointer to target field
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   chi        pointer to difference between the Bardeen potentials which
//              characterizes additional corrections; can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
//
//////////////////////////

template<typename part, typename part_info, typename part_dataType>
void projection_Ti0_project(Particles<part, part_info, part_dataType> * pcls, Field<Real> * Ti0, Field<Real> * phi = NULL, Field<Real> * chi = NULL, double coeff = 1.)
{
	if (Ti0->lattice().halo() == 0)
	{
		cout<< "projection_Ti0_project: target field needs halo > 0" << endl;
		exit(-1);
	}

	Site xPart(pcls->lattice());
	Site xField(Ti0->lattice());

	Real referPos[3];
	Real weightScalarGridUp[3];
	Real weightScalarGridDown[3];
	Real dx = pcls->res();

	Real * q;
	size_t offset_q = offsetof(part,vel);


	double mass = coeff / (dx*dx*dx);
	mass *= *(double*)((char*)pcls->parts_info() + pcls->mass_offset());


	Real localCube[24]; // XYZ = 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111
	Real localCubePhi[8];
	Real localCubeChi[8];

	for (int i = 0; i < 8; i++) localCubePhi[i] = 0.0;
	for (int i = 0; i < 8; i++) localCubeChi[i] = 0.0;

	for (xPart.first(), xField.first(); xPart.test(); xPart.next(), xField.next())
	{
		if (!pcls->field()(xPart).parts.empty())
		{
			for(int i = 0; i < 3; i++) referPos[i] = xPart.coord(i)*dx;
			for(int i = 0; i < 24; i++) localCube[i] = 0.0;

			if (phi != NULL)
			{
				localCubePhi[0] = (*phi)(xField);
				localCubePhi[1] = (*phi)(xField+2);
				localCubePhi[2] = (*phi)(xField+1);
				localCubePhi[3] = (*phi)(xField+1+2);
				localCubePhi[4] = (*phi)(xField+0);
				localCubePhi[5] = (*phi)(xField+0+2);
				localCubePhi[6] = (*phi)(xField+0+1);
				localCubePhi[7] = (*phi)(xField+0+1+2);
			}
			if (chi != NULL)
			{
				localCubeChi[0] = (*chi)(xField);
				localCubeChi[1] = (*chi)(xField+2);
				localCubeChi[2] = (*chi)(xField+1);
				localCubeChi[3] = (*chi)(xField+1+2);
				localCubeChi[4] = (*chi)(xField+0);
				localCubeChi[5] = (*chi)(xField+0+2);
				localCubeChi[6] = (*chi)(xField+0+1);
				localCubeChi[7] = (*chi)(xField+0+1+2);
			}

			for (auto it = (pcls->field())(xPart).parts.begin(); it != (pcls->field())(xPart).parts.end(); ++it)
			{
				for (int i = 0; i < 3; i++)
				{
					weightScalarGridUp[i] = ((*it).pos[i] - referPos[i]) / dx;
					weightScalarGridDown[i] = 1.0l - weightScalarGridUp[i];
				}

				q = (Real*)((char*)&(*it)+offset_q);

                for (int i = 0; i < 3; i++){
                    //000
                    localCube[8*i] += weightScalarGridDown[0]*weightScalarGridDown[1]*weightScalarGridDown[2]*q[i]*(1.+6*localCubePhi[0]-localCubeChi[0]);
                    //001
                    localCube[8*i+1] += weightScalarGridDown[0]*weightScalarGridDown[1]*weightScalarGridUp[2]*q[i]*(1.+6*localCubePhi[1]-localCubeChi[1]);
                    //010
                    localCube[8*i+2] += weightScalarGridDown[0]*weightScalarGridUp[1]*weightScalarGridDown[2]*q[i]*(1.+6*localCubePhi[2]-localCubeChi[2]);
                    //011
                    localCube[8*i+3] += weightScalarGridDown[0]*weightScalarGridUp[1]*weightScalarGridUp[2]*q[i]*(1.+6*localCubePhi[3]-localCubeChi[3]);
                    //100
                    localCube[8*i+4] += weightScalarGridUp[0]*weightScalarGridDown[1]*weightScalarGridDown[2]*q[i]*(1.+6*localCubePhi[4]-localCubeChi[4]);
                    //101
                    localCube[8*i+5] += weightScalarGridUp[0]*weightScalarGridDown[1]*weightScalarGridUp[2]*q[i]*(1.+6*localCubePhi[5]-localCubeChi[5]);
                    //110
                    localCube[8*i+6] += weightScalarGridUp[0]*weightScalarGridUp[1]*weightScalarGridDown[2]*q[i]*(1.+6*localCubePhi[6]-localCubeChi[6]);
                    //111
                    localCube[8*i+7] += weightScalarGridUp[0]*weightScalarGridUp[1]*weightScalarGridUp[2]*q[i]*(1.+6*localCubePhi[7]-localCubeChi[7]);
                }
			}
			for (int i = 0; i < 3; i++)
            {
                (*Ti0)(xField,i)       += localCube[8*i] * mass;
                (*Ti0)(xField+2,i)     += localCube[8*i+1] * mass;
                (*Ti0)(xField+1,i)     += localCube[8*i+2] * mass;
                (*Ti0)(xField+1+2,i)   += localCube[8*i+3] * mass;
                (*Ti0)(xField+0,i)     += localCube[8*i+4] * mass;
                (*Ti0)(xField+0+2,i)   += localCube[8*i+5] * mass;
                (*Ti0)(xField+0+1,i)   += localCube[8*i+6] * mass;
                (*Ti0)(xField+0+1+2,i) += localCube[8*i+7] * mass;
            }
		}
	}
}


__host__ __device__ void particle_Ti0_project(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
{
	Real mass = partInfo.mass * (*params) / (dx*dx*dx);

	Real localCubeWeights[8] = {Real(1), Real(1), Real(1), Real(1), Real(1), Real(1), Real(1), Real(1)};

	if (nfield > 1)
	{
		localCubeWeights[0] += Real(6) * (*fields[1])(sites[1]);
		localCubeWeights[1] += Real(6) * (*fields[1])(sites[1]+2);
		localCubeWeights[2] += Real(6) * (*fields[1])(sites[1]+1);
		localCubeWeights[3] += Real(6) * (*fields[1])(sites[1]+1+2);
		localCubeWeights[4] += Real(6) * (*fields[1])(sites[1]+0);
		localCubeWeights[5] += Real(6) * (*fields[1])(sites[1]+0+2);
		localCubeWeights[6] += Real(6) * (*fields[1])(sites[1]+0+1);
		localCubeWeights[7] += Real(6) * (*fields[1])(sites[1]+0+1+2);
	}

	if (nfield > 2)
	{
		localCubeWeights[0] -= (*fields[2])(sites[2]);
		localCubeWeights[1] -= (*fields[2])(sites[2]+2);
		localCubeWeights[2] -= (*fields[2])(sites[2]+1);
		localCubeWeights[3] -= (*fields[2])(sites[2]+1+2);
		localCubeWeights[4] -= (*fields[2])(sites[2]+0);
		localCubeWeights[5] -= (*fields[2])(sites[2]+0+2);
		localCubeWeights[6] -= (*fields[2])(sites[2]+0+1);
		localCubeWeights[7] -= (*fields[2])(sites[2]+0+1+2);
	}

	localCubeWeights[0] *= (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass;
	localCubeWeights[1] *= (Real(1)-static_cast<Real>(ref_dist[0])) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * mass;
	localCubeWeights[2] *= (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass;
	localCubeWeights[3] *= (Real(1)-static_cast<Real>(ref_dist[0])) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass;
	localCubeWeights[4] *= static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass;
	localCubeWeights[5] *= static_cast<Real>(ref_dist[0]) * (Real(1)-static_cast<Real>(ref_dist[1])) * static_cast<Real>(ref_dist[2]) * mass;
	localCubeWeights[6] *= static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * (Real(1)-static_cast<Real>(ref_dist[2])) * mass;
	localCubeWeights[7] *= static_cast<Real>(ref_dist[0]) * static_cast<Real>(ref_dist[1]) * static_cast<Real>(ref_dist[2]) * mass;

	#ifdef __CUDA_ARCH__
	for (int i = 0; i < 3; i++)
	{
		atomicAdd(&(*fields[0])(sites[0],i)       , localCubeWeights[0] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+2,i)     , localCubeWeights[1] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+1,i)     , localCubeWeights[2] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+1+2,i)   , localCubeWeights[3] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+0,i)     , localCubeWeights[4] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+0+2,i)   , localCubeWeights[5] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+0+1,i)   , localCubeWeights[6] * (*part).vel[i]);
		atomicAdd(&(*fields[0])(sites[0]+0+1+2,i) , localCubeWeights[7] * (*part).vel[i]);
	}
	#else
	for (int i = 0; i < 3; i++)
	{
		(*fields[0])(sites[0],i)       += localCubeWeights[0] * (*part).vel[i];
		(*fields[0])(sites[0]+2,i)     += localCubeWeights[1] * (*part).vel[i];
		(*fields[0])(sites[0]+1,i)     += localCubeWeights[2] * (*part).vel[i];
		(*fields[0])(sites[0]+1+2,i)   += localCubeWeights[3] * (*part).vel[i];
		(*fields[0])(sites[0]+0,i)     += localCubeWeights[4] * (*part).vel[i];
		(*fields[0])(sites[0]+0+2,i)   += localCubeWeights[5] * (*part).vel[i];
		(*fields[0])(sites[0]+0+1,i)   += localCubeWeights[6] * (*part).vel[i];
		(*fields[0])(sites[0]+0+1+2,i) += localCubeWeights[7] * (*part).vel[i];
	}
	#endif
}

// callable struct for particle_Ti0_project

struct particle_Ti0_project_functor
{
	__host__ __device__ void operator()(double dtau, double dx, part_simple * part, double * ref_dist, part_simple_info partInfo, Field<Real> * fields[], Site * sites, int nfield, double * params, double * outputs, int noutputs)
	{
		particle_Ti0_project(dtau, dx, part, ref_dist, partInfo, fields, sites, nfield, params, outputs, noutputs);
	}
};


//////////////////////////
// projection_Ti0_project (2)
//////////////////////////
// Description:
//   Particle-mesh projection for Ti0, including geometric corrections
//
// Arguments:
//   pcls       pointer to particle handler
//   Ti0        pointer to target field
//   phi        pointer to Bardeen potential which characterizes the
//              geometric corrections (volume distortion); can be set to
//              NULL which will result in no corrections applied
//   chi        pointer to difference between the Bardeen potentials which
//              characterizes additional corrections; can be set to
//              NULL which will result in no corrections applied
//   coeff      coefficient applied to the projection operation (default 1)
//
// Returns:
//
//////////////////////////

void projection_Ti0_project(perfParticles<part_simple, part_simple_info> * pcls, Field<Real> * Ti0, Field<Real> * phi = nullptr, Field<Real> * chi = nullptr, double coeff = 1.)
{
	Field<Real> * fields[3];
	fields[0] = Ti0;
	fields[1] = phi;
	fields[2] = chi;

	int nfield = 1;

	if (phi != nullptr)
	{
		nfield++;
		if (chi != nullptr) nfield++;
	}

	pcls->projectParticles(particle_Ti0_project_functor(), fields, nfield, &coeff);
}


//////////////////////////
// projectFTtheta
//////////////////////////
// Description:
//   Compute the diverge of the velocity in Fourier space
//
// Arguments:
//   thFT       reference to the Fourier image of the divergence of the velocity field
//   viFT       reference to the Fourier image of the velocity field
//
// Returns:
//
//////////////////////////

void projectFTtheta(Field<Cplx> & thFT, Field<Cplx> & viFT)
{
	const int linesize = thFT.lattice().size(1);
	int i;
	Real * gridk;
	rKSite k(thFT.lattice());
	Cplx tmp(0., 0.);

	gridk = (Real *) malloc(linesize * sizeof(Real));

	for (i = 0; i < linesize; i++)
		gridk[i] = (Real) linesize * sin(M_PI * 2.0 * (Real) i / (Real) linesize);

	for (k.first(); k.test(); k.next())
		thFT(k) = Cplx(0.,1.)*(gridk[k.coord(0)] * viFT(k,0) + gridk[k.coord(1)] * viFT(k,1) + gridk[k.coord(2)] * viFT(k,2));

	free(gridk);
}


//////////////////////////
// projectFTomega
//////////////////////////
// Description:
//   Compute the curl part of the velocity field in Fourier space
//
// Arguments:
//   viFT      reference to the input Fourier image of the velocity field
//             the divergence part will be projected out
//
// Returns:
//
//////////////////////////

void projectFTomega(Field<Cplx> & viFT)
{
	const int linesize = viFT.lattice().size(1);
	int i;
	Real * gridk2;
	Cplx * kshift;
	Real * gridk;
	rKSite k(viFT.lattice());
	Cplx tmp(0., 0.);
	Cplx vr[3];

	gridk2 = (Real *) malloc(linesize * sizeof(Real));
	gridk = (Real *) malloc(linesize * sizeof(Real));

	for (i = 0; i < linesize; i++)
	{
		gridk[i] = (Real) linesize * sin(M_PI * 2.0 * (Real) i / (Real) linesize);
		gridk2[i] = gridk[i]*gridk[i];
    }

	k.first();
	if (k.coord(0) == 0 && k.coord(1) == 0 && k.coord(2) == 0)
	{
		viFT(k,0) = Cplx(0.,0.);
		viFT(k,1) = Cplx(0.,0.);
		viFT(k,2) = Cplx(0.,0.);
		k.next();
	}
	
	for (; k.test(); k.next())
	{
		if ((k.coord(0) == 0 || k.coord(0) == linesize/2) && (k.coord(1) == 0 || k.coord(1) == linesize/2) && (k.coord(2) == 0 || k.coord(2) == linesize/2))
		{
			viFT(k, 0) = Cplx(0.,0.);
			viFT(k, 1) = Cplx(0.,0.);
			viFT(k, 2) = Cplx(0.,0.);
		}
		else
		{
			tmp = (gridk[k.coord(0)] * viFT(k,0) + gridk[k.coord(1)] * viFT(k,1) + gridk[k.coord(2)] * viFT(k,2)) / (gridk2[k.coord(0)] + gridk2[k.coord(1)] + gridk2[k.coord(2)]);

			vr[0] = (viFT(k,0) - gridk[k.coord(0)] * tmp);
			vr[1] = (viFT(k,1) - gridk[k.coord(1)] * tmp);
			vr[2] = (viFT(k,2) - gridk[k.coord(2)] * tmp);
			
			viFT(k,0) = Cplx(0.,1.)*(gridk[k.coord(1)]*vr[2] - gridk[k.coord(2)]*vr[1]);
			viFT(k,1) = Cplx(0.,1.)*(gridk[k.coord(2)]*vr[0] - gridk[k.coord(0)]*vr[2]);
			viFT(k,2) = Cplx(0.,1.)*(gridk[k.coord(0)]*vr[1] - gridk[k.coord(1)]*vr[0]);
		}
	}

	free(gridk2);
}

#endif

