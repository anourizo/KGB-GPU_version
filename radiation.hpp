//////////////////////////
// radiation.hpp
//////////////////////////
// 
// code components related to radiation and linear relativistic species
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: January 2025
//
//////////////////////////

#ifndef RADIATION_HEADER
#define RADIATION_HEADER

#ifdef HAVE_CLASS

//////////////////////////
// projection_T00_project (radiation module)
//////////////////////////
// Description:
//   provides a realization of the linear density field of radiation and
//   non-cold species using linear transfer functions precomputed with CLASS;
//   the contributions for the various species are included only until some
//   individual redshift values are reached (after which no linear treatment
//   is requested)
// 
// Arguments:
//   class_background  CLASS structure that contains the background
//   class_perturbs    CLASS structure that contains the perturbations
//   source            reference to field that will contain the realization
//   scalarFT          reference to Fourier image of that field
//   plan_source       pointer to FFT planner
//   sim               simulation metadata structure
//   ic                settings for IC generation (contains the random seed)
//   cosmo             cosmological parameter structure
//   fourpiG           4 pi G (in code units)
//   a                 scale factor
//   coeff             multiplicative coefficient (default 1)
//   zetaFT            reference to Fourier image of zeta (default NULL)
//
// Returns:
// 
//////////////////////////

void projection_T00_project(background & class_background, perturbs & class_perturbs, Field<Real> & source, Field<Cplx> & scalarFT, PlanFFT<Cplx> * plan_source, metadata & sim, icsettings & ic, cosmology & cosmo, const double fourpiG, double a, double coeff = 1., Field<Cplx> * zetaFT = NULL)
{
	gsl_spline * tk1 = NULL;
	gsl_spline * tk2 = NULL;
	double * delta = NULL;
	double * k = NULL;
	char ncdm_name[8];
	int n = 0;
	double rescale, Omega_ncdm = 0., Omegaw_ncdm = 0., Omega_rad = 0., Omega_fld = 0., bg_smg = 0.;
	Site x(source.lattice());
	rKSite kFT(scalarFT.lattice());

	if (a < 1. / (sim.z_switch_deltarad + 1.) && cosmo.Omega_g + cosmo.Omega_ur > 0 && sim.radiation_flag == 1)
	{
		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "g", sim.boxsize, (1. / a) - 1., cosmo.h);
		Omega_rad += cosmo.Omega_g;

		n = tk1->size;
		delta = (double *) calloc(n, sizeof(double));
		k = (double *) malloc(n * sizeof(double));

#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			delta[i] = -tk1->y[i] * coeff * cosmo.Omega_g * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i] / a;
			k[i] = tk1->x[i];
		}

		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "ur", sim.boxsize, (1. / a) - 1., cosmo.h);
		Omega_rad += cosmo.Omega_ur;

#pragma omp parallel for
		for (int i = 0; i < n; i++)
				delta[i] -= tk1->y[i] * coeff * cosmo.Omega_ur * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i] / a;

		gsl_spline_free(tk1);
		gsl_spline_free(tk2);
	}

	if (a < 1. && cosmo.Omega_fld > 0 && sim.fluid_flag == 1)
	{
		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "fld", sim.boxsize, (1. / a) - 1., cosmo.h);
		Omega_fld = cosmo.Omega_fld / pow(a, 3. * cosmo.w0_fld);

		if (delta == NULL)
		{
			n = tk1->size;
			delta = (double *) calloc(n, sizeof(double));
			k = (double *) malloc(n * sizeof(double));

#pragma omp parallel for
			for (int i = 0; i < n; i++)
				k[i] = tk1->x[i];
		}

#pragma omp parallel for		
		for (int i = 0; i < n; i++)
			delta[i] -= tk1->y[i] * coeff * Omega_fld * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i];

		gsl_spline_free(tk1);
		gsl_spline_free(tk2);
	}

	for (int p = 0; p < cosmo.num_ncdm; p++)
	{
		if (a < 1. / (sim.z_switch_deltancdm[p] + 1.) && cosmo.Omega_ncdm[p] > 0)
		{
			sprintf(ncdm_name, "ncdm[%d]", p);
			loadTransferFunctions(class_background, class_perturbs, tk1, tk2, ncdm_name, sim.boxsize, (1. / a) - 1., cosmo.h);
			rescale = bg_ncdm(a, cosmo, p);
			Omega_ncdm += rescale;
			Omegaw_ncdm += pressure_ncdm(a, cosmo, p);

			COUT << "ncdm species " << p << " has Omega_ncdm = " << Omega_ncdm << " and effective w = " << Omegaw_ncdm / Omega_ncdm << std::endl;

			if (delta == NULL)
			{
				n = tk1->size;
				delta = (double *) calloc(n, sizeof(double));
				k = (double *) malloc(n * sizeof(double));

#pragma omp parallel for
				for (int i = 0; i < n; i++)
					k[i] = tk1->x[i];
			}

#pragma omp parallel for
			for (int i = 0; i < n; i++)
				delta[i] -= tk1->y[i] * coeff * rescale * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i];

			gsl_spline_free(tk1);
			gsl_spline_free(tk2);
		}
	}

	if (n > 0)
	{
		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (a < 1.) ? (1. / a) - 1. : 0., cosmo.h);

		if (sim.gr_flag == 0) // add gauge correction for N-body gauge: -3 Hconf (1+w) eta' / (a H')
		{
			rescale = 1. / gsl_spline_eval_deriv(cosmo.Hspline, a, cosmo.acc_H) / a / a;

#pragma omp parallel for
			for (int i = 0; i < n; i++)
				delta[i] += coeff * (4. * Omega_rad / a + 3. * Omega_ncdm + 3. * Omegaw_ncdm + 3. * (1. + cosmo.w0_fld) * Omega_fld + 3. * bg_smg) * rescale * M_PI * tk2->y[i] * sqrt(Pk_primordial(tk2->x[i] * cosmo.h / sim.boxsize, ic) / tk2->x[i]) / tk2->x[i];
		}
		else // add gauge correction for Poisson gauge: -3 Hconf (1+w) (3 eta' + 0.5 h') / k^2
		{
			rescale = Hconf(a, fourpiG, cosmo);

#pragma omp parallel for
			for (int i = 0; i < n; i++)
				delta[i] += coeff * (4. * Omega_rad / a + 3. * Omega_ncdm + 3. * Omegaw_ncdm + 3. * (1. + cosmo.w0_fld) * Omega_fld + 3. * bg_smg) * 3. * rescale * M_PI * tk2->y[i] * sqrt(Pk_primordial(tk2->x[i] * cosmo.h / sim.boxsize, ic) / tk2->x[i]) / tk2->x[i] / tk2->x[i] / tk2->x[i];

			gsl_spline_free(tk1);
			gsl_spline_free(tk2);

			loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "h", sim.boxsize, (a < 1.) ? (1. / a) - 1. : 0., cosmo.h);

#pragma omp parallel for
			for (int i = 0; i < n; i++)
				delta[i] += coeff * (4. * Omega_rad / a + 3. * Omega_ncdm + 3. * Omegaw_ncdm + 3. * (1. + cosmo.w0_fld) * Omega_fld + 3. * bg_smg) * 0.5 * rescale * M_PI * tk2->y[i] * sqrt(Pk_primordial(tk2->x[i] * cosmo.h / sim.boxsize, ic) / tk2->x[i]) / tk2->x[i] / tk2->x[i] / tk2->x[i];
		}

		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		tk1 = gsl_spline_alloc(gsl_interp_cspline, n);
		gsl_spline_init(tk1, k, delta, n);		

		if (zetaFT == NULL)
			generateRealization(scalarFT, 0., tk1, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE);
		else
		{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
			for (int i = 0; i < scalarFT.lattice().sizeLocal(1); i++)
			{
				for (int j = 0; j < scalarFT.lattice().sizeLocal(2); j++)
				{
					if (!kFT.setCoord(0, j + scalarFT.lattice().coordSkip()[0], i + scalarFT.lattice().coordSkip()[1]))
					{
						std::cerr << "proc#" << parallel.rank() << ": Error in projection_T00_project! Could not set coordinates at k=(0, " << j + scalarFT.lattice().coordSkip()[0] << ", " << i + scalarFT.lattice().coordSkip()[1] << ")" << std::endl;
						throw std::runtime_error("Error in projection_T00_project: Could not set coordinates.");
					}

					gsl_interp_accel * acc = gsl_interp_accel_alloc();
					for (int z = 0; z < scalarFT.lattice().sizeLocal(0); z++)
					{
						double tmp = kFT.coord(0)*kFT.coord(0);
						if (kFT.coord(1) < (sim.numpts/2) + 1)
							tmp += kFT.coord(1)*kFT.coord(1);
						else
							tmp += (sim.numpts-kFT.coord(1))*(sim.numpts-kFT.coord(1));
						if (kFT.coord(2) < (sim.numpts/2) + 1)
							tmp += kFT.coord(2)*kFT.coord(2);
						else
							tmp += (sim.numpts-kFT.coord(2))*(sim.numpts-kFT.coord(2));
						if (tmp > 0)
						{
							tmp = 2. * M_PI * sqrt(tmp);
							scalarFT(kFT) = (*zetaFT)(kFT) * gsl_spline_eval(tk1, tmp, acc);
						}
						else
							scalarFT(kFT) = Cplx(0.,0.);

						kFT.next();
					}
					gsl_interp_accel_free(acc);
				}
			}
		}

		plan_source->execute(FFT_BACKWARD);

		gsl_spline_free(tk1);
		free(delta);
		free(k);

		Field<Real> * fieldptr = &source;
		double * d_params;

		cudaMalloc(&d_params, sizeof(double));
		cudaMemcpy(d_params, &Omega_ncdm, sizeof(double), cudaMemcpyDefault);

		lattice_for_each<<<dim3(source.lattice().sizeLocal(1), source.lattice().sizeLocal(2)), 128>>>(lattice_add_functor(), sim.numpts, &fieldptr, 1, d_params, nullptr, nullptr);

		cudaDeviceSynchronize();
		cudaFree(d_params);
	}
}


//////////////////////////
// prepareFTchiLinear
//////////////////////////
// Description:
//   provides a (Fourier-space) realization of chi (generated by radiation and
//   non-cold species) from the linear transfer functions precomputed with CLASS
// 
// Arguments:
//   class_background  CLASS structure that contains the background
//   class_perturbs    CLASS structure that contains the perturbations
//   scalarFT          reference to Fourier image of field; will contain the
//                     (Fourier image) of the realization
//   sim               simulation metadata structure
//   ic                settings for IC generation (contains the random seed)
//   cosmo             cosmological parameter structure
//   fourpiG           4 pi G (in code units)
//   a                 scale factor
//   coeff             multiplicative coefficient (default 1)
//   zetaFT            reference to Fourier image of zeta (default NULL)
//
// Returns:
// 
//////////////////////////

void prepareFTchiLinear(background & class_background, perturbs & class_perturbs, Field<Cplx> & scalarFT, metadata & sim, icsettings & ic, cosmology & cosmo, const double fourpiG, double a, double coeff = 1., Field<Cplx> * zetaFT = NULL)
{
	gsl_spline * tk1 = NULL;
	gsl_spline * tk2 = NULL;
	double * chi = NULL;
	rKSite k(scalarFT.lattice());

	loadTransferFunctions(class_background, class_perturbs, tk1, tk2, NULL, sim.boxsize, (1. / a) - 1., cosmo.h);

	chi = (double *) malloc(tk1->size * sizeof(double));

#pragma omp parallel for
	for (int i = 0; i < (int) tk1->size; i++)
		chi[i] = (tk2->y[i] - tk1->y[i]) * coeff * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i];

	gsl_spline_free(tk2);

	if (sim.gr_flag == 0 && a < 0.99) // add gauge correction for N-body gauge
	{
		gsl_spline_free(tk1);

		double rescale = Hconf(a, fourpiG, cosmo);

		double a2dHda[5];

		#pragma unroll
		for (int i = 0; i < 5; i++)
			a2dHda[i] = a * a * gsl_spline_eval_deriv(cosmo.Hspline, a * (0.99 + 0.005 * (double) i), cosmo.acc_H) * (0.99 + 0.005 * (double) i) * (0.99 + 0.005 * (double) i);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (1. / (1.005 * a)) - 1., cosmo.h);

		double * deriv1 = (double *) malloc(tk1->size * sizeof(double));
		double * deriv2 = (double *) malloc(tk1->size * sizeof(double));

#pragma omp parallel for
		for (int i = 0; i < tk1->size; i++)
		{
			deriv1[i] = (1. - (3. * rescale + a2dHda[2]) / a2dHda[3]) * tk2->y[i] / 1.5;
			deriv2[i] = -rescale * tk2->y[i] / a2dHda[3] / 0.75;
		}
			
		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (1. / (0.995 * a)) - 1., cosmo.h);

#pragma omp parallel for
		for (int i = 0; i < tk1->size; i++)
		{
			deriv1[i] -= (1. - (3. * rescale + a2dHda[2]) / a2dHda[1]) * tk2->y[i] / 1.5;
			deriv2[i] -= rescale * tk2->y[i] / a2dHda[1] / 0.75;
		}
			
		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (1. / (1.01 * a)) - 1., cosmo.h);

#pragma omp parallel for
		for (int i = 0; i < tk1->size; i++)
		{
			deriv1[i] -= (1. - (3. * rescale + a2dHda[2]) / a2dHda[4]) * tk2->y[i] / 12.;
			deriv2[i] += rescale * tk2->y[i] / a2dHda[4] / 12.;
		}
			
		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (1. / (0.99 * a)) - 1., cosmo.h);

#pragma omp parallel for
		for (int i = 0; i < tk1->size; i++)
		{
			deriv1[i] += (1. - (3. * rescale + a2dHda[2]) / a2dHda[0]) * tk2->y[i] / 12.;
			deriv2[i] += rescale * tk2->y[i] / a2dHda[0] / 12.;
		}

		gsl_spline_free(tk1);
		gsl_spline_free(tk2);

		loadTransferFunctions(class_background, class_perturbs, tk1, tk2, "eta", sim.boxsize, (1. / a) - 1., cosmo.h);

#pragma omp parallel for
		for (int i = 0; i < tk1->size; i++)
		{
			deriv2[i] += 2.5 * rescale * tk2->y[i] / a2dHda[2];
			chi[i] -= 3. * rescale * (tk2->y[i] + 200. * deriv1[i] + 40000. * deriv2[i]) * M_PI * sqrt(Pk_primordial(tk1->x[i] * cosmo.h / sim.boxsize, ic) / tk1->x[i]) / tk1->x[i] / tk1->x[i] / tk1->x[i];
		}

		free(deriv1);
		free(deriv2);

		gsl_spline_free(tk2);
	}

	tk2 = gsl_spline_alloc(gsl_interp_cspline, tk1->size);
	gsl_spline_init(tk2, tk1->x, chi, tk1->size);

	if (zetaFT == NULL)
		generateRealization(scalarFT, 0., tk2, (unsigned int) ic.seed, ic.flags & ICFLAG_KSPHERE, 0);
	else
	{
#pragma omp parallel for collapse(2) default(shared) firstprivate(k)
		for (int i = 0; i < scalarFT.lattice().sizeLocal(1); i++)
		{
			for (int j = 0; j < scalarFT.lattice().sizeLocal(2); j++)
			{
				if (!k.setCoord(0, j + scalarFT.lattice().coordSkip()[0], i + scalarFT.lattice().coordSkip()[1]))
				{
					std::cerr << "proc#" << parallel.rank() << ": Error in prepareFTchiLinear! Could not set coordinates at k=(0, " << j + scalarFT.lattice().coordSkip()[0] << ", " << i + scalarFT.lattice().coordSkip()[1] << ")" << std::endl;
					throw std::runtime_error("Error in prepareFTchiLinear: Could not set coordinates.");
				}

				gsl_interp_accel * acc = gsl_interp_accel_alloc();
				for (int z = 0; z < scalarFT.lattice().sizeLocal(0); z++)
				{
					double tmp = k.coord(0)*k.coord(0);
					if (k.coord(1) < (sim.numpts/2) + 1)
						tmp += k.coord(1)*k.coord(1);
					else
						tmp += (sim.numpts-k.coord(1))*(sim.numpts-k.coord(1));
					if (k.coord(2) < (sim.numpts/2) + 1)
						tmp += k.coord(2)*k.coord(2);
					else
						tmp += (sim.numpts-k.coord(2))*(sim.numpts-k.coord(2));
					if (tmp > 0)
					{
						tmp = 2. * M_PI * sqrt(tmp);
						scalarFT(k) = (*zetaFT)(k) * gsl_spline_eval(tk2, tmp, acc);
					}
					else
						scalarFT(k) = Cplx(0.,0.);

					k.next();
				}
				gsl_interp_accel_free(acc);
			}
		}
	}

	gsl_spline_free(tk1);
	gsl_spline_free(tk2);
	free(chi);
}
#endif

#endif

