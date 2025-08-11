//////////////////////////
// Particles_gevolution.hpp
//////////////////////////
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: January 2025
//
//////////////////////////

#ifndef PARTICLES_GEVOLUTION_HEADER
#define PARTICLES_GEVOLUTION_HEADER

#include "particles/LATfield2_perfParticles.hpp"

#ifndef PCLBUFFER
#define PCLBUFFER 1048576
#endif

using namespace LATfield2;

template <typename part, typename part_info>
class perfParticles_gevolution;

template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, long * npart, int * npart_row);

template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row);

template <typename part, typename part_info>
__global__ void buffer_tracer_IDs(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count);

template <typename part, typename part_info>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count);

template <typename part, typename part_info, int IDlog_scatter = 0>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, Real * domain, Field<Real> * phi, double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2);

template <typename part, typename part_info>
__global__ void add_particles(perfParticles_gevolution<part, part_info> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx);

template <typename part, typename part_info, typename part_dataType>
class Particles_gevolution: public Particles<part, part_info, part_dataType>
{
	public:
		void saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor = 1, double dtau_pos = 0., double dtau_vel = 0., Field<Real> * phi = NULL);
		template <int IDlog_scatter = 0>
		void saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, set<long> & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor = 1);
		void loadGadget2(string filename, gadget2_header & hdr);
};

template <typename part, typename part_info>
class perfParticles_gevolution: public perfParticles<part, part_info>
{
	public:
		void saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor = 1, double dtau_pos = 0., double dtau_vel = 0., Field<Real> * phi = NULL);
		template <int IDlog_scatter = 0>
		void saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, set<long> & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor = 1);
		void loadGadget2(string filename, gadget2_header & hdr);

		__host__ __device__ void bufferTracerParticle(int row, int idx, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, unsigned long long int buffer_idx, float * pos_offset = nullptr);

		template <typename part2, typename part_info2>
		friend __global__ void count_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, long * npart, int * npart_row);

		template <typename part2, typename part_info2>
		friend __global__ void count_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row);

		template <typename part2, typename part_info2>
		friend __global__ void buffer_tracer_IDs(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count);

		template <typename part2, typename part_info2>
		friend __global__ void buffer_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count);

		template <typename part2, typename part_info2, int IDlog_scatter>
		friend __global__ void buffer_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, Real * domain, Field<Real> * phi, double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2);

		template <typename part2, typename part_info2>
		friend __global__ void add_particles(perfParticles_gevolution<part2, part_info2> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx);
};

template <typename part, typename part_info, typename part_dataType>
void Particles_gevolution<part,part_info,part_dataType>::saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor, double dtau_pos, double dtau_vel, Field<Real> * phi)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File outfile;
	long count, npart;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	uint32_t i;
	double rescale_vel = 1. / sqrt(hdr.time) / GADGET_VELOCITY_CONVERSION;
#ifdef EXACT_OUTPUT_REDSHIFTS
	Real phip = 0.;
	Real gradphi[3] = {0., 0., 0.};
	double ref_dist[3];
	LATfield2::Site xField;
#endif
	
	LATfield2::Site xPart(this->lat_part_);
	
	if (hdr.num_files != 1 && hdr.num_files != parallel.grid_size()[1])
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": number of Gadget2 files does not match the number of processes in dim-1!" << endl;
		return;
	}
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif
	
	npart = 0;
	for(xPart.first(); xPart.test(); xPart.next())
	{
		for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
		{
			if ((*it).ID % tracer_factor == 0)
				npart++;
		}
	}

	if (hdr.num_files == 1)
	{	
		if (parallel.rank() == 0)
		{
			parallel.send<long>(npart, 1);
			parallel.receive<long>(count, parallel.size()-1);
			if (count != hdr.npart[1]) cout << " error: number of particles in saveGadget2 does not match request!" << endl;
			count = 0;
		}
		else
		{
			parallel.receive<long>(count, parallel.rank()-1);
			npart += count;
			parallel.send<long>(npart, (parallel.rank()+1)%parallel.size());
		}
	
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	}
	else
	{
		if (parallel.grid_rank()[0] == 0)
		{
			parallel.send_dim0<long>(npart, 1);
			parallel.receive_dim0<long>(count, parallel.grid_size()[0]-1);
			hdr.npart[1] = (uint32_t) count;
			count = 0;
		}
		else
		{
			parallel.receive_dim0<long>(count, parallel.grid_rank()[0]-1);
			npart += count;
			parallel.send_dim0<long>(npart, (parallel.grid_rank()[0]+1)%parallel.grid_size()[0]);
		}

		parallel.broadcast_dim0<uint32_t>(hdr.npart[1], 0);
		
		std::string filename_dot_number = filename + "." + std::to_string(parallel.grid_rank()[1]);

		MPI_File_open(parallel.dim0_comm()[parallel.grid_rank()[1]], filename_dot_number.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	}
	
	offset_pos = (MPI_Offset) hdr.npart[1];
	offset_pos *= (MPI_Offset) (6 * sizeof(float) + ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
	MPI_File_set_size(outfile, offset_pos);
	
	offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
	offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float)));
	offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1] - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	
	if ((hdr.num_files == 1 && parallel.rank() == 0) || (hdr.num_files > 1 && parallel.grid_rank()[0] == 0))
	{
		blocksize = sizeof(hdr);		
		MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
		MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		blocksize = 3 * sizeof(float) * hdr.npart[1];
		MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
		MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
	}
	
	count = 0;
#ifdef EXACT_OUTPUT_REDSHIFTS
	if (phi != NULL)
		xField.initialize(phi->lattice());
	else
		xField.initialize(this->lat_part_);
	xField.first();
#endif
	for(xPart.first(); xPart.test(); xPart.next())
	{
		for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
		{
			if ((*it).ID % tracer_factor == 0)
			{
#ifdef EXACT_OUTPUT_REDSHIFTS
				if (phi != NULL)
				{
					for (i = 0; i < 3; i++)
						ref_dist[i] = modf((*it).pos[i] / this->lat_resolution_, &phip);
							
					phip = (*phi)(xField) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]);
					phip += (*phi)(xField+0) * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]);
					phip += (*phi)(xField+1) * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]);
					phip += (*phi)(xField+0+1) * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]);
					phip += (*phi)(xField+2) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2];
					phip += (*phi)(xField+0+2) * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2];
					phip += (*phi)(xField+1+2) * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2];
					phip += (*phi)(xField+0+1+2) * ref_dist[0] * ref_dist[1] * ref_dist[2];
					
					gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
					gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
					gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
					gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
					gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
					gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
					gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
					gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
					gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
					gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
					gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
					gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));
				}
					
				ref_dist[0] = (*it).vel[0]*(*it).vel[0] + (*it).vel[1]*(*it).vel[1] + (*it).vel[2]*(*it).vel[2];
				ref_dist[1] = ref_dist[0] + hdr.time * hdr.time;
				ref_dist[2] = sqrt(ref_dist[1]);
				ref_dist[0] += ref_dist[1];
				ref_dist[1] = 1. + (4. - (ref_dist[0] / ref_dist[1])) * phip;
					
				for (i = 0; i < 3; i++)
					posdata[3*count+i] = modf(1. + (*it).pos[i] + dtau_pos * (*it).vel[i] * ref_dist[1] / ref_dist[2], &phip) * hdr.BoxSize;
					
				for (i = 0; i < 3; i++)
					veldata[3*count+i] = ((*it).vel[i] - dtau_vel * ref_dist[0] * gradphi[i] / this->lat_resolution_ / ref_dist[2]) * rescale_vel / hdr.time;
#else						
				for (i = 0; i < 3; i++)
					posdata[3*count+i] = (*it).pos[i] * hdr.BoxSize;
					
				for (i = 0; i < 3; i++)
					veldata[3*count+i] = (*it).vel[i] * rescale_vel / hdr.time;
#endif
					
#if GADGET_ID_BYTES == 8
				*((int64_t *) IDs + count) = (int64_t) (*it).ID;
#else	
				*((int32_t *) IDs + count) = (int32_t) (*it).ID;
#endif
					
				count++;
						
				if (count == PCLBUFFER)
				{
					MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
					offset_pos += 3 * PCLBUFFER * sizeof(float);
					MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
					offset_vel += 3 * PCLBUFFER * sizeof(float);
					count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
					MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
					offset_ID += count;
					count = 0;
				}
			}
		}
#ifdef EXACT_OUTPUT_REDSHIFTS
		xField.next();
#endif
	}
	
	MPI_File_write_at_all(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
	MPI_File_write_at_all(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
	count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
	MPI_File_write_at_all(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
	
	MPI_File_close(&outfile);
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to count particles to be written
template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, long * npart, int * npart_row)
{
	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	long local_count = 0;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
			local_count++;
	}

	cuda::atomic_ref<long, cuda::thread_scope_device> count_ref(*npart);
	count_ref.fetch_add(local_count);

	cuda::atomic_ref<int, cuda::thread_scope_device> row_count_ref(npart_row[row]);
	row_count_ref.fetch_add(local_count);
}

// CUDA kernel to write particles to buffers
template <typename part, typename part_info>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			unsigned long long int buffer_idx = atomicAdd(buffer_count, 1);
			
			pcl->bufferTracerParticle(row, idx, dtau_pos, dtau_vel, a, boxsize, phi, posdata, veldata, IDs, buffer_idx);
		}
	}
}

template <typename part, typename part_info>
__host__ __device__ void perfParticles_gevolution<part,part_info>::bufferTracerParticle(int row, int idx, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, unsigned long long int buffer_idx, float * pos_offset)
{
	Real rescale_vel = Real(1) / sqrt(static_cast<Real>(a)) / GADGET_VELOCITY_CONVERSION;

	#ifdef EXACT_OUTPUT_REDSHIFTS
	Real phip = Real(0);
	Real gradphi[3] = {Real(0), Real(0), Real(0)};
	int coord[3];
	Real frac[3];

	Real dx = this->boxSize_[0] / this->lat_size_[0];
	
	if (phi != NULL)
	{
		LATfield2::Site xField;
		constexpr Real one = Real(1);

		coord[0] = (int) floor(this->row_buffers_[row].p[3*idx]*this->lat_size_[0]) % this->lat_size_[0];
		coord[1] = (int) floor(this->row_buffers_[row].p[3*idx+1]*this->lat_size_[1]) % this->lat_size_[1];
		coord[2] = (int) floor(this->row_buffers_[row].p[3*idx+2]*this->lat_size_[2]) % this->lat_size_[2];

		xField = Site(phi->lattice(), phi->lattice().siteFirst() 
				+ coord[0]*phi->lattice().jump(0) 
				+ (coord[1] - this->coordSkip_[1])*phi->lattice().jump(1)
				+ (coord[2] - this->coordSkip_[0])*phi->lattice().jump(2));

		frac[0] = this->row_buffers_[row].p[3*idx]*this->lat_size_[0] - coord[0];
		frac[1] = this->row_buffers_[row].p[3*idx+1]*this->lat_size_[1] - coord[1];
		frac[2] = this->row_buffers_[row].p[3*idx+2]*this->lat_size_[2] - coord[2];
				
		phip = (*phi)(xField) * (one-frac[0]) * (one-frac[1]) * (one-frac[2]);
		phip += (*phi)(xField+0) * frac[0] * (one-frac[1]) * (one-frac[2]);
		phip += (*phi)(xField+1) * (one-frac[0]) * frac[1] * (one-frac[2]);
		phip += (*phi)(xField+0+1) * frac[0] * frac[1] * (one-frac[2]);
		phip += (*phi)(xField+2) * (one-frac[0]) * (one-frac[1]) * frac[2];
		phip += (*phi)(xField+0+2) * frac[0] * (one-frac[1]) * frac[2];
		phip += (*phi)(xField+1+2) * (one-frac[0]) * frac[1] * frac[2];
		phip += (*phi)(xField+0+1+2) * frac[0] * frac[1] * frac[2];
		
		gradphi[0] = (one-frac[1]) * (one-frac[2]) * ((*phi)(xField+0) - (*phi)(xField));
		gradphi[1] = (one-frac[0]) * (one-frac[2]) * ((*phi)(xField+1) - (*phi)(xField));
		gradphi[2] = (one-frac[0]) * (one-frac[1]) * ((*phi)(xField+2) - (*phi)(xField));
		gradphi[0] += frac[1] * (one-frac[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
		gradphi[1] += frac[0] * (one-frac[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
		gradphi[2] += frac[0] * (one-frac[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
		gradphi[0] += (one-frac[1]) * frac[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
		gradphi[1] += (one-frac[0]) * frac[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
		gradphi[2] += (one-frac[0]) * frac[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
		gradphi[0] += frac[1] * frac[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
		gradphi[1] += frac[0] * frac[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
		gradphi[2] += frac[0] * frac[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));
	}
		
	frac[0] = this->row_buffers_[row].q[3*idx]*this->row_buffers_[row].q[3*idx] + this->row_buffers_[row].q[3*idx+1]*this->row_buffers_[row].q[3*idx+1] + this->row_buffers_[row].q[3*idx+2]*this->row_buffers_[row].q[3*idx+2];
	frac[1] = frac[0] + static_cast<Real>(a) * static_cast<Real>(a);
	frac[2] = sqrt(frac[1]);
	frac[0] += frac[1];
	frac[1] = Real(1) + (Real(4) - (frac[0] / frac[1])) * phip;

	if (pos_offset != nullptr)
	{
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = (static_cast<float>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]) + pos_offset[i]) * static_cast<float>(boxsize);
	}
	else
	{
		for (int i = 0; i < 3; i++)
#ifdef SINGLE
			posdata[3*buffer_idx+i] = modff(float(1) + static_cast<float>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]), static_cast<float *>(&phip)) * static_cast<float>(boxsize);
#else
			posdata[3*buffer_idx+i] = static_cast<float>(modf(double(1) + static_cast<double>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]), static_cast<double *>(&phip)) * static_cast<double>(boxsize));
#endif
	}
		
	for (int i = 0; i < 3; i++)
		veldata[3*buffer_idx+i] = static_cast<float>((this->row_buffers_[row].q[3*idx+i] - static_cast<Real>(dtau_vel) * frac[0] * gradphi[i] / dx / frac[2]) * rescale_vel) / static_cast<float>(a);
#else
	if (pos_offset != nullptr)
	{
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = (static_cast<float>(this->row_buffers_[row].p[3*idx+i]) + pos_offset[i]) * static_cast<float>(boxsize);
	}
	else
	{						
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = static_cast<float>(this->row_buffers_[row].p[3*idx+i]) * static_cast<float>(boxsize);
	}
					
	for (int i = 0; i < 3; i++)
		veldata[3*buffer_idx+i] = static_cast<float>(this->row_buffers_[row].q[3*idx+i] * rescale_vel) / static_cast<float>(a);
#endif

	IDs[buffer_idx] = this->row_buffers_[row].other[idx];
}


template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor, double dtau_pos, double dtau_vel, Field<Real> * phi)
{
	float * posdata;
	float * veldata;
	long * IDs;
	long count, npart;
	int row_start = 0, row_count;
	uint32_t blocksize;
	unsigned long long int buffer_count;
	int npart_row[this->num_row_buffers_];

	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	IDs = (long *) malloc(sizeof(int64_t) * PCLBUFFER);

	if (posdata == NULL || veldata == NULL || IDs == NULL)
	{
		throw std::runtime_error("Error allocating memory for particle buffers");
	}

	nvtxRangePushA("count particles to be written");
	
	npart = 0;
#pragma omp parallel for
	for (int row = 0; row < this->num_row_buffers_; row++)
	{
		npart_row[row] = 0;
	}
	
	// count particles
	count_tracer_particles<part, part_info><<<this->num_row_buffers_, 128>>>(this, tracer_factor, &npart, npart_row);

	auto success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		throw std::runtime_error("CUDA error in count_tracer_particles");
	}

	nvtxRangePop();

	if (hdr.num_files == 1)
	{	
		if (parallel.rank() == 0)
		{
			parallel.send<long>(npart, 1);
			parallel.receive<long>(count, parallel.size()-1);
			if (count != hdr.npart[1]) cout << " error: number of particles in saveGadget2 does not match request!" << endl;
			count = 0;
		}
		else
		{
			parallel.receive<long>(count, parallel.rank()-1);
			npart += count;
			parallel.send<long>(npart, (parallel.rank()+1)%parallel.size());
		}

		MPI_File outfile;
		MPI_Offset offset_pos, offset_vel, offset_ID;
		MPI_Status status;
	
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);

		offset_pos = (MPI_Offset) hdr.npart[1];
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + sizeof(int64_t));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
		
		offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1] - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) sizeof(int64_t));

		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}

		while (row_start < this->num_row_buffers_)
		{
			count = 0;
			row_count = 0;

			do
			{
				count += npart_row[row_start + row_count];
				row_count++;
			} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

			if (count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
			}
			
			if (count > 0)
			{
				nvtxRangePushA("buffer particles");
				buffer_count = 0;

				buffer_tracer_particles<<<row_count, 128>>>(this, tracer_factor, dtau_pos, dtau_vel, hdr.time, hdr.BoxSize, phi, posdata, veldata, IDs, row_start, &buffer_count);

				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in buffer_tracer_particles");
				}
				nvtxRangePop();

				nvtxRangePushA("write particles to disk");
				MPI_File_write_at(outfile, offset_pos, posdata, 3 * buffer_count, MPI_FLOAT, &status);
				offset_pos += 3 * buffer_count * sizeof(float);
				MPI_File_write_at(outfile, offset_vel, veldata, 3 * buffer_count, MPI_FLOAT, &status);
				offset_vel += 3 * buffer_count * sizeof(float);
				buffer_count *= sizeof(int64_t);
				MPI_File_write_at(outfile, offset_ID, IDs, buffer_count, MPI_BYTE, &status);
				offset_ID += buffer_count;
				nvtxRangePop();
			}

			row_start += row_count;
		}

		MPI_File_close(&outfile);
	}
	else // writing multiple files independently - no MPI overhead
	{
		if (hdr.num_files != parallel.size())
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": number of Gadget2 files does not match the number of tasks!" << endl;
			return;
		}

		hdr.npart[1] = (uint32_t) npart;

		std::string filename_dot_number = filename + "." + std::to_string(parallel.rank());

		FILE * outfile = fopen(filename_dot_number.c_str(), "wb");

		if (outfile == NULL)
		{
			throw std::runtime_error("Error opening output file");
		}

		uint64_t offset_pos = (uint64_t) (3 * sizeof(uint32_t) + sizeof(hdr));
		uint64_t offset_vel = offset_pos + (uint64_t) (2 * sizeof(uint32_t)) + ((uint64_t) hdr.npart[1]) * ((uint64_t) (3 * sizeof(float)));
		uint64_t offset_ID = offset_vel + (uint64_t) (2 * sizeof(uint32_t)) + ((uint64_t) hdr.npart[1]) * ((uint64_t) (3 * sizeof(float)));

		blocksize = sizeof(hdr);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fwrite(&hdr, sizeof(hdr), 1, outfile);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		blocksize = 3 * sizeof(float) * hdr.npart[1];
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_vel - 2*sizeof(uint32_t), SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_ID - 2*sizeof(uint32_t), SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		blocksize = sizeof(int64_t) * hdr.npart[1];
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_ID + blocksize, SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);

		while (row_start < this->num_row_buffers_)
		{
			nvtxRangePushA("buffer particles");
			count = 0;
			row_count = 0;

			do
			{
				count += npart_row[row_start + row_count];
				row_count++;
			} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

			if (count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
			}
			
			buffer_count = 0;

			buffer_tracer_particles<<<row_count, 128>>>(this, tracer_factor, dtau_pos, dtau_vel, hdr.time, hdr.BoxSize, phi, posdata, veldata, IDs, row_start, &buffer_count);

			success = cudaDeviceSynchronize();

			if (success != cudaSuccess)
			{
				throw std::runtime_error("CUDA error in buffer_tracer_particles");
			}
			nvtxRangePop();

			nvtxRangePushA("write particles to disk");
			fseek(outfile, offset_pos, SEEK_SET);
			fwrite(posdata, 3 * buffer_count, sizeof(float), outfile);
			offset_pos += 3 * buffer_count * sizeof(float);
			fseek(outfile, offset_vel, SEEK_SET);
			fwrite(veldata, 3 * buffer_count, sizeof(float), outfile);
			offset_vel += 3 * buffer_count * sizeof(float);
			buffer_count *= sizeof(int64_t);
			fseek(outfile, offset_ID, SEEK_SET);
			fwrite(IDs, buffer_count, 1, outfile);
			offset_ID += buffer_count;
			nvtxRangePop();

			row_start += row_count;
		}

		fclose(outfile);
	}

	free(posdata);
	free(veldata);
	free(IDs);
}


template <typename part, typename part_info, typename part_dataType>
template <int IDlog_scatter>
void Particles_gevolution<part,part_info,part_dataType>::saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, set<long> & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File outfile;
	long count, npart;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	uint32_t i;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	double inner = dist - 0.5 * dtau;
	double outer = dist + (0.5 + LIGHTCONE_IDCHECK_ZONE) * dtau_old;
	double d, v2, e2, vlos;
	double ref_dist[3];
	Real gradphi[3];
	
	LATfield2::Site xPart(this->lat_part_);
	LATfield2::Site xField(phi->lattice());

	double domain[4];

	domain[0] = this->lat_part_.coordSkip()[1];
	domain[1] = domain[0] + this->lat_part_.sizeLocal(1);
	domain[2] = this->lat_part_.coordSkip()[0];
	domain[3] = domain[2] + this->lat_part_.sizeLocal(2);

	for (int j = 0; j < 4; j++)
		domain[j] *= this->lat_resolution_;
	
	if (hdr.num_files != 1)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": writing multiple Gadget2 files not currently supported!" << endl;
		return;
	}
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif
	
	npart = 0;
	if (vertexcount > 0)
	{
		for(xPart.first(), xField.first(); xPart.test(); xPart.next(), xField.next())
		{
			for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
			{
				if ((*it).ID % tracer_factor == 0)
				{
					for (i = 0; i < (uint32_t) vertexcount; i++)
					{
						d = sqrt(((*it).pos[0]-vertex[i][0])*((*it).pos[0]-vertex[i][0]) + ((*it).pos[1]-vertex[i][1])*((*it).pos[1]-vertex[i][1]) + ((*it).pos[2]-vertex[i][2])*((*it).pos[2]-vertex[i][2]));

						if (d < inner || d >= outer) continue;

						if (lightcone.opening == -1. || (((*it).pos[0]-vertex[i][0])*lightcone.direction[0] + ((*it).pos[1]-vertex[i][1])*lightcone.direction[1] + ((*it).pos[2]-vertex[i][2])*lightcone.direction[2]) / d > lightcone.opening)
						{
							if (outer - d > 2. * LIGHTCONE_IDCHECK_ZONE * dtau_old || IDbacklog.find((*it).ID) == IDbacklog.end())
							{
								if (d - inner < 2. * LIGHTCONE_IDCHECK_ZONE * dtau)
								{
									if (IDlog_scatter)
									{
										if ((*it).pos[1] - domain[0] < LIGHTCONE_IDCHECK_ZONE * dtau) // left edge
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower left corner
												IDprelog[0].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper left corner
												IDprelog[6].push_back((*it).ID);
											else
												IDprelog[3].push_back((*it).ID);
										}
										else if (domain[1] - (*it).pos[1] < LIGHTCONE_IDCHECK_ZONE * dtau) // right edge
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower right corner
												IDprelog[2].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper right corner
												IDprelog[8].push_back((*it).ID);
											else
												IDprelog[5].push_back((*it).ID);
										}
										else
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower edge
												IDprelog[1].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper edge
												IDprelog[7].push_back((*it).ID);
											else
												IDprelog[4].push_back((*it).ID);
										}
									}
									else
										(*IDprelog).push_back((*it).ID);
								}

								for (int j = 0; j < 3; j++)
									ref_dist[j] = modf((*it).pos[j] / this->lat_resolution_, &v2);
									
								v2 = (*it).vel[0] * (*it).vel[0] + (*it).vel[1] * (*it).vel[1] + (*it).vel[2] * (*it).vel[2];
								e2 = v2 + hdr.time * (hdr.time + (dist - d - 0.5 * dtau_old) * dadtau);
								vlos = ((*it).vel[0]*((*it).pos[0]-vertex[i][0]) + (*it).vel[1]*((*it).pos[1]-vertex[i][1]) + (*it).vel[2]*((*it).pos[2]-vertex[i][2])) / d;
	
								gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
								gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
								gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
								gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
								gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
								gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
								gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
								gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
								gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
								gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
								gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
								gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));

								gradphi[0] *= (v2 + e2) / e2 / this->lat_resolution_;
								gradphi[1] *= (v2 + e2) / e2 / this->lat_resolution_;
								gradphi[2] *= (v2 + e2) / e2 / this->lat_resolution_;
						
								e2 = sqrt(e2);
									
								if (d < dist)
								{
									vlos -= dtau * sqrt(v2 + hdr.time * hdr.time) * (gradphi[0]*((*it).pos[0]-vertex[i][0]) + gradphi[1]*((*it).pos[1]-vertex[i][1]) + gradphi[2]*((*it).pos[2]-vertex[i][2])) / d;
									vlos /= sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
								}
								else
									vlos /= sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));

								for (uint32_t j = 0; j < 3; j++)
									veldata[3*(npart%PCLBUFFER)+j] = ((*it).vel[j] - (((dist - d) / (1. + vlos)) + 0.5 * dtau_old) * e2 * gradphi[j]) * rescale_vel / (hdr.time + ((dist - d) / (1. + vlos)) * dadtau);
										
								if (d >= dist)
								{
									e2 = sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));
										
									for (uint32_t j = 0; j < 3; j++)
										posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * (*it).vel[j] / e2) * hdr.BoxSize;
								}
								else
								{
									e2 = sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
									v2 = sqrt(v2 + hdr.time * hdr.time);
										
									for (uint32_t j = 0; j < 3; j++)
										posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * ((*it).vel[j] - dtau * v2 * gradphi[j]) / e2) * hdr.BoxSize;
								}
					
#if GADGET_ID_BYTES == 8
								*((int64_t *) IDs + (npart%PCLBUFFER)) = (int64_t) (*it).ID;
#else	
								*((int32_t *) IDs + (npart%PCLBUFFER)) = (int32_t) (*it).ID;
#endif

								npart++;
							}

							break;
						}
					}
				}
			}
		}
	}
	
	if (parallel.rank() == 0)
	{
		parallel.send<long>(npart, 1);
		parallel.receive<long>(count, parallel.size()-1);
		hdr.npart[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotal[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotalHW[1] = (uint32_t) (count / (1ll << 32));
		count = 0;
	}
	else
	{
		parallel.receive<long>(count, parallel.rank()-1);
		count += npart;
		parallel.send<long>(count, (parallel.rank()+1)%parallel.size());
		count -= npart;
	}

	parallel.broadcast<uint32_t>(hdr.npartTotal[1], 0);
	parallel.broadcast<uint32_t>(hdr.npartTotalHW[1], 0);

	if (hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32) > 0)
	{
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	
		offset_pos = (MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32));
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
	
		offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32))) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32)) - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	
		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}
		
		count = (npart < PCLBUFFER) ? npart : PCLBUFFER;
		npart -= count;
		MPI_File_write_at_all(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
		offset_pos += 3 * count * sizeof(float);
		MPI_File_write_at_all(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
		offset_vel += 3 * count * sizeof(float);
		count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
		MPI_File_write_at_all(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
		offset_ID += count;
		count = 0;

		if (npart > 0)
		{
			for(xPart.first(); xPart.test() && npart > 0; xPart.next())
			{
				for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
				{
					if ((*it).ID % tracer_factor == 0)
					{
						for (i = 0; i < (uint32_t) vertexcount; i++)
						{
							d = sqrt(((*it).pos[0]-vertex[i][0])*((*it).pos[0]-vertex[i][0]) + ((*it).pos[1]-vertex[i][1])*((*it).pos[1]-vertex[i][1]) + ((*it).pos[2]-vertex[i][2])*((*it).pos[2]-vertex[i][2]));

							if (d < inner || d >= outer) continue;

							if (lightcone.opening == -1. || (((*it).pos[0]-vertex[i][0])*lightcone.direction[0] + ((*it).pos[1]-vertex[i][1])*lightcone.direction[1] + ((*it).pos[2]-vertex[i][2])*lightcone.direction[2]) / d > lightcone.opening)
							{
								if (outer - d > 2. * LIGHTCONE_IDCHECK_ZONE * dtau_old || IDbacklog.find((*it).ID) == IDbacklog.end())
								{
									for (int j = 0; j < 3; j++)
										ref_dist[j] = modf((*it).pos[j] / this->lat_resolution_, &v2);
									
									v2 = (*it).vel[0] * (*it).vel[0] + (*it).vel[1] * (*it).vel[1] + (*it).vel[2] * (*it).vel[2];
									e2 = v2 + hdr.time * (hdr.time + (dist - d - 0.5 * dtau_old) * dadtau);
									vlos = ((*it).vel[0]*((*it).pos[0]-vertex[i][0]) + (*it).vel[1]*((*it).pos[1]-vertex[i][1]) + (*it).vel[2]*((*it).pos[2]-vertex[i][2])) / d;
	
									gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
									gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
									gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
									gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
									gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
									gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
									gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
									gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
									gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
									gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
									gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
									gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));

									gradphi[0] *= (v2 + e2) / e2 / this->lat_resolution_;
									gradphi[1] *= (v2 + e2) / e2 / this->lat_resolution_;
									gradphi[2] *= (v2 + e2) / e2 / this->lat_resolution_;
						
									e2 = sqrt(e2);
										
									if (d < dist)
									{
										vlos -= dtau * sqrt(v2 + hdr.time * hdr.time) * (gradphi[0]*((*it).pos[0]-vertex[i][0]) + gradphi[1]*((*it).pos[1]-vertex[i][1]) + gradphi[2]*((*it).pos[2]-vertex[i][2])) / d;
										vlos /= sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
									}
									else
										vlos /= sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));

									for (uint32_t j = 0; j < 3; j++)
										veldata[3*(npart%PCLBUFFER)+j] = ((*it).vel[j] - (((dist - d) / (1. + vlos)) + 0.5 * dtau_old) * e2 * gradphi[j]) * rescale_vel / (hdr.time + ((dist - d) / (1. + vlos)) * dadtau);
											
									if (d >= dist)
									{
										e2 = sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));
										
										for (uint32_t j = 0; j < 3; j++)
											posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * (*it).vel[j] / e2) * hdr.BoxSize;
									}
									else
									{
										e2 = sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
										v2 = sqrt(v2 + hdr.time * hdr.time);
										
										for (uint32_t j = 0; j < 3; j++)
											posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * ((*it).vel[j] - dtau * v2 * gradphi[j]) / e2) * hdr.BoxSize;
									}
						
#if GADGET_ID_BYTES == 8
									*((int64_t *) IDs + count) = (int64_t) (*it).ID;
#else	
									*((int32_t *) IDs + count) = (int32_t) (*it).ID;
#endif
	
									npart--;
									count++;
								}
								break;
							}
						}
							
						if (count == PCLBUFFER)
						{
							MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
							offset_pos += 3 * PCLBUFFER * sizeof(float);
							MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
							offset_vel += 3 * PCLBUFFER * sizeof(float);
							count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
							MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
							offset_ID += count;
							count = 0;
						}

						if (npart <= 0) break;
					}
				}
			}

			if (count > 0)
			{
					MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
					MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
					count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
					MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
			}
		}	
	
		MPI_File_close(&outfile);
	}
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to count particles to be written
template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row)
{
	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	long local_count = 0;
	long local_count_checkID = 0;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						local_count_checkID++;
					}
					
					local_count++;
					break;
				}
			}
		}
	}

	if (local_count > 0)
	{
		cuda::atomic_ref<long, cuda::thread_scope_device> count_ref(*npart);
		count_ref.fetch_add(local_count);

		cuda::atomic_ref<int, cuda::thread_scope_device> row_count_ref(npart_row[row]);
		row_count_ref.fetch_add(local_count);

		cuda::atomic_ref<int, cuda::thread_scope_device> row_count_checkID_ref(npart_checkID_row[row]);
		row_count_checkID_ref.fetch_add(local_count_checkID);
	}
}


// CUDA kernel to write particle IDs to buffers
template <typename part, typename part_info>
__global__ void buffer_tracer_IDs(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real inner, Real outer, Real dtau_old, double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						unsigned long long int buffer_idx = atomicAdd(buffer_count, 1);
						IDs[buffer_idx] = pcl->row_buffers_[row].other[idx];
					}

					break;
				}
			}
		}
	}
}


// CUDA kernel to write particles to buffers
template <typename part, typename part_info, int IDlog_scatter>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry & lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, Real * domain, Field<Real> * phi, double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;
	unsigned long long int buffer_idx;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						buffer_idx = atomicAdd(buffer_count1, 1);
					}
					else
					{
						buffer_idx = atomicAdd(buffer_count2, 1);
					}

					if (d - inner < Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau)
					{
						if (IDlog_scatter)
						{
							if (pcl->row_buffers_[row].p[3*idx+1] - domain[0] < LIGHTCONE_IDCHECK_ZONE * dtau) // left edge
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower left corner
									loginfo[buffer_idx] = 0;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper left corner
									loginfo[buffer_idx] = 6;
								else
									loginfo[buffer_idx] = 3;
							}
							else if (domain[1] - pcl->row_buffers_[row].p[3*idx+1] < LIGHTCONE_IDCHECK_ZONE * dtau) // right edge
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower right corner
									loginfo[buffer_idx] = 2;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper right corner
									loginfo[buffer_idx] = 8;
								else
									loginfo[buffer_idx] = 5;
							}
							else
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower edge
									loginfo[buffer_idx] = 1;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper edge
									loginfo[buffer_idx] = 7;
								else
									loginfo[buffer_idx] = 4;
							}
						}
						else
						{
							loginfo[buffer_idx] = 0;
						}
					}
					else
					{
						loginfo[buffer_idx] = 255;
					}

					Real vlos = sqrt(pcl->row_buffers_[row].q[3*idx]*pcl->row_buffers_[row].q[3*idx] + pcl->row_buffers_[row].q[3*idx+1]*pcl->row_buffers_[row].q[3*idx+1] + pcl->row_buffers_[row].q[3*idx+2]*pcl->row_buffers_[row].q[3*idx+2] + static_cast<Real>(a * (a + dtau_old * dadtau)));
						
					vlos = (pcl->row_buffers_[row].q[3*idx]*(pcl->row_buffers_[row].p[3*idx]-vertex[v][0]) + pcl->row_buffers_[row].q[3*idx+1]*(pcl->row_buffers_[row].p[3*idx+1]-vertex[v][1]) + pcl->row_buffers_[row].q[3*idx+2]*(pcl->row_buffers_[row].p[3*idx+2]-vertex[v][2])) / d / vlos;

					double dtau_pos = (dist - d) / (1. + vlos);
					float pos_offset[3];

					pos_offset[0] = lightcone.vertex[0] - vertex[v][0];
					pos_offset[1] = lightcone.vertex[1] - vertex[v][1];
					pos_offset[2] = lightcone.vertex[2] - vertex[v][2];

					pcl->bufferTracerParticle(row, idx, dtau_pos, dtau_pos + 0.5 * dtau_old, a + dtau_pos * dadtau, boxsize, phi, posdata, veldata, IDs, buffer_idx, pos_offset);

					break;
				}
			}
		}
	}
}


template <typename part, typename part_info>
template <int IDlog_scatter>
void perfParticles_gevolution<part,part_info>::saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, set<long> & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor)
{
	float * posdata;
	float * veldata;
	long * IDs;
	unsigned char * loginfo;
	long count, npart, reject;
	int row_start = 0, row_count;
	MPI_File outfile;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	unsigned long long int buffer_count1, buffer_count2;
	int npart_row[this->num_row_buffers_];
	int npart_checkID_row[this->num_row_buffers_];
	Real domain[4];
	Real inner = dist - 0.5 * dtau;
	Real outer = dist + (0.5 + LIGHTCONE_IDCHECK_ZONE) * dtau_old;

	if (hdr.num_files != 1)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": writing multiple Gadget2 files not currently supported!" << endl;
		return;
	}

	domain[0] = this->coordSkip_[1] * this->boxSize_[0] / this->lat_size_[0];
	domain[1] = domain[0] + this->lat_size_local_[1] * this->boxSize_[0] / this->lat_size_[0];
	domain[2] = this->coordSkip_[0] * this->boxSize_[0] / this->lat_size_[0];
	domain[3] = domain[2] + this->lat_size_local_[2] * this->boxSize_[0] / this->lat_size_[0];

	IDs = (long *) malloc(sizeof(int64_t) * PCLBUFFER);

	if (IDs == NULL)
	{
		throw std::runtime_error("Error allocating memory for particle IDs");
	}

	nvtxRangePushA("count particles to be written");

	npart = 0;
#pragma omp parallel for
	for (int row = 0; row < this->num_row_buffers_; row++)
	{
		npart_row[row] = 0;
		npart_checkID_row[row] = 0;
	}

	// count particles
	count_tracer_particles<part, part_info><<<this->num_row_buffers_, 128>>>(this, tracer_factor, lightcone, inner, outer, dtau_old, vertex, vertexcount, &npart, npart_row, npart_checkID_row);

	auto success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		throw std::runtime_error("CUDA error in count_tracer_particles");
	}

	// first loop: collect IDs to be checked against IDbacklog
	while (row_start < this->num_row_buffers_)
	{
		count = 0;
		row_count = 0;

		do
		{
			count += npart_checkID_row[row_start + row_count];
			row_count++;
		} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

		if (count > PCLBUFFER)
		{
			long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);

			if (new_IDs == NULL)
			{
				throw std::runtime_error("Error reallocating memory for particle IDs");
			}

			IDs = new_IDs;
		}

		if (count > 0)
		{
			buffer_count1 = 0;

			buffer_tracer_IDs<part, part_info><<<row_count, 128>>>(this, tracer_factor, lightcone, inner, outer, dtau_old, vertex, vertexcount, IDs, row_start, &buffer_count1);

			success = cudaDeviceSynchronize();

			if (success != cudaSuccess)
			{
				throw std::runtime_error("CUDA error in buffer_tracer_IDs");
			}

			// check IDs against IDbacklog
			reject = 0;

#pragma omp parallel for reduction(+:reject)
			for (unsigned long long int i = 0; i < buffer_count1; i++)
			{
				if (IDbacklog.find(IDs[i]) != IDbacklog.end())
				{
					reject++;
				}
			}

			npart -= reject;
		}

		row_start += row_count;
	}

	// communicate number of particles to be written
	if (parallel.rank() == 0)
	{
		parallel.send<long>(npart, 1);
		parallel.receive<long>(count, parallel.size()-1);
		hdr.npart[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotal[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotalHW[1] = (uint32_t) (count / (1ll << 32));
		count = 0;
	}
	else
	{
		parallel.receive<long>(count, parallel.rank()-1);
		count += npart;
		parallel.send<long>(count, (parallel.rank()+1)%parallel.size());
		count -= npart;
	}

	parallel.broadcast<uint32_t>(hdr.npartTotal[1], 0);
	parallel.broadcast<uint32_t>(hdr.npartTotalHW[1], 0);

	nvtxRangePop();

	if (hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32) > 0)
	{
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	
		offset_pos = (MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32));
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + sizeof(int64_t));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
	
		offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32))) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32)) - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) sizeof(int64_t));
	
		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}

		posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
		veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
		loginfo = (unsigned char *) malloc(PCLBUFFER);

		if (posdata == NULL || veldata == NULL || loginfo == NULL)
		{
			throw std::runtime_error("Error allocating memory for particle buffers");
		}

		// second loop: buffer and write particles
		row_start = 0;

		while (row_start < this->num_row_buffers_)
		{
			nvtxRangePushA("buffer particles");
			count = 0;
			row_count = 0;
			buffer_count2 = 0;

			do
			{
				count += npart_row[row_start + row_count];
				buffer_count2 += npart_checkID_row[row_start + row_count];
				row_count++;
			} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

			if (count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);
				unsigned char * new_loginfo = (unsigned char *) realloc(loginfo, count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL || new_loginfo == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
				loginfo = new_loginfo;
			}

			if (count > 0)
			{
				buffer_count1 = 0;

				buffer_tracer_particles<part, part_info, IDlog_scatter><<<row_count, 128>>>(this, tracer_factor, lightcone, (Real) dist, inner, outer, dtau, dtau_old, (double) hdr.time, dadtau, this->boxSize_[0], domain, phi, vertex, vertexcount, posdata, veldata, IDs, loginfo, row_start, &buffer_count1, &buffer_count2);

				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in buffer_tracer_particles");
				}

				if (buffer_count1 > 0)
				{
#pragma omp parallel for
					for (unsigned long long int i = 0; i < buffer_count1; i++)
					{
						if (IDbacklog.find(IDs[i]) != IDbacklog.end()) // need to remove particle from buffers
						{
							#pragma omp critical
							{
								if (count > buffer_count1)
								{
									for (int j = 0; j < 3; j++)
									{
										posdata[3*i+j] = posdata[3*(count-1)+j];
										veldata[3*i+j] = veldata[3*(count-1)+j];
									}

									IDs[i] = IDs[count-1];
									loginfo[i] = loginfo[count-1];

									count--;
								}
							}
						}
					}

					if (count == buffer_count1) // we were unlucky and need to check again in a non-parallel way
					{
						for (unsigned long long int i = 0; i < count; i++)
						{
							if (IDbacklog.find(IDs[i]) != IDbacklog.end())
							{
								for (int j = 0; j < 3; j++)
								{
									posdata[3*i+j] = posdata[3*(count-1)+j];
									veldata[3*i+j] = veldata[3*(count-1)+j];
								}

								IDs[i] = IDs[count-1];
								loginfo[i] = loginfo[count-1];

								count--;
								i--;
							}
						}
					}
				}
			}
			nvtxRangePop();

			nvtxRangePushA("write particles to disk");
			if (count > 0)
			{
				// fill the IDprelogs
#pragma omp parallel for
				for (long i = 0; i < count; i++)
				{
					if (loginfo[i] < 255)
					{
						#pragma omp critical
						{
							IDprelog[loginfo[i]].push_back(IDs[i]);
						}
					}
				}

				MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
				offset_pos += 3 * count * sizeof(float);
				MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
				offset_vel += 3 * count * sizeof(float);
				count *= sizeof(int64_t);
				MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
				offset_ID += count;
			}
			nvtxRangePop();

			row_start += row_count;
		}

		MPI_File_close(&outfile);

		free(posdata);
		free(veldata);
		free(loginfo);
	}

	free(IDs);
}


template <typename part, typename part_info, typename part_dataType>
void Particles_gevolution<part,part_info,part_dataType>::loadGadget2(string filename, gadget2_header & hdr)
{
	float * posdata;
	float * veldata;
	void * IDs;
	part pcl;
	MPI_File infile;
	uint32_t count, npart = 0;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif

	MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

	MPI_File_read_all(infile, &blocksize, 1, MPI_UNSIGNED, &status);

	if (blocksize != sizeof(hdr))
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": file type not recognized when reading Gadget2 file!" << endl;
		MPI_File_close(&infile);
		return;
	}

	MPI_File_read_all(infile, &hdr, sizeof(hdr), MPI_BYTE, &status);

	rescale_vel /= sqrt(hdr.time);
	offset_pos = (MPI_Offset) sizeof(hdr) + (MPI_Offset) (3 * sizeof(uint32_t));
	offset_vel = offset_pos + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float))) + (MPI_Offset) (2 * sizeof(uint32_t));
	offset_ID = offset_vel + offset_vel - offset_pos;

	MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);
	while (npart < hdr.npart[1])
	{
		count = (hdr.npart[1] - npart > PCLBUFFER) ? PCLBUFFER : (hdr.npart[1] - npart);

		MPI_File_read_all(infile, posdata, 3 * count, MPI_FLOAT, &status);
		offset_pos += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_vel, MPI_SEEK_SET);
		MPI_File_read_all(infile, veldata, 3 * count, MPI_FLOAT, &status);
		offset_vel += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_ID, MPI_SEEK_SET);
#if GADGET_ID_BYTES == 8
		MPI_File_read_all(infile, IDs, count * sizeof(int64_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int64_t));
#else
		MPI_File_read_all(infile, IDs, count * sizeof(int32_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int32_t));
#endif
		MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);

		for (int i = 0; i < 3 * count; i++)
		{
			posdata[i] /= hdr.BoxSize;
			while (posdata[i] >= 1.) posdata[i] -= 1.;
			veldata[i] *= hdr.time / rescale_vel;
		}

		for (int i = 0; i < count; i++)
		{
#if GADGET_ID_BYTES == 8
			pcl.ID = *((int64_t *) IDs + i);
#else
			pcl.ID = *((int32_t *) IDs + i);
#endif
			pcl.pos[0] = posdata[3*i];
			pcl.pos[1] = posdata[3*i+1];
			pcl.pos[2] = posdata[3*i+2];
			pcl.vel[0] = veldata[3*i];
			pcl.vel[1] = veldata[3*i+1];
			pcl.vel[2] = veldata[3*i+2];
			this->addParticle_global(pcl);
		}
		
		npart += count;
	}

	MPI_File_close(&infile);
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to add particles
template <typename part, typename part_info>
__global__ void add_particles(perfParticles_gevolution<part, part_info> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
	{
		int coord[3];

#ifdef SINGLE
		pcl->getPartCoordLocal(posdata+3*i, coord);
#else
		double pos[3];
		pos[0] = posdata[3*i];
		pos[1] = posdata[3*i+1];
		pos[2] = posdata[3*i+2];
		pcl->getPartCoordLocal(pos, coord);
#endif

		if (coord[0] >= 0 && coord[0] < pcl->lat_size_local_[0] && coord[1] >= 0 && coord[1] < pcl->lat_size_local_[1] && coord[2] >= 0 && coord[2] < pcl->lat_size_local_[2])
		{
			unsigned long long int idx = atomicAdd(buffer_idx, 1);

			pcl->p[3*idx] = posdata[3*i];
			pcl->p[3*idx+1] = posdata[3*i+1];
			pcl->p[3*idx+2] = posdata[3*i+2];
			pcl->q[3*idx] = veldata[3*i];
			pcl->q[3*idx+1] = veldata[3*i+1];
			pcl->q[3*idx+2] = veldata[3*i+2];
#if GADGET_ID_BYTES == 8
			pcl->other[idx] = *((int64_t *) IDs + i);
#else
			pcl->other[idx] = *((int32_t *) IDs + i);
#endif
		}
	}
}

template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::loadGadget2(string filename, gadget2_header & hdr)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File infile;
	uint32_t count, npart = 0, nfile = 0;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif

	do
	{
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

		// check if file could be opened
		if (infile == MPI_FILE_NULL)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not open file " << filename << "!" << endl;
			return;
		}

		auto read_success = MPI_File_read_all(infile, &blocksize, 1, MPI_UNSIGNED, &status);

		if (blocksize != sizeof(hdr) || read_success != MPI_SUCCESS)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": file type not recognized when reading Gadget2 file!" << endl;
			MPI_File_close(&infile);
			return;
		}

		read_success = MPI_File_read_all(infile, &hdr, sizeof(hdr), MPI_BYTE, &status);

		if (read_success != MPI_SUCCESS)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read header from file " << filename << "!" << endl;
			MPI_File_close(&infile);
			return;
		}

		rescale_vel /= sqrt(hdr.time);
		offset_pos = (MPI_Offset) sizeof(hdr) + (MPI_Offset) (3 * sizeof(uint32_t));
		offset_vel = offset_pos + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float))) + (MPI_Offset) (2 * sizeof(uint32_t));
		offset_ID = offset_vel + offset_vel - offset_pos;

		MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);
		while (npart < hdr.npart[1])
		{
			count = (hdr.npart[1] - npart > PCLBUFFER) ? PCLBUFFER : (hdr.npart[1] - npart);

			read_success = MPI_File_read_all(infile, posdata, 3 * count, MPI_FLOAT, &status);

			if (read_success != MPI_SUCCESS)
			{
				COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle positions from file " << filename << "!" << endl;
				MPI_File_close(&infile);
				return;
			}

			offset_pos += (MPI_Offset) (3 * count * sizeof(float));
			MPI_File_seek(infile, offset_vel, MPI_SEEK_SET);
			read_success = MPI_File_read_all(infile, veldata, 3 * count, MPI_FLOAT, &status);

			if (read_success != MPI_SUCCESS)
			{
				COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle velocities from file " << filename << "!" << endl;
				MPI_File_close(&infile);
				return;
			}

			offset_vel += (MPI_Offset) (3 * count * sizeof(float));
			MPI_File_seek(infile, offset_ID, MPI_SEEK_SET);
#if GADGET_ID_BYTES == 8
			read_success = MPI_File_read_all(infile, IDs, count * sizeof(int64_t), MPI_BYTE, &status);
			offset_ID += (MPI_Offset) (count * sizeof(int64_t));
#else
			read_success = MPI_File_read_all(infile, IDs, count * sizeof(int32_t), MPI_BYTE, &status);
			offset_ID += (MPI_Offset) (count * sizeof(int32_t));
#endif
			MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);

			if (read_success != MPI_SUCCESS)
			{
				COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle IDs from file " << filename << "!" << endl;
				MPI_File_close(&infile);
				return;
			}

#pragma omp parallel for
			for (int i = 0; i < 3 * count; i++)
			{
				posdata[i] /= hdr.BoxSize;
				while (posdata[i] < 0) posdata[i] += 1;
				while (posdata[i] >= 1) posdata[i] -= 1;
				veldata[i] *= hdr.time / rescale_vel;
			}

			int local_count = 0;

#pragma omp parallel for reduction(+:local_count)
			for (int i = 0; i < count; i++)
			{
				int coord[3];
				Site x(*(this->lat_));
#ifdef SINGLE
				this->getPartCoord(posdata+3*i, coord);
#else
				double pos[3];
				pos[0] = posdata[3*i];
				pos[1] = posdata[3*i+1];
				pos[2] = posdata[3*i+2];
				this->getPartCoord(pos, coord);
#endif
				if (x.setCoord(coord))
				{
					local_count++;
				}
			}

			if (this->num_particles_ + local_count > this->total_capacity_)
			{
				this->resizeGlobalBuffers(this->num_particles_ + local_count + this->extra_capacity_);
			}

			if (local_count > 0)
			{
				unsigned long long int buffer_idx = this->num_particles_;

				add_particles<part, part_info><<<count/128+1, 128>>>(this, posdata, veldata, IDs, count, &buffer_idx);

				auto success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": CUDA error when loading particles!" << endl;
					MPI_File_close(&infile);
					throw std::runtime_error("CUDA error when loading particles!");
				}

				this->num_particles_ += local_count;
			}
			
			npart += count;
		}

		MPI_File_close(&infile);
		nfile++;

		if (hdr.num_files > 1)
		{
			filename = filename.substr(0, filename.find_last_of('.')+1) + to_string(nfile);
		}
	} while (nfile < hdr.num_files);

	this->updateRowBuffers();
	
	free(posdata);
	free(veldata);
	free(IDs);
}


#endif
