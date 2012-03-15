/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

extern "C"
{
void CudaUniformMass3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
void CudaUniformMass3f_accFromF(unsigned int size, float mass, void* a, const void* f);
void CudaUniformMass3f_addForce(unsigned int size, const float *mg, void* f);

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaUniformMass3d_addMDx(unsigned int size, double mass, void* res, const void* dx);
void CudaUniformMass3d_accFromF(unsigned int size, double mass, void* a, const void* f);
void CudaUniformMass3d_addForce(unsigned int size, const double *mg, void* f);

#endif // SOFA_GPU_CUDA_DOUBLE

}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void CudaUniformMass1t_addMDx_kernel(int size, const real mass, real* res, const real* dx)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	if (index < size)
	{
		res[index] += dx[index] * mass;
	}
}

template<class real>
__global__ void CudaUniformMass3t_addMDx_kernel(int size, const real mass, CudaVec3<real>* res, const CudaVec3<real>* dx)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	if (index < size)
	{
		//res[index] += dx[index] * mass;
		CudaVec3<real> dxi = dx[index];
		CudaVec3<real> ri = res[index];
		ri += dxi * mass;
		res[index] = ri;
	}
}

template<class real>
__global__ void CudaUniformMass1t_accFromF_kernel(int size, const real inv_mass, real* a, const real* f)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	if (index < size)
	{
		a[index] = f[index] * inv_mass;
	}
}

template<class real>
__global__ void CudaUniformMass3t_accFromF_kernel(int size, const real inv_mass, CudaVec3<real>* a, const CudaVec3<real>* f)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	if (index < size)
	{
		//a[index] = f[index] * inv_mass;
		CudaVec3<real> fi = f[index];
		fi *= inv_mass;
		a[index] = fi;
	}
}

template<class real>
__global__ void CudaUniformMass1t_addForce_kernel(int size, const real mg, real* f)
{
	int index = fastmul(blockIdx.x,BSIZE);
	if (index < size)
	{
		f[index] += mg;
	}
}

template<class real>
//__global__ void CudaUniformMass3t_addForce_kernel(int size, const CudaVec3<real> mg, real* f)
__global__ void CudaUniformMass3t_addForce_kernel(int size, real mg_x, real mg_y, real mg_z, real* f)
{
	//int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	//f[index] += mg;
	f += fastmul(blockIdx.x,BSIZE*3); //blockIdx.x*BSIZE*3;
	int index = threadIdx.x;
	__shared__  real temp[BSIZE*3];
	temp[index] = f[index];
	temp[index+BSIZE] = f[index+BSIZE];
	temp[index+2*BSIZE] = f[index+2*BSIZE];
	
	__syncthreads();
	
	if (fastmul(blockIdx.x,BSIZE)+threadIdx.x < size)
	{
		int index3 = fastmul(index,3); //3*index;
		temp[index3+0] += mg_x;
		temp[index3+1] += mg_y;
		temp[index3+2] += mg_z;
	}
	
	__syncthreads();
	
	f[index] = temp[index];
	f[index+BSIZE] = temp[index+BSIZE];
	f[index+2*BSIZE] = temp[index+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaUniformMass3f_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
	dim3 threads(BSIZE,1);
	//dim3 grid((size+BSIZE-1)/BSIZE,1);
	//CudaUniformMass3t_addMDx_kernel<float><<< grid, threads >>>(size, mass, (CudaVec3<float>*)res, (const CudaVec3<float>*)dx);
	dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	CudaUniformMass1t_addMDx_kernel<float><<< grid, threads >>>(3*size, mass, (float*)res, (const float*)dx);
}

void CudaUniformMass3f_accFromF(unsigned int size, float mass, void* a, const void* f)
{
	dim3 threads(BSIZE,1);
	//dim3 grid((size+BSIZE-1)/BSIZE,1);
	//CudaUniformMass3t_accFromF_kernel<float><<< grid, threads >>>(size, 1.0f/mass, (CudaVec3<float>*)a, (const CudaVec3<float>*)f);
	dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	CudaUniformMass1t_accFromF_kernel<float><<< grid, threads >>>(3*size, 1.0f/mass, (float*)a, (const float*)f);
}

void CudaUniformMass3f_addForce(unsigned int size, const float *mg, void* f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaUniformMass3t_addForce_kernel<float><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (float*)f);
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaUniformMass3d_addMDx(unsigned int size, double mass, void* res, const void* dx)
{
	dim3 threads(BSIZE,1);
	//dim3 grid((size+BSIZE-1)/BSIZE,1);
	//CudaUniformMass3t_addMDx_kernel<double><<< grid, threads >>>(size, mass, (CudaVec3<double>*)res, (const CudaVec3<double>*)dx);
	dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	CudaUniformMass1t_addMDx_kernel<double><<< grid, threads >>>(3*size, mass, (double*)res, (const double*)dx);
}

void CudaUniformMass3d_accFromF(unsigned int size, double mass, void* a, const void* f)
{
	dim3 threads(BSIZE,1);
	//dim3 grid((size+BSIZE-1)/BSIZE,1);
	//CudaUniformMass3t_accFromF_kernel<double><<< grid, threads >>>(size, 1.0f/mass, (CudaVec3<double>*)a, (const CudaVec3<double>*)f);
	dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	CudaUniformMass1t_accFromF_kernel<double><<< grid, threads >>>(3*size, 1.0f/mass, (double*)a, (const double*)f);
}

void CudaUniformMass3d_addForce(unsigned int size, const double *mg, void* f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaUniformMass3t_addForce_kernel<double><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (double*)f);
}

#endif // SOFA_GPU_CUDA_DOUBLE
