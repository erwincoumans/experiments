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
#include "mycuda.h"
#include "cuda.h"

extern "C"
{
void CudaMechanicalObject3f_vClear(unsigned int size, void* res);
void CudaMechanicalObject3f_vEqBF(unsigned int size, void* res, const void* b, float f);
void CudaMechanicalObject3f_vPEqBF(unsigned int size, void* res, const void* b, float f);
void CudaMechanicalObject3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
void CudaMechanicalObject3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float h);
void CudaMechanicalObject3f_vPEq1(unsigned int size, void* res, int index, const float* val);
int CudaMechanicalObject3f_vDotTmpSize(unsigned int size);
void CudaMechanicalObject3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaMechanicalObject3d_vClear(unsigned int size, void* res);
void CudaMechanicalObject3d_vEqBF(unsigned int size, void* res, const void* b, double f);
void CudaMechanicalObject3d_vPEqBF(unsigned int size, void* res, const void* b, double f);
void CudaMechanicalObject3d_vOp(unsigned int size, void* res, const void* a, const void* b, double f);
void CudaMechanicalObject3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double h);
int CudaMechanicalObject3d_vDotTmpSize(unsigned int size);
void CudaMechanicalObject3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void CudaMechanicalObject1t_vEqBF_kernel(int size, real* res, const real* b, real f)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	//if (index < size)
	{
		res[index] = b[index] * f;
	}
}

template<class real>
__global__ void CudaMechanicalObject3t_vEqBF_kernel(int size, real* res, const real* b, real f)
{	
	int index = fastmul(blockIdx.x,BSIZE*3)+threadIdx.x;
	//if (index < size)
	{
		res[index] = b[index] * f;
                index += BSIZE;
		res[index] = b[index] * f;
                index += BSIZE;
		res[index] = b[index] * f;
		//CudaVec3<real> bi = b[index];
		//CudaVec3<real> ri = bi * f;
		//res[index] = ri;
	}
}

template<class real>
__global__ void CudaMechanicalObject1t_vPEqBF_kernel(int size, real* res, const real* b, real f)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	//if (index < size)
	{
		res[index] += b[index] * f;
	}
}

template<class real>
__global__ void CudaMechanicalObject3t_vPEqBF_kernel(int size, real* res, const real* b, real f)
{
	int index = fastmul(blockIdx.x,BSIZE*3)+threadIdx.x;
	//if (index < size)
	{
		res[index] += b[index] * f;
                index += BSIZE;
		res[index] += b[index] * f;
                index += BSIZE;
		res[index] += b[index] * f;
		//CudaVec3<real> bi = b[index];
		//CudaVec3<real> ri = res[index];
		//ri += bi * f;
		//res[index] = ri;
	}
}

template<class real>
struct array3
{
    real v[3];
};

template<class real>
__global__ void CudaMechanicalObject3t_vPEq1_kernel(real* res, array3<real> val)
{
    int index = threadIdx.x;
    res[index] += val.v[index];
}

template<class real>
__global__ void CudaMechanicalObject1t_vOp_kernel(int size, real* res, const real* a, const real* b, real f)
{
	int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
	//if (index < size)
	{
		res[index] = a[index] + b[index] * f;
	}
}

template<class real>
__global__ void CudaMechanicalObject3t_vOp_kernel(int size, real* res, const real* a, const real* b, real f)
{
	int index = fastmul(blockIdx.x,BSIZE*3)+threadIdx.x;
	//if (index < size)
	{
		res[index] = a[index] + b[index] * f;
                index += BSIZE;
		res[index] = a[index] + b[index] * f;
                index += BSIZE;
		res[index] = a[index] + b[index] * f;
		//CudaVec3<real> ai = a[index];
		//CudaVec3<real> bi = b[index];
		//CudaVec3<real> ri = ai + bi * f;
		//res[index] = ri;
	}
}

template<class real>
__global__ void CudaMechanicalObject1t_vIntegrate_kernel(int size, const real* a, real* v, real* x, real h)
{
    int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        real vi = v[index] + a[index] * h;
        v[index] = vi;
        x[index] = x[index] + vi * h;
    }
}

template<class real>
__global__ void CudaMechanicalObject3t_vIntegrate_kernel(int size, const real* a, real* v, real* x, real h)
{
    int index = fastmul(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
	real vi;
	vi = v[index] + a[index] * h;
	v[index] = vi;
	x[index] = x[index] + vi * h;
	index += BSIZE;
	vi = v[index] + a[index] * h;
	v[index] = vi;
	x[index] = x[index] + vi * h;
	index += BSIZE;
	vi = v[index] + a[index] * h;
	v[index] = vi;
	x[index] = x[index] + vi * h;
    }
}


#define RED_BSIZE 128
#define blockSize RED_BSIZE
//template<unsigned int blockSize>
__global__ void CudaMechanicalObject_vDot_kernel(unsigned int n, float* res, const float* a, const float* b)
{
//    extern __shared__ float fdata[];
    __shared__ float fdata[blockSize];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = gridDim.x*(blockSize);
    float r = 0;
    while (i < n) { r += a[i] * b[i]; i += gridSize; }
    fdata[tid] = r;
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { fdata[tid] += fdata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { fdata[tid] += fdata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { fdata[tid] += fdata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32) {
        volatile float* smem = fdata;
#if blockSize >= 64
        //if (blockSize >= 64)
            smem[tid] += smem[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
            smem[tid] += smem[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
            smem[tid] += smem[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
            smem[tid] += smem[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
            smem[tid] += smem[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
            smem[tid] += smem[tid + 1];
#endif
    }
    if (tid == 0) res[blockIdx.x] = fdata[0];
}

__global__ void CudaMechanicalObject_vDot_kernel(unsigned int n, double* res, const double* a, const double* b)
{
    extern __shared__ double ddata[];
//    __shared__ double ddata[blockSize];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = gridDim.x*(blockSize);
    ddata[tid] = 0;
    while (i < n) { ddata[tid] += a[i] * b[i]; i += gridSize; }
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { ddata[tid] += ddata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { ddata[tid] += ddata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { ddata[tid] += ddata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32) {
#if blockSize >= 64
        volatile double* smem = ddata;
        //if (blockSize >= 64)
            smem[tid] += smem[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
            smem[tid] += smem[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
            smem[tid] += smem[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
            smem[tid] += smem[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
            smem[tid] += smem[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
            smem[tid] += smem[tid + 1];
#endif
    }
    if (tid == 0) res[blockIdx.x] = ddata[0];
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaMechanicalObject3f_vClear(unsigned int size, void* res)
{
	cudaMemset(res, 0, size*3*sizeof(float));
}

void CudaMechanicalObject3f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void CudaMechanicalObject3f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vPEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vPEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void CudaMechanicalObject3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vOp_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vOp_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b, f);
}

void CudaMechanicalObject3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float h)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, h);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vIntegrate_kernel<float><<< grid, threads >>>(3*size, (const float*)a, (float*)v, (float*)x, h);
}

void CudaMechanicalObject3f_vPEq1(unsigned int size, void* res, int index, const float* val)
{
    if ((unsigned)index >= size) return;
	dim3 threads(3,1);
	dim3 grid(1,1);
    array3<float> v;
    v.v[0] = val[0];
    v.v[1] = val[1];
    v.v[2] = val[2];
	CudaMechanicalObject3t_vPEq1_kernel<float><<< grid, threads >>>(((float*)res)+(3*index), v);
}

int CudaMechanicalObject3f_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void CudaMechanicalObject3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 3;
    if (size==0)
    {
            *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        CudaMechanicalObject_vDot_kernel /*<float>*/ <<< grid, threads >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0;i<nblocs;i++)
                    r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}


#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaMechanicalObject3d_vClear(unsigned int size, void* res)
{
	cudaMemset(res, 0, size*3*sizeof(double));
}

void CudaMechanicalObject3d_vEqBF(unsigned int size, void* res, const void* b, double f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void CudaMechanicalObject3d_vPEqBF(unsigned int size, void* res, const void* b, double f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vPEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vPEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void CudaMechanicalObject3d_vOp(unsigned int size, void* res, const void* a, const void* b, double f)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vOp_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b, f);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vOp_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a, (const double*)b, f);
}

void CudaMechanicalObject3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double h)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaMechanicalObject3t_vIntegrate_kernel<double><<< grid, threads >>>(size, (const double*)a, (double*)v, (double*)x, h);
	//dim3 grid((3*size+BSIZE-1)/BSIZE,1);
	//CudaMechanicalObject1t_vIntegrate_kernel<double><<< grid, threads >>>(3*size, (const double*)a, (double*)v, (double*)x, h);
}


int CudaMechanicalObject3d_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void CudaMechanicalObject3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 3;
    if (size==0)
    {
            *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        CudaMechanicalObject_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0;i<nblocs;i++)
                    r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

#endif // SOFA_GPU_CUDA_DOUBLE
