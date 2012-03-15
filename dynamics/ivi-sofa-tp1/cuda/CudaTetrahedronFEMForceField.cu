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
#include "CudaTexture.h"
#include "cuda.h"
#include "mycuda.h"
//#define ATOMIC_ADD_FORCE
//#define ATOMIC_SHARED

#define USE_TEXTURE_ELEMENT_FORCE
#define USE_TEXTURE_X
#if defined(VERSION) && VERSION == 8
#define USE_VEC4
#endif

#define USE_ROT6

extern "C"
{
void CudaTetrahedronFEMForceField3f_prepareX(unsigned int nbVertex, void* x4, const void* x);
void CudaTetrahedronFEMForceField3f_prepareDx(unsigned int nbVertex, void* dx4, const void* dx);
void CudaTetrahedronFEMForceField3f_addForce(unsigned int nbElem, unsigned int nbVertex, bool add,
                                             const void* elems, void* state,
                                             void* f, const void* x,
                                             unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE,
                                             void* eforce, const void* velems);

void CudaTetrahedronFEMForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, bool add, double factor,
                                              const void* elems, const void* state,
                                              void* df, const void* dx,
                                              unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE,
                                              void* eforce, const void* velems);
}

template<class real>
class __align__(16) GPUElement
{
public:
        /// index of the 4 connected vertices
        //Vec<4,int> tetra;
        int ia[BSIZE];
        int ib[BSIZE];
        int ic[BSIZE];
        int id[BSIZE];
        /// material stiffness matrix
        //Mat<6,6,Real> K;
        real gamma_bx2[BSIZE], mu2_bx2[BSIZE];
        /// initial position of the vertices in the local (rotated) coordinate system
        //Vec3f initpos[4];
        real bx[BSIZE],cx[BSIZE];
        real cy[BSIZE],dx[BSIZE],dy[BSIZE],dz[BSIZE];
        /// strain-displacement matrix
        //Mat<12,6,Real> J;
        real Jbx_bx[BSIZE],Jby_bx[BSIZE],Jbz_bx[BSIZE];
        /// unused value to align to 64 bytes
        //real dummy[BSIZE];
};

template<class real>
class GPUElementForce
{
public:
    CudaVec4<real> fA,fB,fC,fD;
};

//////////////////////
// GPU-side methods //
//////////////////////

// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldInputTextures
{
public:

    static __host__ void setX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
	return CudaVec3<real>::make(x[i]);
    }

    static __host__ void setDX(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* x)
    {
	return CudaVec3<real>::make(x[i]);
    }
};


// no texture is used unless this template is specialized
template<typename real, class TIn>
class CudaTetrahedronFEMForceFieldTempTextures
{
public:

    static __host__ void setElementForce(const void* /*x*/)
    {
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
	return CudaVec3<real>::make(x[i]);
    }
};

#if defined(USE_TEXTURE_X) && defined(USE_VEC4)

static texture<float4,1,cudaReadModeElementType> tex_4f_x;
static texture<float4,1,cudaReadModeElementType> tex_4f_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<float, CudaVec4<float> >
{
public:
    typedef float real;
    typedef CudaVec4<real> TIn;

    static __host__ void setX(const void* x)
    {
	static const void* cur = NULL;
	if (x!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_4f_x, x);
	    cur = x;
	}
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
	float4 x4 = tex1Dfetch(tex_4f_x, i);
	return CudaVec3<real>::make(x4);
    }

    static __host__ void setDX(const void* dx)
    {
	static const void* cur = NULL;
	if (dx!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_4f_dx, dx);
	    cur = dx;
	}
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
	float4 x4 = tex1Dfetch(tex_4f_dx, i);
	return CudaVec3<float>::make(x4);
    }
};

#endif

#if defined(USE_TEXTURE_X) && !defined(USE_VEC4)

static texture<float,1,cudaReadModeElementType> tex_3f_x;
static texture<float,1,cudaReadModeElementType> tex_3f_dx;

template<>
class CudaTetrahedronFEMForceFieldInputTextures<float, CudaVec3<float> >
{
public:
    typedef float real;
    typedef CudaVec3<real> TIn;

    static __host__ void setX(const void* x)
    {
	static const void* cur = NULL;
	if (x!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_3f_x, x);
	    cur = x;
	}
    }

    static __inline__ __device__ CudaVec3<real> getX(int i, const TIn* x)
    {
	int i3 = fastmul(i,3);
	float x1 = tex1Dfetch(tex_3f_x, i3);
	float x2 = tex1Dfetch(tex_3f_x, i3+1);
	float x3 = tex1Dfetch(tex_3f_x, i3+2);
	return CudaVec3<real>::make(x1,x2,x3);
    }

    static __host__ void setDX(const void* dx)
    {
	static const void* cur = NULL;
	if (dx!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_3f_dx, dx);
	    cur = dx;
	}
    }

    static __inline__ __device__ CudaVec3<real> getDX(int i, const TIn* dx)
    {
	int i3 = fastmul(i,3);
	float x1 = tex1Dfetch(tex_3f_dx, i3);
	float x2 = tex1Dfetch(tex_3f_dx, i3+1);
	float x3 = tex1Dfetch(tex_3f_dx, i3+2);
	return CudaVec3<real>::make(x1,x2,x3);
    }
};

#endif

#if defined(USE_TEXTURE_ELEMENT_FORCE)
/*
static texture<float,1,cudaReadModeElementType> tex_3f_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<float, CudaVec3<float> >
{
public:
    typedef float real;
    typedef CudaVec3<real> TIn;

    static __host__ void setElementForce(const void* x)
    {
	static const void* cur = NULL;
	if (x!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_3f_eforce, x);
	    cur = x;
	}
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
	int i3 = fastmul(i,3);
	float x1 = tex1Dfetch(tex_3f_eforce, i3);
	float x2 = tex1Dfetch(tex_3f_eforce, i3+1);
	float x3 = tex1Dfetch(tex_3f_eforce, i3+2);
	return CudaVec3<real>::make(x1,x2,x3);
    }
};
*/

static texture<float4,1,cudaReadModeElementType> tex_4f_eforce;

template<>
class CudaTetrahedronFEMForceFieldTempTextures<float, CudaVec4<float> >
{
public:
    typedef float real;
    typedef CudaVec4<real> TIn;

    static __host__ void setElementForce(const void* x)
    {
	static const void* cur = NULL;
	if (x!=cur)
	{
	    cudaBindTexture((size_t*)NULL, tex_4f_eforce, x);
	    cur = x;
	}
    }

    static __inline__ __device__ CudaVec3<real> getElementForce(int i, const TIn* x)
    {
	float4 x4 = tex1Dfetch(tex_4f_eforce, i);
	return CudaVec3<real>::make(x4);
    }
};

#endif


template<typename real, class TIn>
#ifndef ATOMIC_ADD_FORCE
__global__ void CudaTetrahedronFEMForceField3t_calcForce_kernel(int nbElem, const GPUElement<real>* elems, real* rotations, real* eforce, const TIn* x)
#else 
__global__ void CudaTetrahedronFEMForceField3t_calcForce_kernel(int nbElem, const GPUElement<real>* elems, real* rotations, const TIn* x,real * f)
#endif
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement<real> e = elems[index];
    //GPUElementState<real> s;
    const GPUElement<real>* e = elems + blockIdx.x;
    matrix3<real> Rt;
#ifdef USE_ROT6
    rotations += fastmul(index0,6)+index1;
#else
    rotations += fastmul(index0,9)+index1;
#endif
    //GPUElementForce<real> f;
    CudaVec3<real> fB,fC,fD;

#ifdef ATOMIC_ADD_FORCE
    int ia = e->ia[index1];
    int ib = e->ib[index1];
    int ic = e->ic[index1];
    int id = e->id[index1];
#endif

    if (index < nbElem)
    {
#ifndef ATOMIC_ADD_FORCE      
        CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ia[index1], x); //((const CudaVec3<real>*)x)[e.ia];
        CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ib[index1], x); //((const CudaVec3<real>*)x)[e.ib];
#else
	CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(ia, x); //((const CudaVec3<real>*)x)[e.ia];
	CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(ib, x); //((const CudaVec3<real>*)x)[e.ib];
#endif
        B -= A;

        // Compute R
        real bx = norm(B);
        Rt.x = B/bx;
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        CudaVec3<real> JtRtX0,JtRtX1;

        bx -= e->bx[index1];
        //                    ( bx)
        // RtB =              ( 0 )
        //                    ( 0 )
        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        real e_Jbx_bx = e->Jbx_bx[index1];
        real e_Jby_bx = e->Jby_bx[index1];
        real e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * bx;
        JtRtX0.y = 0;
        JtRtX0.z = 0;
        JtRtX1.x = e_Jby_bx * bx;
        JtRtX1.y = 0;
        JtRtX1.z = e_Jbz_bx * bx;
#ifndef ATOMIC_ADD_FORCE    
        CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->ic[index1], x); //((const CudaVec3<real>*)x)[e.ic];
#else
        CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(ic, x); //((const CudaVec3<real>*)x)[e.ic];
#endif	
        C -= A;
        Rt.z = cross(B,C);
        Rt.y = cross(Rt.z,B);
        Rt.y *= invnorm(Rt.y);
        Rt.z *= invnorm(Rt.z);

        real e_cx = e->cx[index1];
        real e_cy = e->cy[index1];
        real cx = Rt.mulX(C) - e_cx;
        real cy = Rt.mulY(C) - e_cy;
        //                    ( cx)
        // RtC =              ( cy)
        //                    ( 0 )
        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        real e_dy = e->dy[index1];
        real e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y += e_dz * cy;
        //JtRtX0.z += 0;
        JtRtX1.x += e_dz * cx;
        JtRtX1.y -= e_dy * cy;
        JtRtX1.z -= e_dy * cx;
#ifndef ATOMIC_ADD_FORCE    
        CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(e->id[index1], x); //((const CudaVec3<real>*)x)[e.id];
#else
        CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getX(id, x); //((const CudaVec3<real>*)x)[e.id];
#endif		
        D -= A;

        real e_dx = e->dx[index1];
        real dx = Rt.mulX(D) - e_dx;
        real dy = Rt.mulY(D) - e_dy;
        real dz = Rt.mulZ(D) - e_dz;
        //                    ( dx)
        // RtD =              ( dy)
        //                    ( dz)
        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z += e_cy * dz;
        //JtRtX1.x += 0;
        JtRtX1.y += e_cy * dy;
        JtRtX1.z += e_cy * dx;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        real e_mu2_bx2 = e->mu2_bx2[index1];
        CudaVec3<real> S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        CudaVec3<real> S1  = JtRtX1*(e_mu2_bx2*0.5f);

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(CudaVec3<real>::make(
                                                                             e_cy * S1.z,
                                                                e_cy * S1.y,
                                      e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(CudaVec3<real>::make(
            e_dz * S1.x - e_dy * S1.z,
            e_dz * S0.y - e_dy * S1.y,
            e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(CudaVec3<real>::make(
            e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                              e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                                                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
#ifdef USE_ROT6
    Rt.writeAoS6(rotations, BSIZE);
#else
    Rt.writeAoS(rotations, BSIZE);
#endif
    //((rmatrix3*)rotations)[index] = Rt;
    //((GPUElementForce<real>*)eforce)[index] = f;
#ifndef ATOMIC_ADD_FORCE
    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*13];
    int index13 = fastmul(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    real* out = ((real*)eforce)+(fastmul(blockIdx.x,BSIZE*16))+index1;
    real v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
#else
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 200
#ifndef ATOMIC_SHARED
    int ia3 = fastmul(ia,3);
    atomicAdd(f+ia3+0,fB.x+fC.x+fD.x);
    atomicAdd(f+ia3+1,fB.y+fC.y+fD.y);
    atomicAdd(f+ia3+2,fB.z+fC.z+fD.z);
    int ib3 = fastmul(ib,3);
    atomicAdd(f+ib3+0,-fB.x);
    atomicAdd(f+ib3+1,-fB.y);
    atomicAdd(f+ib3+2,-fB.z);
    int ic3 = fastmul(ic,3);
    atomicAdd(f+ic3+0,-fC.x);
    atomicAdd(f+ic3+1,-fC.y);
    atomicAdd(f+ic3+2,-fC.z);
    int id3 = fastmul(id,3);
    atomicAdd(f+id3+0,-fD.x);
    atomicAdd(f+id3+1,-fD.y);
    atomicAdd(f+id3+2,-fD.z);
#else 
    __shared__  real temp[BSIZE*12];
    __shared__  int gidx[BSIZE*12];
    int index12 = fastmul(index1,12);
    temp[index12+0 ] = (fB.x+fC.x+fD.x); gidx[index12+0 ] = fastmul(ia,3) + 0;
    temp[index12+1 ] = (fB.y+fC.y+fD.y); gidx[index12+1 ] = fastmul(ia,3) + 1;
    temp[index12+2 ] = (fB.z+fC.z+fD.z); gidx[index12+2 ] = fastmul(ia,3) + 2;
    temp[index12+3 ] = -fB.x;            gidx[index12+3 ] = fastmul(ib,3) + 0; 
    temp[index12+4 ] = -fB.y;            gidx[index12+4 ] = fastmul(ib,3) + 1; 
    temp[index12+5 ] = -fB.z;            gidx[index12+5 ] = fastmul(ib,3) + 2; 
    temp[index12+6 ] = -fC.x;            gidx[index12+6 ] = fastmul(ic,3) + 0; 
    temp[index12+7 ] = -fC.y;            gidx[index12+7 ] = fastmul(ic,3) + 1; 
    temp[index12+8 ] = -fC.z;            gidx[index12+8 ] = fastmul(ic,3) + 2; 
    temp[index12+9 ] = -fD.x;            gidx[index12+9 ] = fastmul(id,3) + 0; 
    temp[index12+10] = -fD.y;            gidx[index12+10] = fastmul(id,3) + 1; 
    temp[index12+11] = -fD.z;            gidx[index12+11] = fastmul(id,3) + 2; 
    __syncthreads();
    
    for (int i=index1;i<12*BSIZE;i+=BSIZE) atomicAdd(f + gidx[i],temp[i]);
#endif
#else 
    real * f0 = f + fastmul(ia,3);
    real * f1 = f + fastmul(ib,3);
    real * f2 = f + fastmul(ic,3);
    real * f3 = f + fastmul(id,3);

    // WARNING This code is incorrect but atomic operations on float are not supported for architecture < 2.0 
    f0[0] += -(fB.x+fC.x+fD.x);
    f0[1] += -(fB.y+fC.y+fD.y);
    f0[2] += -(fB.z+fC.z+fD.z);
    f1[0] += fB.x;
    f1[1] += fB.y;
    f1[2] += fB.z;
    f2[0] += fC.x;
    f2[1] += fC.y;
    f2[2] += fC.z;
    f3[0] += fD.x;
    f3[1] += fD.y;
    f3[2] += fD.z;
#endif
#endif
}

template<typename real, bool add, int BSIZE>
__global__ void CudaTetrahedronFEMForceField3t_addForce1_kernel(int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = fastmul(index1,3); //3*index1;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;
    
    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=fastmul(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
    for (int s = 0;s < nbElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
	    force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    temp[index3  ] = force.x;
    temp[index3+1] = force.y;
    temp[index3+2] = force.z;

    __syncthreads();

    if (add)
    {
        f[iext        ] += temp[index1        ];
        f[iext+  BSIZE] += temp[index1+  BSIZE];
        f[iext+2*BSIZE] += temp[index1+2*BSIZE];
    }
    else
    {
        f[iext        ] = temp[index1        ];
        f[iext+  BSIZE] = temp[index1+  BSIZE];
        f[iext+2*BSIZE] = temp[index1+2*BSIZE];
    }
}

template<typename real, bool add, int BSIZE>
__global__ void CudaTetrahedronFEMForceField3t_addForce4_kernel(int nbVertex, unsigned int nb4ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb4ElemPerVertex)+index1;

    //if (index0+index1 < (nbVertex<<2))
    for (int s = 0;s < nb4ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
	    force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    //int iout = (index1>>2)*3 + (index1&3)*((BSIZE/4)*3);
    int iout = fastmul((index1>>2) + ((index1&3)*(BSIZE/4)),3);
    temp[iout  ] = force.x;
    temp[iout+1] = force.y;
    temp[iout+2] = force.z;

    __syncthreads();

    // we need to merge 4 values together
    if (index1 < (BSIZE/4)*3)
    {

        real res = temp[index1] + temp[index1+ (BSIZE/4)*3] + temp[index1+ 2*(BSIZE/4)*3] + temp[index1+ 3*(BSIZE/4)*3];

        int iext = fastmul(blockIdx.x,(BSIZE/4)*3)+index1; //index0*3+index1;

        if (add)
        {
            f[iext] += res;
        }
        else
        {
            f[iext] = res;
        }
    }
}

template<typename real, bool add, int BSIZE>
__global__ void CudaTetrahedronFEMForceField3t_addForce8_kernel(int nbVertex, unsigned int nb8ElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Shared memory buffer to reorder global memory access
    __shared__  real temp[BSIZE/2*3];

    CudaVec3<real> force = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=(index0*nb8ElemPerVertex)+index1;

    //if (index0+index1 < (nbVertex<<2))
    for (int s = 0;s < nb8ElemPerVertex; s++)
    {
        int i = *velems -1;
        if (i == -1) break;
        velems+=BSIZE;
        //if (i != -1)
        {
	    force -= CudaTetrahedronFEMForceFieldTempTextures<real,CudaVec4<real> >::getElementForce(i,eforce);
        }
    }

    //int iout = (index1>>2)*3 + (index1&7)*((BSIZE/8)*3);
    int iout = fastmul((index1>>3) + ((index1&3)*(BSIZE/8)),3);
    if (index1&4)
    {
        temp[iout  ] = force.x;
        temp[iout+1] = force.y;
        temp[iout+2] = force.z;
    }
    __syncthreads();
    if (!(index1&4))
    {
        temp[iout  ] += force.x;
        temp[iout+1] += force.y;
        temp[iout+2] += force.z;
    }
    __syncthreads();

    if (index1 < (BSIZE/8)*3)
    {
        // we need to merge 4 values together
        real res = temp[index1] + temp[index1+ (BSIZE/8)*3] + temp[index1+ 2*(BSIZE/8)*3] + temp[index1+ 3*(BSIZE/8)*3];

        int iext = fastmul(blockIdx.x,(BSIZE/8)*3)+index1; //index0*3+index1;

        if (add)
        {
            f[iext] += res;
        }
        else
        {
            f[iext] = res;
        }
    }
}

template<typename real, class TIn>
#ifndef ATOMIC_ADD_FORCE
__global__ void CudaTetrahedronFEMForceField3t_calcDForce_kernel(int nbElem, const GPUElement<real>* elems, const real* rotations, real* eforce, const TIn* x, real factor)
#else
__global__ void CudaTetrahedronFEMForceField3t_calcDForce_kernel(int nbElem, const GPUElement<real>* elems, const real* rotations, const TIn* x, real factor , float * df)
#endif
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement<real> e = elems[index];
    const GPUElement<real>* e = elems + blockIdx.x;
    //GPUElementState<real> s = state[index];
    //GPUElementForce<real> f;
    CudaVec3<real> fB,fC,fD;
    matrix3<real> Rt;
#ifdef USE_ROT6
    rotations += fastmul(index0,6)+index1;
    Rt.readAoS6(rotations, BSIZE);
#else
    rotations += fastmul(index0,9)+index1;
    Rt.readAoS(rotations, BSIZE);
#endif
    //Rt = ((const rmatrix3*)rotations)[index];

#ifdef ATOMIC_ADD_FORCE
    int ia = e->ia[index1];
    int ib = e->ib[index1];
    int ic = e->ic[index1];
    int id = e->id[index1];
#endif    
    
    if (index < nbElem)
    {
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

#ifndef ATOMIC_ADD_FORCE   
        CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ia[index1], x); //((const CudaVec3<real>*)x)[e.ia];
	CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ib[index1], x); //((const CudaVec3<real>*)x)[e.ib];
#else 
        CudaVec3<real> A = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(ia, x); //((const CudaVec3<real>*)x)[e.ia];
	CudaVec3<real> B = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(ib, x); //((const CudaVec3<real>*)x)[e.ib];
#endif
	CudaVec3<real> JtRtX0,JtRtX1;
        B = Rt * (B-A);

        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        real e_Jbx_bx = e->Jbx_bx[index1];
        real e_Jby_bx = e->Jby_bx[index1];
        real e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x = e_Jbx_bx * B.x;
        JtRtX0.y =                  e_Jby_bx * B.y;
        JtRtX0.z =                                   e_Jbz_bx * B.z;
        JtRtX1.x = e_Jby_bx * B.x + e_Jbx_bx * B.y;
        JtRtX1.y =                  e_Jbz_bx * B.y + e_Jby_bx * B.z;
        JtRtX1.z = e_Jbz_bx * B.x                  + e_Jbx_bx * B.z;

#ifndef ATOMIC_ADD_FORCE   	
        CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->ic[index1], x); //((const CudaVec3<real>*)x)[e.ic];
#else
	CudaVec3<real> C = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(ic, x); //((const CudaVec3<real>*)x)[e.ic];
#endif
        C = Rt * (C-A);

        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        real e_dy = e->dy[index1];
        real e_dz = e->dz[index1];
        //JtRtX0.x += 0;
        JtRtX0.y +=              e_dz * C.y;
        JtRtX0.z +=                         - e_dy * C.z;
        JtRtX1.x += e_dz * C.x;
        JtRtX1.y +=            - e_dy * C.y + e_dz * C.z;
        JtRtX1.z -= e_dy * C.x;

        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
#ifndef ATOMIC_ADD_FORCE   	
        CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(e->id[index1], x); //((const CudaVec3<real>*)x)[e.id];
#else
	CudaVec3<real> D = CudaTetrahedronFEMForceFieldInputTextures<real,TIn>::getDX(id, x); //((const CudaVec3<real>*)x)[e.id];
#endif
        D = Rt * (D-A);

        real e_cy = e->cy[index1];
        //JtRtX0.x += 0;
        //JtRtX0.y += 0;
        JtRtX0.z +=                           e_cy * D.z;
        //JtRtX1.x += 0;
        JtRtX1.y +=              e_cy * D.y;
        JtRtX1.z += e_cy * D.x;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        real e_mu2_bx2 = e->mu2_bx2[index1];
        CudaVec3<real> S0  = JtRtX0*e_mu2_bx2;
        S0 += (JtRtX0.x+JtRtX0.y+JtRtX0.z)*e->gamma_bx2[index1];
        CudaVec3<real> S1  = JtRtX1*(e_mu2_bx2*0.5f);

	S0 *= factor;
	S1 *= factor;

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.mulT(CudaVec3<real>::make(
            e_cy * S1.z,
            e_cy * S1.y,
            e_cy * S0.z)));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.mulT(CudaVec3<real>::make(
            e_dz * S1.x - e_dy * S1.z,
            e_dz * S0.y - e_dy * S1.y,
            e_dz * S1.y - e_dy * S0.z)));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.mulT(CudaVec3<real>::make(
            e_Jbx_bx * S0.x                                     + e_Jby_bx * S1.x                   + e_Jbz_bx * S1.z,
                              e_Jby_bx * S0.y                   + e_Jbx_bx * S1.x + e_Jbz_bx * S1.y,
                                                e_Jbz_bx * S0.z                   + e_Jby_bx * S1.y + e_Jbx_bx * S1.z)));
        //fA.x = -(fB.x+fC.x+fD.x);
        //fA.y = -(fB.y+fC.y+fD.y);
        //fA.z = -(fB.z+fC.z+fD.z);
    }

    //state[index] = s;
    //((GPUElementForce<real>*)eforce)[index] = f;

#ifndef ATOMIC_ADD_FORCE     
    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*13];
    int index13 = fastmul(index1,13);
    temp[index13+0 ] = -(fB.x+fC.x+fD.x);
    temp[index13+1 ] = -(fB.y+fC.y+fD.y);
    temp[index13+2 ] = -(fB.z+fC.z+fD.z);
    temp[index13+3 ] = fB.x;
    temp[index13+4 ] = fB.y;
    temp[index13+5 ] = fB.z;
    temp[index13+6 ] = fC.x;
    temp[index13+7 ] = fC.y;
    temp[index13+8 ] = fC.z;
    temp[index13+9 ] = fD.x;
    temp[index13+10] = fD.y;
    temp[index13+11] = fD.z;
    __syncthreads();
    real* out = ((real*)eforce)+(fastmul(blockIdx.x,BSIZE*16))+index1;
    real v = 0;
    bool read = true; //(index1&4)<3;
    index1 += (index1>>4) - (index1>>2); // remove one for each 4-values before this thread, but add an extra one each 16 threads (so each 12 input cells, to align to 13)

    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
    if (read) v = temp[index1]; index1 += (BSIZE+(BSIZE>>4)-(BSIZE>>2));
    *out = v; out += BSIZE;
#else
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 200
#ifndef ATOMIC_SHARED
    int ia3 = fastmul(ia,3);
    atomicAdd(df+ia3+0,fB.x+fC.x+fD.x);
    atomicAdd(df+ia3+1,fB.y+fC.y+fD.y);
    atomicAdd(df+ia3+2,fB.z+fC.z+fD.z);
    int ib3 = fastmul(ib,3);
    atomicAdd(df+ib3+0,-fB.x);
    atomicAdd(df+ib3+1,-fB.y);
    atomicAdd(df+ib3+2,-fB.z);
    int ic3 = fastmul(ic,3);
    atomicAdd(df+ic3+0,-fC.x);
    atomicAdd(df+ic3+1,-fC.y);
    atomicAdd(df+ic3+2,-fC.z);
    int id3 = fastmul(id,3);
    atomicAdd(df+id3+0,-fD.x);
    atomicAdd(df+id3+1,-fD.y);
    atomicAdd(df+id3+2,-fD.z);
#else 
    __shared__  real temp[BSIZE*12];
    __shared__  int gidx[BSIZE*12];
    int index12 = fastmul(index1,12);
    temp[index12+0 ] = (fB.x+fC.x+fD.x); gidx[index12+0 ] = fastmul(ia,3) + 0;
    temp[index12+1 ] = (fB.y+fC.y+fD.y); gidx[index12+1 ] = fastmul(ia,3) + 1;
    temp[index12+2 ] = (fB.z+fC.z+fD.z); gidx[index12+2 ] = fastmul(ia,3) + 2;
    temp[index12+3 ] = -fB.x;            gidx[index12+3 ] = fastmul(ib,3) + 0; 
    temp[index12+4 ] = -fB.y;            gidx[index12+4 ] = fastmul(ib,3) + 1; 
    temp[index12+5 ] = -fB.z;            gidx[index12+5 ] = fastmul(ib,3) + 2; 
    temp[index12+6 ] = -fC.x;            gidx[index12+6 ] = fastmul(ic,3) + 0; 
    temp[index12+7 ] = -fC.y;            gidx[index12+7 ] = fastmul(ic,3) + 1; 
    temp[index12+8 ] = -fC.z;            gidx[index12+8 ] = fastmul(ic,3) + 2; 
    temp[index12+9 ] = -fD.x;            gidx[index12+9 ] = fastmul(id,3) + 0; 
    temp[index12+10] = -fD.y;            gidx[index12+10] = fastmul(id,3) + 1; 
    temp[index12+11] = -fD.z;            gidx[index12+11] = fastmul(id,3) + 2; 
    __syncthreads();
    
    for (int i=index1;i<12*BSIZE;i+=BSIZE) atomicAdd(df + gidx[i],temp[i]);
#endif    
#else 
    real * df0 = df + fastmul(ia,3);
    real * df1 = df + fastmul(ib,3);
    real * df2 = df + fastmul(ic,3);
    real * df3 = df + fastmul(id,3);
    
    // WARNING This code is incorrect but atomic operations on float are not supported for architecture < 2.0 
    df0[0] += -(fB.x+fC.x+fD.x);
    df0[1] += -(fB.y+fC.y+fD.y);
    df0[2] += -(fB.z+fC.z+fD.z);
    df1[0] += fB.x;
    df1[1] += fB.y;
    df1[2] += fB.z;
    df2[0] += fC.x;
    df2[1] += fC.y;
    df2[2] += fC.z;
    df3[0] += fD.x;
    df3[1] += fD.y;
    df3[2] += fD.z;
#endif
#endif
}

template<class real>
__global__ void CudaTetrahedronFEMForceField3t_copyVec4(int size, CudaVec4<real>* x4, const CudaVec3<real>* x)
{
    //__shared__ real temp[BSIZE*3];
    int index0 = fastmul(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0 + index1;
    //if (index < size)
    {
	//res[index] = b[index] * f;
	x4[index] = CudaVec4<real>::make(x[index]);
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

template<typename real, bool add, int BSIZE>
void CudaTetrahedronFEMForceField3t_addForce_launch3(int PT, const dim3& grid, const dim3& threads, int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    if (PT == 1)
        CudaTetrahedronFEMForceField3t_addForce1_kernel<real,add,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, eforce, velems, f);
    else if (PT == 4)
        CudaTetrahedronFEMForceField3t_addForce4_kernel<real,add,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, eforce, velems, f);
    else if (PT == 8)
        CudaTetrahedronFEMForceField3t_addForce8_kernel<real,add,BSIZE><<< grid, threads >>>(nbVertex, nbElemPerVertex, eforce, velems, f);
}

template<typename real, bool add>
void CudaTetrahedronFEMForceField3t_addForce_launch2(int PT, int BSIZE, const dim3& grid, const dim3& threads, int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    switch(BSIZE)
    {
    case  32: CudaTetrahedronFEMForceField3t_addForce_launch3<real,add, 32>(PT, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f); break;
    case  64: CudaTetrahedronFEMForceField3t_addForce_launch3<real,add, 64>(PT, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f); break;
    case 128: CudaTetrahedronFEMForceField3t_addForce_launch3<real,add,128>(PT, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f); break;
    case 256: CudaTetrahedronFEMForceField3t_addForce_launch3<real,add,256>(PT, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f); break;
    }
}

template<typename real>
void CudaTetrahedronFEMForceField3t_addForce_launch1(bool add, int PT, int BSIZE, const dim3& grid, const dim3& threads, int nbVertex, unsigned int nbElemPerVertex, const CudaVec4<real>* eforce, const int* velems, real* f)
{
    if (add)
        CudaTetrahedronFEMForceField3t_addForce_launch2<real, true>(PT, BSIZE, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f);
    else
        CudaTetrahedronFEMForceField3t_addForce_launch2<real, false>(PT, BSIZE, grid, threads, nbVertex, nbElemPerVertex, eforce, velems, f);
}

const void* cur_x4 = NULL;
const void* cur_dx4 = NULL;

void CudaTetrahedronFEMForceField3f_prepareX(unsigned int size, void* x4, const void* x)
{
#ifdef USE_VEC4
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    CudaTetrahedronFEMForceField3t_copyVec4<float><<< grid, threads >>>(size, (CudaVec4<float>*) x4, (const CudaVec3<float>*) x);
    cur_x4 = x4;
#endif
}

void CudaTetrahedronFEMForceField3f_prepareDx(unsigned int size, void* dx4, const void* dx)
{
#ifdef USE_VEC4
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    CudaTetrahedronFEMForceField3t_copyVec4<float><<< grid, threads >>>(size, (CudaVec4<float>*) dx4, (const CudaVec3<float>*) dx);
    cur_dx4 = dx4;
#endif
}

void CudaTetrahedronFEMForceField3f_addForce(unsigned int nbElem, unsigned int nbVertex, bool add,
                                             const void* elems, void* state,
                                             void* f, const void* x,
                                             unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE,
                                             void* eforce, const void* velems)
{
#ifdef USE_VEC4
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec4<float> >::setX(cur_x4);
#else
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setX(x);
#endif
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);  
  
#ifndef ATOMIC_ADD_FORCE  
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
#ifdef USE_VEC4
    CudaTetrahedronFEMForceField3t_calcForce_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (float*)state, (float*)eforce, (const CudaVec4<float>*)cur_x4);
#else
    CudaTetrahedronFEMForceField3t_calcForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (float*)state, (float*)eforce, (const CudaVec3<float>*)x);
#endif
    dim3 threads2(addForce_BSIZE,1);
    dim3 grid2((nbVertex*addForce_PT+addForce_BSIZE-1)/addForce_BSIZE,1);
    nbElemPerVertex = (nbElemPerVertex + addForce_PT-1)/addForce_PT;
    CudaTetrahedronFEMForceField3t_addForce_launch1<float>(add, addForce_PT, addForce_BSIZE, grid2, threads2, nbVertex, nbElemPerVertex, (const CudaVec4<float>*)eforce, (const int*)velems, (float*)f);
#else
    if (! add) mycudaMemset(f,0,nbVertex * 3 * sizeof(float));//memset because all threads accumulate on the array
   
    CudaTetrahedronFEMForceField3t_calcForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (float*)state, (const CudaVec3<float>*)x,(float*)f);
#endif    
}

void CudaTetrahedronFEMForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, bool add, double factor,
                                              const void* elems, const void* state,
                                              void* df, const void* dx,
                                              unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE,
                                              void* eforce, const void* velems)
{
#ifdef USE_VEC4
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec4<float> >::setDX(cur_dx4);
#else 
    CudaTetrahedronFEMForceFieldInputTextures<float,CudaVec3<float> >::setDX(dx);
#endif
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);

#ifndef ATOMIC_ADD_FORCE      
    CudaTetrahedronFEMForceFieldTempTextures<float,CudaVec4<float> >::setElementForce(eforce);
#ifdef USE_VEC4
    CudaTetrahedronFEMForceField3t_calcDForce_kernel<float, CudaVec4<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (const float*)state, (float*)eforce, (const CudaVec4<float>*)cur_dx4, (float) factor);
#else
    CudaTetrahedronFEMForceField3t_calcDForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (const float*)state, (float*)eforce, (const CudaVec3<float>*)dx, (float) factor);
#endif
    dim3 threads2(addForce_BSIZE,1);
    dim3 grid2((nbVertex*addForce_PT+addForce_BSIZE-1)/addForce_BSIZE,1);
    nbElemPerVertex = (nbElemPerVertex + addForce_PT-1)/addForce_PT;
    CudaTetrahedronFEMForceField3t_addForce_launch1<float>(add, addForce_PT, addForce_BSIZE, grid2, threads2, nbVertex, nbElemPerVertex, (const CudaVec4<float>*)eforce, (const int*)velems, (float*)df);
#else
    if (! add) mycudaMemset(df,0,nbVertex * 3 * sizeof(float));//memset because all threads accumulate on the array

    CudaTetrahedronFEMForceField3t_calcDForce_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const GPUElement<float>*)elems, (const float*)state, (const CudaVec3<float>*)dx, (float) factor, (float*)df);
#endif    
}
