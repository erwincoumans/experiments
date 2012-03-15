#ifndef KERNELS_H
#define KERNELS_H

#include "simulation.h"

#if !defined(VERSION) || VERSION >= 5
// FLAG to activate linear algebra kernels merging optimization
#define MERGE_CG_KERNELS
#endif
#if !defined(VERSION) || VERSION >= 6
// FLAG to reduction kernels merging optimization
#define MERGE_REDUCTION_KERNELS
#endif

#ifdef PARALLEL_GATHER

// FEM add force: number of threads per point
#define GATHER_PT 4
//#define GATHER_PT 1

// FEM add force: number of threads per bloc
//#define GATHER_BSIZE 32
#define GATHER_BSIZE 64
//#define GATHER_BSIZE 128
//#define GATHER_BSIZE 256

#endif

#if defined( SOFA_DEVICE_CUDA )
#define DEVICE_METHOD(name) sofa_concat(Cuda,name)
#define DEVICE_PTR(T) void*
#elif defined( SOFA_DEVICE_CPU )
#define DEVICE_METHOD(name) sofa_concat(CPU,name)
#define DEVICE_PTR(T) T*
#endif

bool kernels_init();

//// EXTERNAL CUDA KERNELS ////

extern "C" // TetrahedronFEMForceField
{
#ifdef USE_VEC4
void DEVICE_METHOD(TetrahedronFEMForceField3f_prepareX)(unsigned int nbVertex, DEVICE_PTR(TCoord4) x4, const DEVICE_PTR(TCoord) x);
void DEVICE_METHOD(TetrahedronFEMForceField3f_prepareDx)(unsigned int nbVertex, DEVICE_PTR(TCoord4) dx4, const DEVICE_PTR(TCoord) dx);
#endif

void DEVICE_METHOD(TetrahedronFEMForceField3f_addForce)( unsigned int nbElem, unsigned int nbVertex, bool add
    , const DEVICE_PTR(GPUElement<TReal>) elems, DEVICE_PTR(GPUElementRotation<TReal>) state
    , DEVICE_PTR(TDeriv) f, const DEVICE_PTR(TCoord) x
#ifdef PARALLEL_GATHER
    , unsigned int nbElemPerVertex, int gather_PT, int gather_BSIZE
    , DEVICE_PTR(GPUElementForce<TReal>) eforce, const DEVICE_PTR(int) velems
#endif
);
void DEVICE_METHOD(TetrahedronFEMForceField3f_addDForce)( unsigned int nbElem, unsigned int nbVertex, bool add, double factor
    , const DEVICE_PTR(GPUElement<TReal>) elems, const DEVICE_PTR(GPUElementRotation<TReal>) state
    , DEVICE_PTR(TDeriv) df, const DEVICE_PTR(TDeriv) dx
#ifdef PARALLEL_GATHER
    , unsigned int nbElemPerVertex, int gather_PT, int gather_BSIZE
    , DEVICE_PTR(GPUElementForce<TReal>) eforce, const DEVICE_PTR(int) velems
#endif
);
}

extern "C" // MergedKernels
{
#if defined(MERGE_REDUCTION_KERNELS)
#ifdef PARALLEL_REDUCTION
int DEVICE_METHOD(MergedKernels3f_cgDot3TmpSize)( unsigned int size );
#endif
// d.q, r.q, q.q
void DEVICE_METHOD(MergedKernels3f_cgDot3)( unsigned int size, float* dot3
    , const DEVICE_PTR(TDeriv) r, const DEVICE_PTR(TDeriv) q, const DEVICE_PTR(TDeriv) d
#ifdef PARALLEL_REDUCTION
    , DEVICE_PTR(TReal) tmp, float* cputmp
#endif
);
// b.q, b.q, q.q
void DEVICE_METHOD(MergedKernels3f_cgDot3First)( unsigned int size, float* dot3
    , const DEVICE_PTR(TDeriv) b, const DEVICE_PTR(TDeriv) q
#ifdef PARALLEL_REDUCTION
    , DEVICE_PTR(TReal) tmp, float* cputmp
#endif
);
// a = a + alpha d, r = r - alpha q, d = r + beta d
void DEVICE_METHOD(MergedKernels3f_cgOp3)( unsigned int size, float alpha, float beta
    , DEVICE_PTR(TDeriv) r, DEVICE_PTR(TDeriv) a, DEVICE_PTR(TDeriv) d, const DEVICE_PTR(TDeriv) q
);
// a = alpha b, r = b - alpha q, d = r + beta b
void DEVICE_METHOD(MergedKernels3f_cgOp3First)( unsigned int size, float alpha, float beta
    , DEVICE_PTR(TDeriv) r, DEVICE_PTR(TDeriv) a, DEVICE_PTR(TDeriv) d, const DEVICE_PTR(TDeriv) q
    , const DEVICE_PTR(TDeriv) b
);
#elif defined(MERGE_CG_KERNELS)
#ifdef PARALLEL_REDUCTION
int DEVICE_METHOD(MergedKernels3f_cgDeltaTmpSize)( unsigned int size );
#endif
void DEVICE_METHOD(MergedKernels3f_cgDelta)( bool first, unsigned int size, float* delta, float alpha
    , DEVICE_PTR(TDeriv) r, DEVICE_PTR(TDeriv) a, const DEVICE_PTR(TDeriv) q, const DEVICE_PTR(TDeriv) d
#ifdef PARALLEL_REDUCTION
    , DEVICE_PTR(TReal) tmp, float* cputmp
#endif
);
#endif
}

extern "C" // MechanicalObject
{
void DEVICE_METHOD(MechanicalObject3f_vClear)( unsigned int size, DEVICE_PTR(TDeriv) res );
void DEVICE_METHOD(MechanicalObject3f_vEqBF)( unsigned int size, DEVICE_PTR(TDeriv) res, const DEVICE_PTR(TDeriv) b, float f );
void DEVICE_METHOD(MechanicalObject3f_vPEqBF)( unsigned int size, DEVICE_PTR(TDeriv) res, const DEVICE_PTR(TDeriv) b, float f );
void DEVICE_METHOD(MechanicalObject3f_vOp)( unsigned int size, DEVICE_PTR(TDeriv) res, const DEVICE_PTR(TDeriv) a, const DEVICE_PTR(TDeriv) b, float f );
void DEVICE_METHOD(MechanicalObject3f_vIntegrate)( unsigned int size, const DEVICE_PTR(TDeriv) a, DEVICE_PTR(TDeriv) v, DEVICE_PTR(TCoord) x, float h );
void DEVICE_METHOD(MechanicalObject3f_vPEq1)( unsigned int size, DEVICE_PTR(TDeriv) res, int index, const float* val );
#ifdef PARALLEL_REDUCTION
int DEVICE_METHOD(MechanicalObject3f_vDotTmpSize)( unsigned int size );
#endif
void DEVICE_METHOD(MechanicalObject3f_vDot)( unsigned int size, float* res
    , const DEVICE_PTR(TDeriv) a, const DEVICE_PTR(TDeriv) b
#ifdef PARALLEL_REDUCTION
    , DEVICE_PTR(TReal) tmp, float* cputmp
#endif
);
}

extern "C" // UniformMass
{
void DEVICE_METHOD(UniformMass3f_addMDx)( unsigned int size, float mass, DEVICE_PTR(TDeriv) res, const DEVICE_PTR(TDeriv) dx );
void DEVICE_METHOD(UniformMass3f_accFromF)( unsigned int size, float mass, DEVICE_PTR(TDeriv) a, const DEVICE_PTR(TDeriv) f );
void DEVICE_METHOD(UniformMass3f_addForce)( unsigned int size, const float *mg, DEVICE_PTR(TDeriv) f );
}

extern "C" // FixedConstraint
{
void DEVICE_METHOD(FixedConstraint3f_projectResponseIndexed)( unsigned int size, const DEVICE_PTR(int) indices, DEVICE_PTR(TDeriv) dx );
}

extern "C" // PlaneForceField
{
void DEVICE_METHOD(PlaneForceField3f_addForce)( unsigned int size, GPUPlane<float>* plane, DEVICE_PTR(TReal) penetration, DEVICE_PTR(TDeriv) f, const DEVICE_PTR(TCoord) x, const DEVICE_PTR(TDeriv) v );
void DEVICE_METHOD(PlaneForceField3f_addDForce)( unsigned int size, GPUPlane<float>* plane, const DEVICE_PTR(TReal) penetration, DEVICE_PTR(TDeriv) f, const DEVICE_PTR(TDeriv) dx );
}

extern "C" // SphereForceField
{
void DEVICE_METHOD(SphereForceField3f_addForce)( unsigned int size, GPUSphere<float>* sphere, DEVICE_PTR(TReal) penetration, DEVICE_PTR(TDeriv) f, const DEVICE_PTR(TCoord) x, const DEVICE_PTR(TDeriv) v );
void DEVICE_METHOD(SphereForceField3f_addDForce)( unsigned int size, GPUSphere<float>* sphere, const DEVICE_PTR(TReal) penetration, DEVICE_PTR(TDeriv) f, const DEVICE_PTR(TDeriv) dx );
}

extern "C" // TetraMapper
{
void DEVICE_METHOD(TetraMapper3f_apply)( unsigned int size, const DEVICE_PTR(TTetra) map_i, const DEVICE_PTR(TCoord4) map_f, DEVICE_PTR(TDeriv) out, const DEVICE_PTR(TDeriv) in );
}

extern "C" // VisualModel
{
void DEVICE_METHOD(VisualModel3f_calcTNormals)( unsigned int nbElem, unsigned int nbVertex, const DEVICE_PTR(TTriangle) elems, DEVICE_PTR(TDeriv) fnormals, const DEVICE_PTR(TDeriv) x );
#ifdef PARALLEL_GATHER
void DEVICE_METHOD(VisualModel3f_calcVNormals)( unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const DEVICE_PTR(int) velems, DEVICE_PTR(TDeriv) vnormals, const DEVICE_PTR(TDeriv) fnormals, const DEVICE_PTR(TDeriv) x );
#endif
void DEVICE_METHOD(VisualModel3f_calcNormals)( unsigned int nbElem, unsigned int nbVertex
    , const DEVICE_PTR(TTriangle) elems, const DEVICE_PTR(TDeriv) x
    , DEVICE_PTR(TDeriv) fnormals, DEVICE_PTR(TDeriv) vnormals
#ifdef PARALLEL_GATHER
    , unsigned int nbElemPerVertex, const DEVICE_PTR(int) velems
#endif
);

void DEVICE_METHOD(VisualModel3f_calcTNormalsAndTangents)( unsigned int nbElem, unsigned int nbVertex, const DEVICE_PTR(TTriangle) elems, DEVICE_PTR(TDeriv) fnormals, DEVICE_PTR(TDeriv) ftangents, const DEVICE_PTR(TDeriv) x, const DEVICE_PTR(TTexCoord) tc );
#ifdef PARALLEL_GATHER
void DEVICE_METHOD(VisualModel3f_calcVNormalsAndTangents)( unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const DEVICE_PTR(int) velems, DEVICE_PTR(TDeriv) vnormals, DEVICE_PTR(TDeriv) vtangents, const DEVICE_PTR(TDeriv) fnormals, const DEVICE_PTR(TDeriv) ftangents, const DEVICE_PTR(TDeriv) x, const DEVICE_PTR(TTexCoord) tc );
#endif
void DEVICE_METHOD(VisualModel3f_calcNormalsAndTangents)( unsigned int nbElem, unsigned int nbVertex
    , const DEVICE_PTR(TTriangle) elems, const DEVICE_PTR(TDeriv) x, const DEVICE_PTR(TTexCoord) tc
    , DEVICE_PTR(TDeriv) fnormals, DEVICE_PTR(TDeriv) ftangents, DEVICE_PTR(TDeriv) vnormals, DEVICE_PTR(TDeriv) vtangents
#ifdef PARALLEL_GATHER
    , unsigned int nbElemPerVertex, const DEVICE_PTR(int) velems
#endif
);
}

#endif
