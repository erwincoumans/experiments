#include "../kernels.h"
#include <string.h>

void CPUMechanicalObject3f_vClear( unsigned int size, TDeriv* res )
{
    //memset(res, 0, size*sizeof(TDeriv));
    for (unsigned int i=0;i<size;++i)
        res[i].clear();
}

void CPUMechanicalObject3f_vEqBF( unsigned int size, TDeriv* res, const TDeriv* b, float f )
{
    for (unsigned int i=0;i<size;++i)
        res[i] = b[i]*f;
}

void CPUMechanicalObject3f_vPEqBF( unsigned int size, TDeriv* res, const TDeriv* b, float f )
{
    for (unsigned int i=0;i<size;++i)
        res[i] += b[i]*f;
}

void CPUMechanicalObject3f_vOp( unsigned int size, TDeriv* res, const TDeriv* a, const TDeriv* b, float f )
{
    for (unsigned int i=0;i<size;++i)
        res[i] = a[i] + b[i]*f;
}

void CPUMechanicalObject3f_vIntegrate( unsigned int size, const TDeriv* a, TDeriv* v, TCoord* x, float h )
{
    for (unsigned int i=0;i<size;++i)
    {
        v[i] += a[i]*h;
        x[i] += v[i]*h;
    }
}

void CPUMechanicalObject3f_vPEq1( unsigned int size, TDeriv* res, int index, const float* val )
{
    res[index] += TDeriv(val);
}

#ifdef PARALLEL_REDUCTION
int CPUMechanicalObject3f_vDotTmpSize( unsigned int size )
{
    return 0;
}
#endif

void CPUMechanicalObject3f_vDot( unsigned int size, float* res
                                           , const TDeriv* a, const TDeriv* b
#ifdef PARALLEL_REDUCTION
                                           , TReal* tmp, float* cputmp
#endif
)
{
    float sum = 0.0f;
    for (unsigned int i=0;i<size;++i)
        sum += a[i]*b[i];
    *res = sum;
}
