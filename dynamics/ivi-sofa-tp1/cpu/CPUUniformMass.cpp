#include "../kernels.h"

void CPUUniformMass3f_addMDx( unsigned int size, float mass, TDeriv* res, const TDeriv* dx )
{
    for (unsigned int i=0;i<size;++i)
        res[i] += dx[i] * mass;
}

void CPUUniformMass3f_accFromF( unsigned int size, float mass, TDeriv* a, const TDeriv* f )
{
    TReal inv_mass = 1.0f / mass;
    for (unsigned int i=0;i<size;++i)
        a[i] = f[i] * inv_mass;
}

void CPUUniformMass3f_addForce( unsigned int size, const float *mg, TDeriv* f )
{
    const TDeriv v(mg);
    for (unsigned int i=0;i<size;++i)
        f[i] += v;
}
