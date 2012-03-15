#include "../kernels.h"

void CPUPlaneForceField3f_addForce( unsigned int size, GPUPlane<float>* plane, TReal* penetration, TDeriv* f, const TCoord* x, const TDeriv* v )
{
    const TDeriv plane_normal ( plane->normal_x, plane->normal_y, plane->normal_z );
    const TReal plane_d = plane->d;

    for (unsigned int i=0;i<size;++i)
    {
        TDeriv xi = x[i];
        TReal d = dot(xi, plane_normal) - plane_d;
        penetration[i] = d;
        if (d < 0)
        {
            TReal forceIntensity = -plane->stiffness*d;
            TReal dampingIntensity = -plane->damping*d;
            f[i] += plane_normal*forceIntensity - v[i]*dampingIntensity;
        }
    }
}

void CPUPlaneForceField3f_addDForce( unsigned int size, GPUPlane<float>* plane, const TReal* penetration, TDeriv* f, const TDeriv* dx )
{
    const TDeriv plane_normal ( plane->normal_x, plane->normal_y, plane->normal_z );

    for (unsigned int i=0;i<size;++i)
    {
        TReal d = penetration[i];
        if (d < 0)
        {
            f[i] += plane_normal * -plane->stiffness*dot(plane_normal,dx[i]);
        }
    }
}
