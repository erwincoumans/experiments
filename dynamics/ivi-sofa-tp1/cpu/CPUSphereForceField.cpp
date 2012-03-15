#include "../kernels.h"

void CPUSphereForceField3f_addForce( unsigned int size, GPUSphere<float>* sphere, TReal* penetration, TDeriv* f, const TCoord* x, const TDeriv* v )
{
    const TDeriv sphere_center ( sphere->center_x, sphere->center_y, sphere->center_z );
    const TReal sphere_radius = sphere->radius;

    // TODO: force computation
}

void CPUSphereForceField3f_addDForce( unsigned int size, GPUSphere<float>* sphere, const TReal* penetration, TDeriv* f, const TDeriv* dx )
{
    const TDeriv sphere_center ( sphere->center_x, sphere->center_y, sphere->center_z );
    const TReal sphere_radius = sphere->radius;

    // TODO: dforce computation
}
