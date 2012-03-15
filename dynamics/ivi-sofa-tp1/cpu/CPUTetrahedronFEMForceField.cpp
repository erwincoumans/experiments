#include "../kernels.h"

void CPUTetrahedronFEMForceField3f_addForce( unsigned int nbElem, unsigned int nbVertex, bool add
                                           , const GPUElement<TReal>* elems, GPUElementRotation<TReal>* state
                                           , TDeriv* f, const TCoord* x
#ifdef PARALLEL_GATHER
                                           , unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE
                                           , GPUElementForce<TReal>* eforce, const int* velems
#endif
)
{
    if (!add)
        for (unsigned int i=0;i<nbVertex;++i)
            f[i].clear();
    const int index1 = 0;
    for (unsigned int i=0;i<nbElem;++i)
    {
        const GPUElement<TReal>* e = elems + i;
        sofa::defaulttype::Mat<3,3,TReal> Rt;
        TDeriv fB,fC,fD;

        TDeriv A = x[e->ia[index1]];
        TDeriv B = x[e->ib[index1]];
        B -= A;

        // Compute R
        TReal bx = B.norm();
        Rt.x() = B/bx;
        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        TDeriv JtRtX0,JtRtX1;

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
        TReal e_Jbx_bx = e->Jbx_bx[index1];
        TReal e_Jby_bx = e->Jby_bx[index1];
        TReal e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x() = e_Jbx_bx * bx;
        JtRtX0.y() = 0;
        JtRtX0.z() = 0;
        JtRtX1.x() = e_Jby_bx * bx;
        JtRtX1.y() = 0;
        JtRtX1.z() = e_Jbz_bx * bx;

        TDeriv C = x[e->ic[index1]];
        C -= A;
        Rt.z() = cross(B,C);
        Rt.y() = cross(Rt.z(),B);
        Rt.y() *= 1/Rt.y().norm();
        Rt.z() *= 1/Rt.z().norm();

        TReal e_cx = e->cx[index1];
        TReal e_cy = e->cy[index1];
        TReal cx = Rt.x() * C - e_cx;
        TReal cy = Rt.y() * C - e_cy;
        //                    ( cx)
        // RtC =              ( cy)
        //                    ( 0 )
        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        TReal e_dy = e->dy[index1];
        TReal e_dz = e->dz[index1];
        //JtRtX0.x() += 0;
        JtRtX0.y() += e_dz * cy;
        //JtRtX0.z() += 0;
        JtRtX1.x() += e_dz * cx;
        JtRtX1.y() -= e_dy * cy;
        JtRtX1.z() -= e_dy * cx;

        TDeriv D = x[e->id[index1]];
        D -= A;

        TReal e_dx = e->dx[index1];
        TReal dx = Rt.x() * D - e_dx;
        TReal dy = Rt.y() * D - e_dy;
        TReal dz = Rt.z() * D - e_dz;
        //                    ( dx)
        // RtD =              ( dy)
        //                    ( dz)
        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        //JtRtX0.x() += 0;
        //JtRtX0.y() += 0;
        JtRtX0.z() += e_cy * dz;
        //JtRtX1.x() += 0;
        JtRtX1.y() += e_cy * dy;
        JtRtX1.z() += e_cy * dx;

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        TReal e_mu2_bx2 = e->mu2_bx2[index1];
        TDeriv S0  = JtRtX0*e_mu2_bx2;
        TReal s0 = (JtRtX0.x()+JtRtX0.y()+JtRtX0.z())*e->gamma_bx2[index1];
        S0.x() += s0;  S0.y() += s0;  S0.z() += s0;
        TDeriv S1  = JtRtX1*(e_mu2_bx2*0.5f);

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.multTranspose(TDeriv(
                                                                             e_cy * S1.z(),
                                                                e_cy * S1.y(),
                                      e_cy * S0.z())));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.multTranspose(TDeriv(
            e_dz * S1.x() - e_dy * S1.z(),
            e_dz * S0.y() - e_dy * S1.y(),
            e_dz * S1.y() - e_dy * S0.z())));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.multTranspose(TDeriv(
            e_Jbx_bx * S0.x()                                     + e_Jby_bx * S1.x()                   + e_Jbz_bx * S1.z(),
                              e_Jby_bx * S0.y()                   + e_Jbx_bx * S1.x() + e_Jbz_bx * S1.y(),
                                                e_Jbz_bx * S0.z()                   + e_Jby_bx * S1.y() + e_Jbx_bx * S1.z())));
        //fA.x() = -(fB.x()+fC.x()+fD.x());
        //fA.y() = -(fB.y()+fC.y()+fD.y());
        //fA.z() = -(fB.z()+fC.z()+fD.z());
        f[e->ia[index1]] += (fB+fC+fD);
        f[e->ib[index1]] -= fB;
        f[e->ic[index1]] -= fC;
        f[e->id[index1]] -= fD;

#ifdef USE_ROT6
        state[i].rx[0][0] = Rt.x().x();
        state[i].rx[1][0] = Rt.x().y();
        state[i].rx[2][0] = Rt.x().z();
        state[i].ry[0][0] = Rt.y().x();
        state[i].ry[1][0] = Rt.y().y();
        state[i].ry[2][0] = Rt.y().z();
#else
        *(sofa::defaulttype::Mat<3,3,TReal>*)&state[i] = Rt;
#endif
    }
}

void CPUTetrahedronFEMForceField3f_addDForce( unsigned int nbElem, unsigned int nbVertex, bool add, double factor
                                            , const GPUElement<TReal>* elems, const GPUElementRotation<TReal>* state
                                            , TDeriv* df, const TDeriv* dx
#ifdef PARALLEL_GATHER
                                            , unsigned int nbElemPerVertex, int addForce_PT, int addForce_BSIZE
                                            , GPUElementForce<TReal>* eforce, const int* velems
#endif
)
{
    if (!add)
        for (unsigned int i=0;i<nbVertex;++i)
            df[i].clear();

    const int index1 = 0;
    for (unsigned int i=0;i<nbElem;++i)
    {
        const GPUElement<TReal>* e = elems + i;
#ifdef USE_ROT6
        sofa::defaulttype::Mat<3,3,TReal> Rt;
        Rt.x().x() = state[i].rx[0][0];
        Rt.x().y() = state[i].rx[0][1];
        Rt.x().z() = state[i].rx[0][2];
        Rt.y().x() = state[i].ry[0][0];
        Rt.y().y() = state[i].ry[0][1];
        Rt.y().z() = state[i].ry[0][2];
        Rt.z() = cross(Rt.x(), Rt.y());
#else
        sofa::defaulttype::Mat<3,3,TReal> Rt = *(sofa::defaulttype::Mat<3,3,TReal>*)&state[i];
#endif
        TDeriv fB,fC,fD;

        // Compute JtRtX = JbtRtB + JctRtC + JdtRtD

        TDeriv A = dx[e->ia[index1]];
        TDeriv JtRtX0,JtRtX1;

        TDeriv B = dx[e->ib[index1]];
        B = Rt * (B-A);

        // Jtb = (Jbx  0   0 )
        //       ( 0  Jby  0 )
        //       ( 0   0  Jbz)
        //       (Jby Jbx  0 )
        //       ( 0  Jbz Jby)
        //       (Jbz  0  Jbx)
        TReal e_Jbx_bx = e->Jbx_bx[index1];
        TReal e_Jby_bx = e->Jby_bx[index1];
        TReal e_Jbz_bx = e->Jbz_bx[index1];
        JtRtX0.x() = e_Jbx_bx * B.x();
        JtRtX0.y() =                  e_Jby_bx * B.y();
        JtRtX0.z() =                                   e_Jbz_bx * B.z();
        JtRtX1.x() = e_Jby_bx * B.x() + e_Jbx_bx * B.y();
        JtRtX1.y() =                  e_Jbz_bx * B.y() + e_Jby_bx * B.z();
        JtRtX1.z() = e_Jbz_bx * B.x()                  + e_Jbx_bx * B.z();

        TDeriv C = dx[e->ic[index1]];
        C = Rt * (C-A);

        // Jtc = ( 0   0   0 )
        //       ( 0   dz  0 )
        //       ( 0   0  -dy)
        //       ( dz  0   0 )
        //       ( 0  -dy  dz)
        //       (-dy  0   0 )
        TReal e_dy = e->dy[index1];
        TReal e_dz = e->dz[index1];
        //JtRtX0.x() += 0;
        JtRtX0.y() +=              e_dz * C.y();
        JtRtX0.z() +=                         - e_dy * C.z();
        JtRtX1.x() += e_dz * C.x();
        JtRtX1.y() +=            - e_dy * C.y() + e_dz * C.z();
        JtRtX1.z() -= e_dy * C.x();

        // Jtd = ( 0   0   0 )
        //       ( 0   0   0 )
        //       ( 0   0   cy)
        //       ( 0   0   0 )
        //       ( 0   cy  0 )
        //       ( cy  0   0 )
        TDeriv D = dx[e->id[index1]];
        D = Rt * (D-A);

        TReal e_cy = e->cy[index1];
        //JtRtX0.x() += 0;
        //JtRtX0.y() += 0;
        JtRtX0.z() +=                           e_cy * D.z();
        //JtRtX1.x() += 0;
        JtRtX1.y() +=              e_cy * D.y();
        JtRtX1.z() += e_cy * D.x();

        // Compute S = K JtRtX

        // K = [ gamma+mu2 gamma gamma 0 0 0 ]
        //     [ gamma gamma+mu2 gamma 0 0 0 ]
        //     [ gamma gamma gamma+mu2 0 0 0 ]
        //     [ 0 0 0             mu2/2 0 0 ]
        //     [ 0 0 0             0 mu2/2 0 ]
        //     [ 0 0 0             0 0 mu2/2 ]
        // S0 = JtRtX0*mu2 + dot(JtRtX0,(gamma gamma gamma))
        // S1 = JtRtX1*mu2/2

        TReal e_mu2_bx2 = e->mu2_bx2[index1];
        TDeriv S0 = JtRtX0*e_mu2_bx2;
        TReal s0 = (JtRtX0.x()+JtRtX0.y()+JtRtX0.z())*e->gamma_bx2[index1];
        S0.x() += s0;  S0.y() += s0;  S0.z() += s0;
        TDeriv S1  = JtRtX1*(e_mu2_bx2*0.5f);

        S0 *= factor;
        S1 *= factor;

        // Jd = ( 0   0   0   0   0  cy )
        //      ( 0   0   0   0  cy   0 )
        //      ( 0   0   cy  0   0   0 )
        fD = (Rt.multTranspose(TDeriv(
            e_cy * S1.z(),
            e_cy * S1.y(),
            e_cy * S0.z())));
        // Jc = ( 0   0   0  dz   0 -dy )
        //      ( 0   dz  0   0 -dy   0 )
        //      ( 0   0  -dy  0  dz   0 )
        fC = (Rt.multTranspose(TDeriv(
            e_dz * S1.x() - e_dy * S1.z(),
            e_dz * S0.y() - e_dy * S1.y(),
            e_dz * S1.y() - e_dy * S0.z())));
        // Jb = (Jbx  0   0  Jby  0  Jbz)
        //      ( 0  Jby  0  Jbx Jbz  0 )
        //      ( 0   0  Jbz  0  Jby Jbx)
        fB = (Rt.multTranspose(TDeriv(
            e_Jbx_bx * S0.x()                                     + e_Jby_bx * S1.x()                   + e_Jbz_bx * S1.z(),
                              e_Jby_bx * S0.y()                   + e_Jbx_bx * S1.x() + e_Jbz_bx * S1.y(),
                                                e_Jbz_bx * S0.z()                   + e_Jby_bx * S1.y() + e_Jbx_bx * S1.z())));
        //fA.x() = -(fB.x()+fC.x()+fD.x());
        //fA.y() = -(fB.y()+fC.y()+fD.y());
        //fA.z() = -(fB.z()+fC.z()+fD.z());
        df[e->ia[index1]] += (fB+fC+fD);
        df[e->ib[index1]] -= fB;
        df[e->ic[index1]] -= fC;
        df[e->id[index1]] -= fD;
    }
}
