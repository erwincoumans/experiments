#include "simulation.h"
#include "kernels.h"

#include <mesh/read_mesh_netgen.h>

#include <iostream>


//// DATA ////

double simulation_time = 0;
TCoord simulation_bbox[2];
TCoord simulation_center;
double simulation_size;
TCoord plane_position;
double plane_size;
TCoord sphere_position0;
TCoord sphere_position;
TDeriv sphere_velocity;
double sphere_radius;

extern int verbose;

FEMMesh* fem_mesh = NULL;

//// INIT METHODS ////

bool simulation_preload()
{
    if (!kernels_init()) return false;
    return true;
}

bool simulation_load_fem_mesh(const char* filename)
{
    FEMMesh* mesh = new FEMMesh;
    if (!read_mesh_netgen(filename, mesh->positions, mesh->tetrahedra, mesh->triangles))
    {
        delete mesh;
        return false;
    }
    mesh->filename = filename;

    if (mesh && mesh->positions.size() > 0)
    {
        mesh->bbox[0] = mesh->positions[0];
        mesh->bbox[1] = mesh->positions[0];
        for (unsigned int i=1;i<mesh->positions.size();++i)
        {
            TCoord p = mesh->positions[i];
            for (unsigned int c=0;c<p.size();++c)
                if (p[c] < mesh->bbox[0][c]) mesh->bbox[0][c] = p[c];
                else if (p[c] > mesh->bbox[1][c]) mesh->bbox[1][c] = p[c];
        }
    }

    fem_mesh = mesh;
    return true;
}

void simulation_reorder_fem_mesh()
{
    if (fem_mesh)
        fem_mesh->reorder();
}

bool simulation_init()
{
    FEMMesh* mesh = fem_mesh;

    if (mesh)
    {
        simulation_bbox[0] = mesh->bbox[0];
        simulation_bbox[1] = mesh->bbox[1];
    }
    simulation_size = (simulation_bbox[1]-simulation_bbox[0]).norm();
    simulation_center = (simulation_bbox[0] + simulation_bbox[1]) * 0.5f;
    

    plane_position = simulation_center * 0.5f;
    plane_position[1] = simulation_bbox[0][1];
    plane_size = simulation_size*0.75f;

    sphere_position0[0] = simulation_bbox[1][0] + simulation_size*0.5f;
    sphere_position0[1] = simulation_center[1];
    sphere_position0[2] = simulation_bbox[0][2]*0.33f + simulation_bbox[1][2]*0.67f;
    sphere_position = sphere_position0;
    sphere_velocity[0] = -simulation_size*0.1f;
    sphere_radius = simulation_size*0.05f;

    if (mesh)
    {
        mesh->init(&simulation_params);

        for (unsigned int i = 0; i < render_meshes.size(); ++i)
            render_meshes[i]->init(mesh);
    }

    return true;
}

extern bool simulation_mapping_needed;

void simulation_reset()
{
    FEMMesh* mesh = fem_mesh;
    sphere_position = sphere_position0;
    mesh->reset();
    mesh->setPushForce(&simulation_params);
    mesh->update(&simulation_params);
    simulation_mapping_needed = true;
    simulation_time = 0;
}

void simulation_save()
{
    FEMMesh* mesh = fem_mesh;
    if (simulation_time)
        mesh->save(mesh->filename + ".state");
    simulation_mapping();
    std::string suffix = (simulation_time ? "-deformed" : "-initial");
    for (unsigned int i = 0; i < render_meshes.size(); ++i)
    {
        std::string filename(render_meshes[i]->filename, 0, render_meshes[i]->filename.size()-4);
        render_meshes[i]->saveObj(filename + suffix + ".obj", filename + suffix + ".mtl");
    }
    {
        std::string filename(mesh->filename, 0, mesh->filename.size()-5);
        mesh->saveObj(filename + suffix + ".obj", filename + suffix + ".mtl");
    }
}

void simulation_load()
{
    FEMMesh* mesh = fem_mesh;
    mesh->load(mesh->filename + ".state");
    simulation_mapping_needed = true;
    simulation_time = 1;
}

//// INTERNAL METHODS ////

template<class T1, class T2>
bool SortPairFirstFn(const std::pair<T1,T2>& a, const std::pair<T1,T2>& b) { return a.first < b.first; }

void FEMMesh::reorder()
{
    // simple reordering of the vertices using the largest dimension of the mesh
    int sort_coord = 0;
    if (bbox[1][1]-bbox[0][1] > bbox[1][sort_coord]-bbox[0][sort_coord])
        sort_coord = 1;
    if (bbox[1][2]-bbox[0][2] > bbox[1][sort_coord]-bbox[0][sort_coord])
        sort_coord = 2;
    std::cout << "Reordering particles based on " << (char)('X'+sort_coord) << std::endl;
    std::vector< std::pair<TReal,int> > sortp;
    sortp.resize(positions.size());
    for (unsigned int i=0;i<positions.size();++i)
        sortp[i] = std::make_pair(positions[i][sort_coord], i);
    std::sort(sortp.begin(),sortp.end(),SortPairFirstFn<TReal,int>);
    std::vector<int> old2newpos;
    old2newpos.resize(positions.size());
    for (unsigned int i=0;i<positions.size();++i)
        old2newpos[sortp[i].second] = i;
    TVecCoord newpos;
    newpos.resize(positions.size());
    for (unsigned int i=0;i<positions.size();++i)
        newpos[i] = positions[sortp[i].second];
    positions.swap(newpos);
    for (unsigned int i=0;i<tetrahedra.size();++i)
        for (unsigned int j=0;j<tetrahedra[i].size();++j)
            tetrahedra[i][j] = old2newpos[tetrahedra[i][j]];
    for (unsigned int i=0;i<triangles.size();++i)
        for (unsigned int j=0;j<triangles[i].size();++j)
            triangles[i][j] = old2newpos[triangles[i][j]];
    
    std::cout << "Reordering tetrahedra based on connected particles" << std::endl;
    std::vector< std::pair<int,int> > sortt;
    sortt.resize(tetrahedra.size());
    for (unsigned int i=0;i<tetrahedra.size();++i)
        sortt[i] = std::make_pair(std::min(std::min(tetrahedra[i][0],tetrahedra[i][1]),std::min(tetrahedra[i][2],tetrahedra[i][3])), i);
    std::sort(sortt.begin(),sortt.end(),SortPairFirstFn<int,int>);
    TVecTetra newtetra;
    newtetra.resize(tetrahedra.size());
    for (unsigned int i=0;i<tetrahedra.size();++i)
        newtetra[i] = tetrahedra[sortt[i].second];
    tetrahedra.swap(newtetra);
    std::cout << "Mesh reordering done" << std::endl;
}

bool FEMMesh::isFixedParticle(int index) const
{
    if (nbFixedParticles == 0) return false;
    // we use a bitmask instead of a indices vector to more easily search for the particle
    if (fixedMask.empty()) return false;
    int mi = index / 32;
    int mb = index % 32;
    if (fixedMask[mi] & (1 << mb)) return true;
    return false;

    //for (unsigned int i=0;i<fixedParticles.size();++i)
    //    if (fixedParticles[i] == index) return true;
    //return false;
}

void FEMMesh::addFixedParticle(int index)
{
    // for merged kernels we use a bitmask instead of a indices vector
    if (fixedMask.empty())
        fixedMask.resize((positions.size()+31) / 32);
    int mi = index / 32;
    int mb = index % 32;
    if (fixedMask[mi] & (1 << mb)) return; // already fixed
    fixedMask[mi] |= (1 << mb);

    // for standard kernels we use an indices vector
    fixedParticles.push_back(index);

    ++nbFixedParticles;
}

void FEMMesh::removeFixedParticle(int index)
{
    // for merged kernels we use a bitmask instead of a indices vector
    if (fixedMask.empty()) return;
    int mi = index / 32;
    int mb = index % 32;
    if (!(fixedMask[mi] & (1 << mb))) return; // not fixed
    fixedMask[mi] &= ~(1 << mb);

    // for standard kernels we use an indices vector
    unsigned int i;
    for (i=0;i<fixedParticles.size();++i)
        if (fixedParticles[i] == index) break;
    if (i < fixedParticles.size())
    {
        if (i < fixedParticles.size()-1) // move to last
            fixedParticles[i] = fixedParticles[fixedParticles.size()-1];
        // and remove last
        fixedParticles.resize(fixedParticles.size()-1);
    }

    --nbFixedParticles;
}

void FEMMesh::init(SimulationParameters* params)
{
    if (positions0.size() != positions.size())
    {
        positions0 = positions;
        velocity.resize(positions.size());
    }

    const TVecCoord& x0 = positions0;

    // Plane
    plane.normal_x = 0;
    plane.normal_y = 1;
    plane.normal_z = 0;
    plane.d = plane_position[1];
    plane.stiffness = (TReal)params->planeRepulsion;
    plane.damping = 0;
    std::cout << "Plane d = " << plane.d << " stiffness = " << plane.stiffness << " damping = " << plane.damping << std::endl;

    // Fixed

    fixedParticles.clear();
    fixedMask.clear();
    nbFixedParticles = 0;
    if (params->fixedHeight > 0)
    {
        TReal maxY = (TReal)(bbox[0][1] + (bbox[1][1]-bbox[0][1]) * params->fixedHeight);
        TReal maxZ = (TReal)(bbox[0][2] + (bbox[1][2]-bbox[0][2]) * 0.75f);
        std::cout << "Fixed box = " << bbox[0] << "    " << TCoord(bbox[1][0],maxY,maxZ) << std::endl;
        for (unsigned int i=0;i<positions.size();++i)
            if (x0[i][1] <= maxY && x0[i][2] <= maxZ)
            {
                //fixedParticles.push_back(i);
                addFixedParticle(i);
                positions[i] = x0[i];
                if (!velocity.empty())
                    velocity[i].clear();
            }
    }

    // Push Force
    setPushForce(params);

    // FEM

    const int nbp = positions.size();
    const int nbe = tetrahedra.size();
    const int nbBe = (nbe + BSIZE-1)/BSIZE;
    femElem.resize(nbBe);
    femElemRotation.resize(nbBe);
#ifdef PARALLEL_GATHER
    femElemForce.resize(nbe);
#endif
    // FEM matrices
    for (int eindex = 0; eindex < nbe; ++eindex)
    {
        TTetra& tetra = tetrahedra[eindex];
        //std::cout << "Elem " << eindex << " : indices = " << tetra << std::endl;
        TCoord A = x0[tetra[0]];
        TCoord B = x0[tetra[1]]-A;
        TCoord C = x0[tetra[2]]-A;
        TCoord D = x0[tetra[3]]-A;
        TReal vol36 = 6*dot( cross(B, C), D );
        if (vol36<0)
        {
            std::cerr << "ERROR: Negative volume for tetra "<<eindex<<" <"<<A<<','<<B<<','<<C<<','<<D<<"> = "<<vol36/36<<std::endl;
            vol36 *= -1;
            TCoord tmp = C; C = D; D = tmp;
            int itmp = tetra[2]; tetra[2] = tetra[3]; tetra[3] = itmp;
        }
        sofa::defaulttype::Mat<3,3,TReal> Rt;
        Rt[0] = B;
        Rt[2] = cross( B, C );
        Rt[1] = cross( Rt[2], B );
        Rt[0].normalize();
        Rt[1].normalize();
        Rt[2].normalize();
        TCoord b = Rt * B;
        TCoord c = Rt * C;
        TCoord d = Rt * D;
        //std::cout << "Elem " << eindex << " : b = " << b << "  c = " << c << "  d = " << d << std::endl;
        TReal y = (A[1] + (B[1]+C[1]+D[1])*0.25f - bbox[0][1])/(bbox[1][1]-bbox[0][1]);
        TReal youngModulus = tetraYoungModulus(eindex, params);
        TReal poissonRatio = tetraPoissonRatio(eindex, params);
        TReal gamma = (youngModulus*poissonRatio) / ((1+poissonRatio)*(1-2*poissonRatio));
        TReal mu2 = youngModulus / (1+poissonRatio);
        // divide by 36 times vol of the element
        gamma /= vol36;
        mu2 /= vol36;

        TReal bx2 = b[0] * b[0];

        const int block  = eindex / BSIZE;
        const int thread = eindex % BSIZE;
        GPUElement<TReal>& e = femElem[block];
        e.ia[thread] = tetra[0];
        e.ib[thread] = tetra[1];
        e.ic[thread] = tetra[2];
        e.id[thread] = tetra[3];
        e.bx[thread] = b[0];
        e.cx[thread] = c[0]; e.cy[thread] = c[1];
        e.dx[thread] = d[0]; e.dy[thread] = d[1]; e.dz[thread] = d[2];
        e.gamma_bx2[thread] = gamma * bx2;
        e.mu2_bx2[thread] = mu2 * bx2;
        e.Jbx_bx[thread] = (c[1] * d[2]) / b[0];
        e.Jby_bx[thread] = (-c[0] * d[2]) / b[0];
        e.Jbz_bx[thread] = (c[0]*d[1] - c[1]*d[0]) / b[0];
    }
#ifdef PARALLEL_GATHER
    // elements <-> particles table
    // first find number of elements per particle
    std::vector<int> p_nbe;
    p_nbe.resize(nbp);
    for (int eindex = 0; eindex < nbe; ++eindex)
        for (int j = 0; j < 4; ++j)
            ++p_nbe[tetrahedra[eindex][j]];
    // then compute max value
    nbElemPerVertex = 0;
    for (int i=0;i<nbp;++i)
        if (p_nbe[i] > nbElemPerVertex) nbElemPerVertex = p_nbe[i];
#if GATHER_PT > 1
    // we will create group of GATHER_PT elements
    int nbElemPerThread = (nbElemPerVertex+GATHER_PT-1)/GATHER_PT;
    const int nbBpt = (nbp*GATHER_PT + GATHER_BSIZE-1)/GATHER_BSIZE;
    // finally fill velems array
    femVElems.resize(nbBpt*nbElemPerThread*GATHER_BSIZE);
#else
    const int nbBp = (nbp + GATHER_BSIZE-1)/GATHER_BSIZE;
    // finally fill velems array
    femVElems.resize(nbBp*nbElemPerVertex*GATHER_BSIZE);
#endif
    p_nbe.clear();
    p_nbe.resize(nbp);
    for (int eindex = 0; eindex < nbe; ++eindex)
        for (int j = 0; j < 4; ++j)
        {
            int p = tetrahedra[eindex][j];
            int num = p_nbe[p]++;
#if GATHER_PT > 1
            const int block  = (p*GATHER_PT) / GATHER_BSIZE;
            const int thread = (p*GATHER_PT+(num%GATHER_PT)) % GATHER_BSIZE;
            num = num/GATHER_PT;
            femVElems[ block * (nbElemPerThread * GATHER_BSIZE) +
                       num * GATHER_BSIZE + thread ] = 1 + eindex * 4 + j;
#else
            const int block  = p / GATHER_BSIZE;
            const int thread = p % GATHER_BSIZE;
            femVElems[ block * (nbElemPerVertex * BSIZE) +
                       num * BSIZE + thread ] = 1 + eindex * 4 + j;
#endif
        }
#endif
    std::cout << "FEM init done: " << positions.size() << " particles, " << tetrahedra.size() << " elements";
#ifdef PARALLEL_GATHER
    std::cout << ", up to " << nbElemPerVertex << " elements on each particle";
#endif
    std::cout << "." << std::endl;
    update(params);
}


void FEMMesh::update(SimulationParameters* params)
{

    // Sphere
    sphere.center_x = sphere_position[0];
    sphere.center_y = sphere_position[1];
    sphere.center_z = sphere_position[2];
    sphere.velocity_x = sphere_velocity[0];
    sphere.velocity_y = sphere_velocity[1];
    sphere.velocity_z = sphere_velocity[2];
    sphere.radius = sphere_radius;
    sphere.stiffness = (TReal)params->sphereRepulsion;
    sphere.damping = 0;
    //std::cout << "sphere r = " << sphere.radius << " stiffness = " << sphere.stiffness << " damping = " << sphere.damping << std::endl;
}

void FEMMesh::setPushForce(SimulationParameters* params)
{
    // find the most forward point in front of the mesh
    {
        int best = -1;
        TReal bestz = bbox[0][2];
        TReal minx = bbox[0][0] * 0.6f + bbox[1][0] * 0.4f;
        TReal maxx = bbox[0][0] * 0.4f + bbox[1][0] * 0.6f;
        for (unsigned int i=0;i<positions0.size();++i)
        {
            TCoord x = positions0[i];
            if (x[2] > bestz && x[0] >= minx && x[0] <= maxx)
            {
                best = i;
                bestz = x[2];
            }
        }
        externalForce.index = best;
        externalForce.value = params->pushForce * simulation_size;
    }
}

void FEMMesh::reset()
{
    positions = positions0;
    velocity.clear();
    velocity.resize(positions.size());
}

bool FEMMesh::save(const std::string& filename)
{
    std::ofstream out(filename.c_str());
    if (!out)
    {
        std::cerr << "Cannot write to file " << filename << std::endl;
        return false;
    }
    out << positions.size() << " " << 6 << std::endl;
    for (unsigned int i=0;i<positions.size();++i)
    {
        out << positions[i] << " " << velocity[i] << "\n";
    }
    out.flush();
    out.close();
    return true;
}

bool FEMMesh::load(const std::string& filename)
{
    std::ifstream in(filename.c_str());
    if (!in)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }
    int nbp = 0, nbc = 0;
    in >> nbp >> nbc;
    if (nbp != positions.size())
    {
        std::cerr << "ERROR: file " << filename << " contains " << nbp << " vertices while the mesh contains " << positions.size() << std::endl;
        return false;
    }
    if (nbc != 6)
    {
        std::cerr << "ERROR: file " << filename << " contains " << nbc << " values instead of 6" << std::endl;
        return false;
    }
    for (unsigned int i=0;i<positions.size();++i)
    {
        in >> positions[i] >> velocity[i];
    }
    in.close();
    return true;
}

//// SIMULATION METHODS ////

struct MechanicalMatrix
{
    double mFactor, kFactor;
    MechanicalMatrix(): mFactor(0), kFactor(0) {}
};

template<class TVec>
void showDebug(const TVec& v, const char* name)
{
    std::cout << name << " =";
    if (v.size() < 10) std::cout << ' ' << v;
    else
    {
        for (unsigned int i=0;i<5;++i) std::cout << ' ' << v[i];
        std::cout << " ...";
        for (unsigned int i=v.size()-5;i<v.size();++i) std::cout << ' ' << v[i];
    }
    std::cout << std::endl;
}

// Compute b = f
void computeForce(const SimulationParameters* params, FEMMesh* mesh, TVecDeriv& result)
{
    const unsigned int size = mesh->positions.size();
    const double mass = params->massDensity;
    const TDeriv mg = params->gravity * mass;
    const TVecCoord& x = mesh->positions;
    const TVecDeriv& v = mesh->velocity;
    TVecDeriv& f = result;
    result.recreate(size);

    // it is no longer necessary to clear the result vector as the addForce
    // kernel from TetrahedronFEMForceField will do an assignement instead of
    // an addition
    if (params->youngModulusTop == 0 && params->youngModulusBottom == 0)
        DEVICE_METHOD(MechanicalObject3f_vClear)( size, result.deviceWrite() );

    // Internal Forces
    // result = f(x,v)
    if (params->youngModulusTop != 0 || params->youngModulusBottom != 0)
    {
#ifdef USE_VEC4
        mesh->x4.fastResize(size);
        DEVICE_METHOD(TetrahedronFEMForceField3f_prepareX)(size, mesh->x4.deviceWrite(), x.deviceRead());
#endif
        DEVICE_METHOD(TetrahedronFEMForceField3f_addForce)( mesh->tetrahedra.size(), size, false
                                                            , mesh->femElem.deviceRead(), mesh->femElemRotation.deviceWrite()
                                                            , result.deviceWrite(), x.deviceRead()
#ifdef PARALLEL_GATHER
                                                            , mesh->nbElemPerVertex, GATHER_PT, GATHER_BSIZE
                                                            , mesh->femElemForce.deviceWrite(), mesh->femVElems.deviceRead()
#endif
        );
    }
    
    // External forces
    if (mesh->externalForce.index >= 0)
    {
        //result[mesh->externalForce.index] += mesh->externalForce.value;
        DEVICE_METHOD(MechanicalObject3f_vPEq1)( size, result.deviceWrite(), mesh->externalForce.index, mesh->externalForce.value.ptr() );
    }

    if (mesh->plane.stiffness != 0)
    {
        mesh->planePenetration.recreate(size);
        DEVICE_METHOD(PlaneForceField3f_addForce)( size, &mesh->plane, mesh->planePenetration.deviceWrite(), result.deviceWrite(), x.deviceRead(), v.deviceRead() );
    }

    if (mesh->sphere.stiffness != 0)
    {
        mesh->spherePenetration.recreate(size);
        DEVICE_METHOD(SphereForceField3f_addForce)( size, &mesh->sphere, mesh->spherePenetration.deviceWrite(), result.deviceWrite(), x.deviceRead(), v.deviceRead() );
    }

    // Gravity
    DEVICE_METHOD(UniformMass3f_addForce)( size, mg.ptr(), result.deviceWrite() );
}

void applyConstraints(const SimulationParameters* /*params*/, FEMMesh* mesh, TVecDeriv& result)
{
    if (mesh->nbFixedParticles > 0)
    {
        DEVICE_METHOD(FixedConstraint3f_projectResponseIndexed)( mesh->fixedParticles.size(), mesh->fixedParticles.deviceRead(), result.deviceWrite() );
    }
}

// Compute a = M^-1 f
void accFromF(const SimulationParameters* params, FEMMesh* mesh, const TVecDeriv& f)
{
    const unsigned int size = mesh->positions.size();
    const double mass = params->massDensity;
    TVecDeriv& a = mesh->a;
    a.recreate(size);
    DEVICE_METHOD(UniformMass3f_addMDx)( size, mass, a.deviceWrite(), f.deviceRead() );
}

// Compute b += kFactor * K * v
void addKv(const SimulationParameters* params, FEMMesh* mesh, double kFactor)
{
    const unsigned int size = mesh->positions.size();
    const double mass = params->massDensity;
    const TDeriv mg = params->gravity * mass;
    const TVecCoord& x = mesh->positions;
    const TVecDeriv& v = mesh->velocity;
    TVecDeriv& b = mesh->b;

    // b += kFactor * K * v
    if (params->youngModulusTop != 0 || params->youngModulusBottom != 0)
    {
#ifdef USE_VEC4
        mesh->dx4.fastResize(size);
        DEVICE_METHOD(TetrahedronFEMForceField3f_prepareDx)(size, mesh->dx4.deviceWrite(), v.deviceRead());
#endif
        DEVICE_METHOD(TetrahedronFEMForceField3f_addDForce)( mesh->tetrahedra.size(), size, true, kFactor
                                                             , mesh->femElem.deviceRead(), mesh->femElemRotation.deviceRead()
                                                             , b.deviceWrite(), v.deviceRead()
#ifdef PARALLEL_GATHER
                                                             , mesh->nbElemPerVertex, GATHER_PT, GATHER_BSIZE
                                                             , mesh->femElemForce.deviceWrite(), mesh->femVElems.deviceRead()
#endif
        );
    }

    if (mesh->plane.stiffness != 0)
    {
        GPUPlane<TReal> plane2 = mesh->plane;
        plane2.stiffness *= (TReal)kFactor;
        DEVICE_METHOD(PlaneForceField3f_addDForce)( size, &plane2, mesh->planePenetration.deviceRead(), b.deviceWrite(), v.deviceRead() );
    }

    if (mesh->sphere.stiffness != 0)
    {
        GPUSphere<TReal> sphere2 = mesh->sphere;
        sphere2.stiffness *= (TReal)kFactor;
        DEVICE_METHOD(SphereForceField3f_addDForce)( size, &sphere2, mesh->spherePenetration.deviceRead(), b.deviceWrite(), v.deviceRead() );
    }

    if (mesh->nbFixedParticles > 0)
    {
        DEVICE_METHOD(FixedConstraint3f_projectResponseIndexed)( mesh->fixedParticles.size(), mesh->fixedParticles.deviceRead(), b.deviceWrite() );
    }

    //std::cout << "b = " << b << std::endl;
    if (verbose >= 2) showDebug(b, "b");

}

void mulMatrixVector(const SimulationParameters* params, FEMMesh* mesh, MechanicalMatrix matrix, TVecDeriv& result, const TVecDeriv& input)
{
    const unsigned int size = mesh->positions.size();
    const double mass = params->massDensity;

//    std::cout << "matrix = " << matrix.mFactor << " * " << mass << " = " << matrix.mFactor * mass << std::endl;

    // it is no longer necessary to clear the result vector as the addDForce
    // kernel from TetrahedronFEMForceField will do an assignement instead of
    // an addition
    if (params->youngModulusTop == 0 && params->youngModulusBottom == 0)
        DEVICE_METHOD(MechanicalObject3f_vClear)( size, result.deviceWrite() );

    if (params->youngModulusTop != 0 || params->youngModulusBottom != 0)
    {
#ifdef USE_VEC4
        mesh->dx4.fastResize(size);
        DEVICE_METHOD(TetrahedronFEMForceField3f_prepareDx)(size, mesh->dx4.deviceWrite(), input.deviceRead());
#endif
        DEVICE_METHOD(TetrahedronFEMForceField3f_addDForce)( mesh->tetrahedra.size(), size, false, matrix.kFactor
                                                             , mesh->femElem.deviceRead(), mesh->femElemRotation.deviceRead()
                                                             , result.deviceWrite(), input.deviceRead()
#ifdef PARALLEL_GATHER
                                                             , mesh->nbElemPerVertex, GATHER_PT, GATHER_BSIZE
                                                             , mesh->femElemForce.deviceWrite(), mesh->femVElems.deviceRead()
#endif
        );
    }

    DEVICE_METHOD(UniformMass3f_addMDx)( size, matrix.mFactor * mass, result.deviceWrite(), input.deviceRead() );

    if (mesh->plane.stiffness != 0)
    {
        GPUPlane<TReal> plane2 = mesh->plane;
        plane2.stiffness *= matrix.kFactor;
        DEVICE_METHOD(PlaneForceField3f_addDForce)( size, &plane2, mesh->planePenetration.deviceRead(), result.deviceWrite(), input.deviceRead() );
    }

    if (mesh->sphere.stiffness != 0)
    {
        GPUSphere<TReal> sphere2 = mesh->sphere;
        sphere2.stiffness *= matrix.kFactor;
        DEVICE_METHOD(SphereForceField3f_addDForce)( size, &sphere2, mesh->spherePenetration.deviceRead(), result.deviceWrite(), input.deviceRead() );
    }

    if (mesh->nbFixedParticles > 0)
    {
        DEVICE_METHOD(FixedConstraint3f_projectResponseIndexed)( mesh->fixedParticles.size(), mesh->fixedParticles.deviceRead(), result.deviceWrite() );
    }
}

int simulation_cg_iter = 0;

void linearSolver_ConjugateGradient(const SimulationParameters* params, FEMMesh* mesh, MechanicalMatrix matrix)
{
    const unsigned int size = mesh->positions.size();
    const double mass = params->massDensity;
    const int maxIter = params->maxIter;
    const double tolerance = params->tolerance;
    const TVecDeriv& b = mesh->b;
    TVecDeriv& a = mesh->a;
    TVecDeriv& q = mesh->q;
    TVecDeriv& d = mesh->d;
    TVecDeriv& r = mesh->r;
    a.recreate(size);
    q.recreate(size);
    d.recreate(size);
    r.recreate(size);

    // for parallel reductions (vDot)
#ifdef PARALLEL_REDUCTION
    TVecReal& tmp = mesh->dottmp;
    int tmpsize = std::max(
        DEVICE_METHOD(MechanicalObject3f_vDotTmpSize)( size ),
#if defined(MERGE_REDUCTION_KERNELS)
        DEVICE_METHOD(MergedKernels3f_cgDot3TmpSize)( size )
#elif defined(MERGE_CG_KERNELS)
        DEVICE_METHOD(MergedKernels3f_cgDeltaTmpSize)( size )
#else
        0
#endif
    );
    tmp.recreate(tmpsize);
    DEVICE_PTR(TReal) dottmp = tmp.deviceWrite();
    TReal* cputmp = (TReal*)(&(tmp.getCached(0)));
#endif
    float dotresult = 0;
#if defined(MERGE_REDUCTION_KERNELS)
    float dot3result[3] = {0,0,0};
#endif

    int i = 0;

    // Here we assume a initial guess of 0
    // Therefore we can replace "r = b - Aa" in the initial algorithm by "r = b"
    // As a consequence, we also do not copy b to r and d before the loop, but use
    // it directly in the first iteration

    // r = b - Aa;
    // d = r;
    // => d = r = b

    // delta0 = dot(r,r) = dot(b,b);
    DEVICE_METHOD(MechanicalObject3f_vDot)( size, &dotresult, b.deviceRead(), b.deviceRead()
#ifdef PARALLEL_REDUCTION
                                          , dottmp, cputmp
#endif
                  );
    double delta_0 = dotresult;

    if (verbose >= 2) std::cout << "CG Init delta = " << delta_0 << std::endl;
    const double delta_threshold = delta_0 * (tolerance * tolerance);
    double delta_new = delta_0;
    if (delta_new <= delta_threshold) // no iteration, solution is 0
    {
        DEVICE_METHOD(MechanicalObject3f_vClear)( size, a.deviceWrite() );
        simulation_cg_iter = 0;
        return;
    }
    while (i < maxIter && delta_new > delta_threshold)
    {
        // q = Ad;
        mulMatrixVector(params, mesh, matrix, q, ((i==0)?b:d));
        if (verbose >= 2) showDebug(q, "q");
#if defined(MERGE_REDUCTION_KERNELS)
        if (i==0)
            DEVICE_METHOD(MergedKernels3f_cgDot3First)( size, dot3result
                                                        , b.deviceRead(), q.deviceRead()
#ifdef PARALLEL_REDUCTION
                                                        , dottmp, cputmp
#endif
            );
        else
            DEVICE_METHOD(MergedKernels3f_cgDot3)( size, dot3result
                                                   , r.deviceRead(), q.deviceWrite(), d.deviceRead()
#ifdef PARALLEL_REDUCTION
                                                   , dottmp, cputmp
#endif
            );

        double dot_dq = dot3result[0];
        double dot_rq = dot3result[1];
        double dot_qq = dot3result[2];
        double den = dot_dq;
        if (verbose >= 2) std::cout << "CG i="<<i<<" den = " << den << std::endl;
        double alpha = delta_new / den;
        if (verbose >= 2) std::cout << "CG i="<<i<<" alpha = " << alpha << std::endl;
        double delta_old = delta_new;
        // r_new = r - q * alpha
        // delta_new = dot(r_new,r_new) = dot(r - q * alpha,r - q * alpha) = dot(r,r) -2*alpha*dot(r,q) + alpha^2*(dot(q,q)
        delta_new = delta_old - 2*alpha*dot_rq + alpha*alpha*dot_qq;

        if (verbose >= 2) std::cout << "CG i="<<i<<" delta = " << delta_new << std::endl;

        double beta = delta_new / delta_old;
        // a = a + d * alpha;
        // r = r - q * alpha;
        // d = r + d * beta;
        if (i==0)
            DEVICE_METHOD(MergedKernels3f_cgOp3First)( size, alpha, beta
                                                  , r.deviceWrite(), a.deviceWrite(), d.deviceWrite(), q.deviceRead(), b.deviceRead()
            );
        else
            DEVICE_METHOD(MergedKernels3f_cgOp3)( size, alpha, beta
                                                  , r.deviceWrite(), a.deviceWrite(), d.deviceWrite(), q.deviceRead()
            );

        if (verbose >= 2) showDebug(a, "a");
        if (verbose >= 2) showDebug(r, "r");
        if (verbose >= 2) showDebug(d, "d");

#else
        // den = dot(d,q)
        DEVICE_METHOD(MechanicalObject3f_vDot)( size, &dotresult, ((i==0)?b.deviceRead():d.deviceRead()), q.deviceRead()
#ifdef PARALLEL_REDUCTION
                                              , dottmp, cputmp
#endif
                      );
        double den = dotresult;
        if (verbose >= 2) std::cout << "CG i="<<i<<" den = " << den << std::endl;
        double alpha = delta_new / den;
        if (verbose >= 2) std::cout << "CG i="<<i<<" alpha = " << alpha << std::endl;
        double delta_old = delta_new;
        // a = a + d * alpha
        // r = r - q * alpha
        // delta_new = dot(r,r)
#if defined(MERGE_CG_KERNELS)
        DEVICE_METHOD(MergedKernels3f_cgDelta)( (i==0), size, &dotresult, (TReal)alpha
                                              , r.deviceWrite(), a.deviceWrite(), q.deviceRead(), ((i==0)?b.deviceRead():d.deviceRead())
#ifdef PARALLEL_REDUCTION
                                              , dottmp, cputmp
#endif
        );
        delta_new = dotresult;
#else
        if (i==0)
        {
            DEVICE_METHOD(MechanicalObject3f_vEqBF)( size, a.deviceWrite(), b.deviceRead(), alpha );
            DEVICE_METHOD(MechanicalObject3f_vOp)( size, r.deviceWrite(), b.deviceRead(), q.deviceRead(), -alpha );
        }
        else
        {
            DEVICE_METHOD(MechanicalObject3f_vPEqBF)( size, a.deviceWrite(), d.deviceRead(), alpha );
            DEVICE_METHOD(MechanicalObject3f_vPEqBF)( size, r.deviceWrite(), q.deviceRead(), -alpha );
        }
        DEVICE_METHOD(MechanicalObject3f_vDot)( size, &dotresult, r.deviceRead(), r.deviceRead()
#ifdef PARALLEL_REDUCTION
                                              , dottmp, cputmp
#endif
        );
        delta_new = dotresult;
#endif

        if (verbose >= 2) showDebug(a, "a");
        if (verbose >= 2) showDebug(r, "r");

        if (verbose >= 2) std::cout << "CG i="<<i<<" delta = " << delta_new << std::endl;
        double beta = delta_new / delta_old;
        // d = r + d * beta;
        if (i==0)
            DEVICE_METHOD(MechanicalObject3f_vOp)( size, d.deviceWrite(), r.deviceRead(), b.deviceRead(), (TReal)beta );
        else
            DEVICE_METHOD(MechanicalObject3f_vOp)( size, d.deviceWrite(), r.deviceRead(), d.deviceRead(), (TReal)beta );
        if (verbose >= 2) showDebug(d, "d");
#endif
        ++i;
    }
    simulation_cg_iter = i;
    if (verbose >= 1) std::cout << "CG iterations = " << i << " residual error = " << sqrt(delta_new / delta_0) << std::endl;
}

void timeIntegrator_EulerImplicit(const SimulationParameters* params, FEMMesh* mesh)
{
    const double h  = params->timeStep;
    const double rM = params->rayleighMass;
    const double rK = params->rayleighStiffness;

    // Compute right-hand term b
    TVecDeriv& b = mesh->b;
    computeForce(params, mesh, b);
    // no need to apply constraints as it will be done in addKv()
    addKv(params, mesh, h);

    // Compute matrix
    MechanicalMatrix systemMatrix;
    systemMatrix.mFactor = 1 - h*rM;
    systemMatrix.kFactor =   - h*rK - h*h;

    // Solve system for a
    linearSolver_ConjugateGradient(params, mesh, systemMatrix);

    // Apply solution:  v = v + h a        x = x + h v
    TVecCoord& x = mesh->positions;
    TVecDeriv& v = mesh->velocity;
    const TVecDeriv& a = mesh->a;
#ifdef MERGE_CG_KERNELS
    DEVICE_METHOD(MechanicalObject3f_vIntegrate)( x.size(), a.deviceRead(), v.deviceWrite(), x.deviceWrite(), (TReal)h );
#else
    DEVICE_METHOD(MechanicalObject3f_vPEqBF)( x.size(), v.deviceWrite(), a.deviceRead(), (TReal)h );
    DEVICE_METHOD(MechanicalObject3f_vPEqBF)( x.size(), x.deviceWrite(), v.deviceRead(), (TReal)h );
#endif

}

void timeIntegrator_EulerExplicit(const SimulationParameters* params, FEMMesh* mesh)
{
    const double h  = params->timeStep;
    const double rM = params->rayleighMass;
    TVecCoord& x = mesh->positions;
    TVecDeriv& v = mesh->velocity;
    TVecDeriv& f = mesh->f;
    TVecDeriv& a = mesh->a;

    // Compute force
    computeForce(params, mesh, f);

    // Apply constraints
    applyConstraints(params, mesh, f);

    // Compute acceleration
    accFromF(params, mesh, f);

    // Apply damping
    if (rM != 0.0)
        DEVICE_METHOD(MechanicalObject3f_vEqBF)( v.size(), v.deviceWrite(), v.deviceRead(), 1-rM);
    
    // Apply solution:  v = v + h a        x = x + h v
    DEVICE_METHOD(MechanicalObject3f_vPEqBF)( x.size(), v.deviceWrite(), a.deviceRead(), (TReal)h );
    DEVICE_METHOD(MechanicalObject3f_vPEqBF)( x.size(), x.deviceWrite(), v.deviceRead(), (TReal)h );
}

//// MAIN METHOD ////

void simulation_animate()
{
    FEMMesh* mesh = fem_mesh;
    if (!mesh) return;
    switch (simulation_params.odeSolver)
    {
    case ODE_EulerExplicit:
        timeIntegrator_EulerExplicit(&simulation_params, mesh);
        break;
    case ODE_EulerImplicit:
        timeIntegrator_EulerImplicit(&simulation_params, mesh);
        break;
    }

    // non-simulated objects
    sphere_position += sphere_velocity * simulation_params.timeStep;
    mesh->update(&simulation_params);

    simulation_time += simulation_params.timeStep;
    simulation_mapping_needed = true;
}
