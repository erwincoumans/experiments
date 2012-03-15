#ifndef SIMULATION_H
#define SIMULATION_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>

#if defined(SOFA_DEVICE_CPU)
#define SOFA_DEVICE "CPU"
#elif defined(SOFA_DEVICE_CUDA)
#define SOFA_DEVICE "CUDA"
#else
#error Please define SOFA_DEVICE_CPU or SOFA_DEVICE_CUDA
#endif

#if defined(SOFA_DEVICE_CUDA)
#include <cuda/CudaMemoryManager.h>
using namespace sofa::gpu::cuda;
#define MyMemoryManager sofa::gpu::cuda::CudaMemoryManager
#else
enum { BSIZE=1 };
#define MyMemoryManager sofa::helper::CPUMemoryManager
#endif
// we can't use templated typedefs yet...
#define MyVector(T) sofa::helper::vector<T,MyMemoryManager< T > >

// Flag to use 2x3 values instead of 3x3 for co-rotational FEM rotation matrices
#define USE_ROT6

#if defined(SOFA_DEVICE_CUDA)
#define PARALLEL_REDUCTION
#define PARALLEL_GATHER
#define USE_VEC4
#endif

typedef float TReal;
typedef sofa::defaulttype::Vec<3,TReal> TCoord;
typedef sofa::defaulttype::Vec<3,TReal> TDeriv;
typedef sofa::defaulttype::Vec<4,TReal> TCoord4;
typedef MyVector(TReal) TVecReal;
typedef MyVector(TCoord) TVecCoord;
typedef MyVector(TDeriv) TVecDeriv;

typedef sofa::helper::fixed_array<unsigned int, 3> TTriangle;
typedef sofa::helper::fixed_array<unsigned int, 4> TTetra;
typedef MyVector(TTriangle) TVecTriangle;
typedef MyVector(TTetra) TVecTetra;

typedef sofa::defaulttype::Vec<2,float> TTexCoord;
typedef sofa::defaulttype::Vec<4,float> TColor;
typedef MyVector(TTexCoord) TVecTexCoord;

typedef sofa::defaulttype::Vec<3,float> Vec3f;
typedef sofa::defaulttype::Vec<3,double> Vec3d;
typedef sofa::defaulttype::Vec<4,float> Vec4f;
typedef sofa::defaulttype::Vec<4,double> Vec4d;
typedef sofa::defaulttype::Vec<4,int> Vec4i;
typedef sofa::defaulttype::Mat<3,3,float> Mat3x3f;
typedef sofa::defaulttype::Mat<3,3,double> Mat3x3d;

enum TimeIntegration
{
    ODE_EulerExplicit = 0,
    ODE_EulerImplicit,
};

struct SimulationParameters
{
    // Time integration
    double timeStep;
    TimeIntegration odeSolver;
    double rayleighMass;
    double rayleighStiffness;
    // CG Solver
    int maxIter;
    double tolerance;
    // Material properties
    double youngModulusTop,youngModulusBottom;
    double poissonRatio;
    double massDensity;
    // External forces
    TDeriv gravity;
    TDeriv pushForce;
    double planeRepulsion;
    double sphereRepulsion;
    // Constraints
    double fixedHeight;

    SimulationParameters();
};

extern SimulationParameters simulation_params;
extern double simulation_time;
extern TCoord simulation_bbox[2];
extern TCoord simulation_center;
extern double simulation_size;
extern TCoord plane_position;
extern double plane_size;
extern TCoord sphere_position;
extern TDeriv sphere_velocity;
extern double sphere_radius;

template<class real>
struct GPUElement
{
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
};

template<class real>
struct GPUElementRotation
{
#ifdef USE_ROT6
    real rx[3][BSIZE];
    real ry[3][BSIZE];
#else
    real r[9][BSIZE];
#endif
};

template<class real>
struct GPUElementForce
{
    sofa::defaulttype::Vec<4,real> fA,fB,fC,fD;
};


template<class real>
struct GPUPlane
{
    real normal_x, normal_y, normal_z;
    real d;
    real stiffness;
    real damping;
};

template<class real>
struct GPUSphere
{
    real center_x, center_y, center_z;
    real velocity_x, velocity_y, velocity_z;
    real radius;
    real stiffness;
    real damping;
};

struct FEMMesh
{
    std::string filename;
    TVecCoord positions;
    TVecTetra tetrahedra;
    TVecTriangle triangles;
    TCoord bbox[2];
    TVecDeriv velocity;
    TVecCoord positions0; // rest positions

    // Description of external forces
    // In a real application, this could be extended to a different force for each particle
    struct ExternalForce
    {
        int index;
        TDeriv value;
    };
    ExternalForce externalForce;

    // Description of constraints
    int nbFixedParticles;
    MyVector(int) fixedParticles;
    MyVector(unsigned int) fixedMask;
    bool isFixedParticle(int index) const;
    void addFixedParticle(int index);
    void removeFixedParticle(int index);

    // Internal data and methods for simulation
    TVecDeriv f; // force vector when using Euler explicit
    TVecDeriv a,b; // solution and right-hand term when calling CG solver
    TVecDeriv r,d,q; // temporary vectors used by CG solver
#ifdef PARALLEL_REDUCTION
    TVecReal dottmp; // temporary buffer for dot product reductions
#endif

    // PlaneForceField
    GPUPlane<TReal> plane;
    TVecReal planePenetration;

    // SphereForceField
    GPUSphere<TReal> sphere;
    TVecReal spherePenetration;

    // TetrahedronFEMForceField
    MyVector(GPUElement<TReal>) femElem;
    MyVector(GPUElementRotation<TReal>) femElemRotation;
#ifdef PARALLEL_GATHER
    // data for parallel gather operation
    MyVector(GPUElementForce<TReal>) femElemForce;
    int nbElemPerVertex;
    MyVector(int) femVElems;
#endif

#ifdef USE_VEC4
    MyVector(TCoord4) x4,dx4;
#endif

    void reorder();

    void init(SimulationParameters* params);
    void update(SimulationParameters* params);

    void reset();
    void setPushForce(SimulationParameters* params);

    bool save(const std::string& filename);
    bool load(const std::string& filename);

    TReal tetraYoungModulus(int index, SimulationParameters* params)
    {
        const TReal youngModulusTop = (TReal)params->youngModulusTop;
        const TReal youngModulusBottom = (TReal)params->youngModulusBottom;
        TTetra t = tetrahedra[index];
        TReal y = (positions0[t[0]][1]+positions0[t[1]][1]+positions0[t[2]][1]+positions0[t[3]][1])*0.25f;
        y = (y - bbox[0][1])/(bbox[1][1]-bbox[0][1]);
        TReal youngModulus = youngModulusBottom + (youngModulusTop-youngModulusBottom) * y;
        return youngModulus;
    }
    TReal tetraPoissonRatio(int index, SimulationParameters* params)
    {
        return (TReal)params->poissonRatio;
    }

    void saveObj(const std::string& filename, const std::string& mtlfilename);

    FEMMesh()
    {
        nbFixedParticles = 0;
        externalForce.index = -1;
        plane.normal_x = 0;
        plane.normal_y = 1;
        plane.normal_z = 0;
        plane.d = 0;
        plane.stiffness = 0;
        plane.damping = 0;
        sphere.center_x = 0;
        sphere.center_y = 0;
        sphere.center_z = 0;
        sphere.velocity_x = 0;
        sphere.velocity_y = 0;
        sphere.velocity_z = 0;
        sphere.radius = 0;
        sphere.stiffness = 0;
        sphere.damping = 0;
#ifdef PARALLEL_GATHER
        nbElemPerVertex = 0;
#endif
    }
};

extern FEMMesh* fem_mesh;

struct SurfaceMesh
{
    std::string filename;
    TVecCoord positions;
    TVecTriangle triangles;
    TVecCoord normals;
    TVecTexCoord texcoords;
    TVecCoord tangents;

    bool computeTangents;

    std::string textureFilename;
    TColor color;

    // Internal data to map FEM mesh deformation to this surface mesh
    MyVector(TTetra) map_i;
    MyVector(TCoord4) map_f;

#ifdef PARALLEL_GATHER
    // Internal data to compute normals
    TVecCoord fnormals;
    TVecCoord ftangents;
    int nbElemPerVertex;
    MyVector(int) velems;
#endif

    SurfaceMesh()
    : computeTangents(false)
    {
    }

    void init(FEMMesh* inputMesh);
    void updatePositions(FEMMesh* inputMesh);
    void updateNormals();

    void saveObj(const std::string& filename, const std::string& mtlfilename);
};

extern std::vector<SurfaceMesh*> render_meshes;

bool simulation_preload();
bool simulation_load_fem_mesh(const char* filename);
void simulation_reorder_fem_mesh();
bool simulation_load_render_mesh(const char* filename);
bool simulation_init();

void simulation_animate();
void simulation_mapping();
void simulation_reset();
void simulation_save();
void simulation_load();

#endif
