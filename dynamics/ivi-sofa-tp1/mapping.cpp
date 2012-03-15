#include "simulation.h"
#include "kernels.h"
#include "octree.h"

#include <mesh/read_mesh_obj.h>
#include <mesh/write_mesh_obj.h>

#include <iostream>
#include <fstream>

//// DATA ////

extern int verbose;

std::vector<SurfaceMesh*> render_meshes;

bool simulation_mapping_needed = true;

//// METHODS ////

bool simulation_load_render_mesh(const char* filename)
{
    SurfaceMesh* mesh = new SurfaceMesh;
    if (!read_mesh_obj(filename, mesh->positions, mesh->triangles, (render_meshes.empty() ? &mesh->texcoords : NULL))) // we use the texture only for the first OBJ
    {
        delete mesh;
        return false;
    }
    mesh->filename = filename;
    mesh->updateNormals();
    if (render_meshes.size()&1)
        mesh->color = TColor(0.15f, 0.15f, 0.15f, 1.0f);
    else
        mesh->color = TColor(0.8f, 0.8f, 0.8f, 1.0f);
    render_meshes.push_back(mesh);
    return true;
}

void SurfaceMesh::saveObj(const std::string& filename, const std::string& mtlfilename)
{
    std::vector<int> smooth;
    smooth.push_back(1);
    std::vector<std::string> groups;
    groups.push_back("Group1");
    std::vector<std::string> mats;
    std::string mtlfile;
    if (!mtlfilename.empty())
    {
        std::string matname;
        if (!textureFilename.empty()) matname = "texturedMat";
        else if (color[0] < 0.5) matname = "darkMat";
        else matname = "defaultMat";
        mats.push_back(matname);
        std::size_t p = mtlfilename.rfind('/');
        mtlfile = std::string(mtlfilename, p == std::string::npos ? 0 : p+1);
        std::ofstream out(mtlfilename.c_str());
        if (out)
        {
            out << "newmtl " << matname << std::endl;
            out << "illum 2" << std::endl;
            out << "Ka " << color[0] << " " << color[1] << " " << color[2] << std::endl;
            out << "Kd " << color[0] << " " << color[1] << " " << color[2] << std::endl;
            out << "Ks " << 1 << " " << 1 << " " << 1 << std::endl;
            out << "Ns " << 20 << std::endl;
            if (!textureFilename.empty())
            {
                std::size_t p = textureFilename.rfind('/');
                out << "map_Kd " << textureFilename.substr(p == std::string::npos ? 0 : p+1) << std::endl;
            }
            out.close();
        }
    }
    write_mesh_obj(filename.c_str(), mtlfile.c_str(), positions, triangles, !texcoords.empty() ? &texcoords : NULL, &groups, &mats, &smooth);
}

void FEMMesh::saveObj(const std::string& filename, const std::string& mtlfilename)
{
    std::vector<int> smooth;
    smooth.push_back(-1);
    std::vector<std::string> groups;
    groups.push_back("Tetras");
    std::vector<std::string> mats;
    std::string mtlfile;
    std::string matname[4];
    if (!mtlfilename.empty())
    {
        matname[0] = "fD";
        matname[1] = "fA";
        matname[2] = "fB";
        matname[3] = "fC";
        TColor colors[4];
        colors[0] = TColor(0,0,1,1);
        colors[1] = TColor(0,0.5f,1,1);
        colors[2] = TColor(0,1,1,1);
        colors[3] = TColor(0.5f,1,1,1);
        std::size_t p = mtlfilename.rfind('/');
        mtlfile = std::string(mtlfilename, p == std::string::npos ? 0 : p+1);
        std::ofstream out(mtlfilename.c_str());
        if (out)
        {
            for (int m=0;m<4;++m)
            {
                out << "newmtl " << matname[m] << std::endl;
                TColor color = colors[m];
                out << "illum 2" << std::endl;
                out << "Ka " << color[0] << " " << color[1] << " " << color[2] << std::endl;
                out << "Kd " << color[0] << " " << color[1] << " " << color[2] << std::endl;
                out << "Ks " << 1 << " " << 1 << " " << 1 << std::endl;
                out << "Ns " << 20 << std::endl;
            }
            out.close();
        }
    }
    TVecCoord points; points.reserve(tetrahedra.size()*4);
    TVecTriangle triangles; triangles.reserve(tetrahedra.size()*4);
    mats.reserve(tetrahedra.size()*4);
    for (unsigned int i=0;i<tetrahedra.size();++i)
    {
        TTetra t = tetrahedra[i];

        TCoord a = positions[t[0]];
        TCoord b = positions[t[1]];
        TCoord c = positions[t[2]];
        TCoord d = positions[t[3]];
        TCoord center = (a+b+c+d)*(TReal)0.125;
        a = (a+center)*(TReal)0.666667;
        b = (b+center)*(TReal)0.666667;
        c = (c+center)*(TReal)0.666667;
        d = (d+center)*(TReal)0.666667;
        unsigned int pi = points.size();
        points.push_back(a);
        points.push_back(b);
        points.push_back(c);
        points.push_back(d);
        
        //glColor4f(0,0,1,1);
        //glVertex3fv(a.ptr()); glVertex3fv(b.ptr()); glVertex3fv(c.ptr());
        mats.push_back(matname[0]);
        triangles.push_back(TTriangle(pi+0, pi+1, pi+2));
        
        //glColor4f(0,0.5f,1,1);
        //glVertex3fv(b.ptr()); glVertex3fv(c.ptr()); glVertex3fv(d.ptr());
        mats.push_back(matname[1]);
        triangles.push_back(TTriangle(pi+1, pi+2, pi+3));
        
        //glColor4f(0,1,1,1);
        //glVertex3fv(c.ptr()); glVertex3fv(d.ptr()); glVertex3fv(a.ptr());
        mats.push_back(matname[2]);
        triangles.push_back(TTriangle(pi+2, pi+3, pi+0));
        
        //glColor4f(0.5f,1,1,1);
        //glVertex3fv(d.ptr()); glVertex3fv(a.ptr()); glVertex3fv(b.ptr());
        mats.push_back(matname[3]);
        triangles.push_back(TTriangle(pi+3, pi+0, pi+1));
    }
    write_mesh_obj(filename.c_str(), mtlfile.c_str(), points, triangles, (const TVecTexCoord*)NULL, &groups, &mats, &smooth);
}

void SurfaceMesh::init(FEMMesh* inputMesh)
{
    // elements <-> particles table
#ifdef PARALLEL_GATHER
    {
        const int nbp = positions.size();
        const int nbe = triangles.size();
        const int nbBp = (nbp + BSIZE-1)/BSIZE;
        const int nbBe = (nbe + BSIZE-1)/BSIZE;
        // first find number of elements per particle
        std::vector<int> p_nbe;
        p_nbe.resize(nbp);
        for (int eindex = 0; eindex < nbe; ++eindex)
            for (int j = 0; j < 3; ++j)
                ++p_nbe[triangles[eindex][j]];
        // then compute max value
        nbElemPerVertex = 0;
        for (int i=0;i<nbp;++i)
            if (p_nbe[i] > nbElemPerVertex) nbElemPerVertex = p_nbe[i];
        // finally fill velems array
        velems.resize(nbBp*nbElemPerVertex*BSIZE);
        p_nbe.clear();
        p_nbe.resize(nbp);
        for (int eindex = 0; eindex < nbe; ++eindex)
            for (int j = 0; j < 3; ++j)
            {
                int p = triangles[eindex][j];
                int num = p_nbe[p]++;
                const int block  = p / BSIZE;
                const int thread = p % BSIZE;
                velems[ block * (nbElemPerVertex * BSIZE) +
                           num * BSIZE + thread ] = 1 + eindex;
            }
    }
#endif
    // FEMMesh -> SurfaceMesh mapping
    if (inputMesh)
    {
        std::cout << "Creating mapping between simulation mesh \"" << inputMesh->filename << "\" and surface mesh \"" << filename << "\"..." << std::endl;
        static std::string input_filename;
        static sofa::helper::vector<Mat3x3d> bases;
        static sofa::helper::vector<Vec3d> centers;
        static Octree<Vec3d> octree;
        const TVecTetra& tetras = inputMesh->tetrahedra;
        const TVecCoord& in = inputMesh->positions;
        const TVecCoord& out = positions;
        map_i.resize(out.size());
        map_f.resize(out.size());
        if (input_filename != inputMesh->filename || bases.size() != tetras.size()) // we have to recompute the octree and bases
        {
            input_filename = inputMesh->filename;
            sofa::helper::vector< BBox<Vec3d> > bbox;
            bases.resize(tetras.size());
            centers.resize(tetras.size());
            bbox.resize(tetras.size());
            std::cout << "  Preparing tetrahedra" << std::endl;
            for (unsigned int t=0; t<tetras.size(); ++t)
            {
                Mat3x3d m, mt;
                m[0] = in[tetras[t][1]]-in[tetras[t][0]];
                m[1] = in[tetras[t][2]]-in[tetras[t][0]];
                m[2] = in[tetras[t][3]]-in[tetras[t][0]];
                mt.transpose(m);
                bases[t].invert(mt);
                centers[t] = (in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]])*0.25;
                bbox[t].add(tetras[t].begin(), tetras[t].end(), in);
            }
            std::cout << "  Building octree" << std::endl;
            octree.init(bbox,8,8);
        }
        std::cout << "  Processing vertices" << std::endl;
        int outside = 0;
        sofa::helper::vector<Octree<Vec3d>*> cells;
        for (unsigned int i=0;i<out.size();i++)
        {
            Vec3d pos = out[i];
            Vec3d coefs;
            int index = -1;
            double distance = 1e10;
            Octree<Vec3d>* cell = octree.findNear(pos);
            if (cell)
            {
                const sofa::helper::vector<int>& elems = cell->elems();
                for (unsigned int e = 0; e < elems.size(); e++)
                {
                    unsigned int t = elems[e];
                    Vec3d v = bases[t] * (pos - in[tetras[t][0]]);
                    double d = std::max(std::max(-v[0],-v[1]),std::max(-v[2],v[0]+v[1]+v[2]-1));
                    if (d>0) d = (pos-centers[t]).norm2();
                    if (d<distance) { coefs = v; distance = d; index = t; }
                }
            }
            if (distance > 0)
            { // pos is outside of the fem mesh, find the nearest tetra
                
                // first let's find at least one tetra that is close, if not already found
                if (index >= 0) // we already have a close tetra, we need to look only for closer ones
                {
                    cells.clear();
                    octree.findAllAround(cells, pos, sqrt(distance)*1.5);
                    for (unsigned int ci = 0; ci < cells.size(); ++ci)
                    {
                        if (cells[ci] == cell) continue; // already processed this cell
                        const sofa::helper::vector<int>& elems = cells[ci]->elems();
                        for (unsigned int e = 0; e < elems.size(); e++)
                        {
                            unsigned int t = elems[e];
                            double d = (pos-centers[t]).norm2();
                            if (d<distance)
                            {
                                coefs = bases[t] * (pos - in[tetras[t][0]]);
                                distance = d; index = t;
                            }
                        }
                    }
                }
                else
                {
                    // failsafe case (should not happen...), to be sure we do a brute-force search
                    for (unsigned int t = 0; t < tetras.size(); t++)
                    {
                        double d = (pos-centers[t]).norm2();
                        if (d<distance)
                        {
                            coefs = bases[t] * (pos - in[tetras[t][0]]);
                            distance = d; index = t;
                        }
                    }
                }
                if (index >= 0)
                {
                    if (verbose >= 1) std::cout << "Surface vertex " << i << " mapped outside of tetra " << index << " with coefs " << coefs << std::endl;
                    ++outside;
                }
            }
            if (index >= 0)
            {
                //std::cout << "Surface vertex " << i << " mapped from tetra " << index << " with coefs " << coefs << std::endl;
                map_i[i][0] = tetras[index][0];  map_f[i][0] = (float)(1-coefs[0]-coefs[1]-coefs[2]);
                map_i[i][1] = tetras[index][1];  map_f[i][1] = (float)(coefs[0]);
                map_i[i][2] = tetras[index][2];  map_f[i][2] = (float)(coefs[1]);
                map_i[i][3] = tetras[index][3];  map_f[i][3] = (float)(coefs[2]);
            }
        }
        std::cout << "Mapping done: " << outside << " / " << out.size() << " vertices outside of simulation mesh" << std::endl;
    }
}

//// MAPPING METHODS ////

void SurfaceMesh::updatePositions(FEMMesh* inputMesh)
{
    const TVecCoord& in = inputMesh->positions;
    TVecCoord& out = positions;
    if (map_f.size() != out.size() || map_i.size() != out.size()) return;

    DEVICE_METHOD(TetraMapper3f_apply)( out.size(), map_i.deviceRead(), map_f.deviceRead(), out.deviceWrite(), in.deviceRead() );

    /*for (unsigned int i=0;i<out.size();++i)
    {
        out[i] = 
            in[map_i[i][0]] * map_f[i][0] +
            in[map_i[i][1]] * map_f[i][1] +
            in[map_i[i][2]] * map_f[i][2] +
            in[map_i[i][3]] * map_f[i][3];
    }*/
}

void SurfaceMesh::updateNormals()
{
    normals.recreate(positions.size());
    if (computeTangents)
        tangents.recreate(positions.size());
#ifdef SOFA_DEVICE_CUDA
    if (!velems.empty())
    { // use GPU
        if (!computeTangents)
        {
            fnormals.recreate(triangles.size());
            CudaVisualModel3f_calcTNormals(triangles.size(), positions.size(), triangles.deviceRead(), fnormals.deviceWrite(), positions.deviceRead());
            CudaVisualModel3f_calcVNormals(triangles.size(), positions.size(), nbElemPerVertex, velems.deviceRead(), normals.deviceWrite(), fnormals.deviceRead(), positions.deviceRead());
        }
        else
        {
            fnormals.recreate(triangles.size());
            ftangents.recreate(triangles.size());
            CudaVisualModel3f_calcTNormalsAndTangents(triangles.size(), positions.size(), triangles.deviceRead(), fnormals.deviceWrite(), ftangents.deviceWrite(), positions.deviceRead(), texcoords.deviceRead());
            CudaVisualModel3f_calcVNormalsAndTangents(triangles.size(), positions.size(), nbElemPerVertex, velems.deviceRead(), normals.deviceWrite(), tangents.deviceWrite(), fnormals.deviceRead(), ftangents.deviceRead(), positions.deviceRead(), texcoords.deviceRead());
        }
    }
    else
#endif
    { // use CPU
        if (!computeTangents)
        {
            for (unsigned int i=0;i<normals.size();++i)
                normals[i].clear();
            for (unsigned int i=0;i<triangles.size();++i)
            {
                TCoord n = cross(positions[triangles[i][1]]-positions[triangles[i][0]], 
                                 positions[triangles[i][2]]-positions[triangles[i][0]]);
                n.normalize();
                for (unsigned int j=0;j<3;++j)
                    normals[triangles[i][j]] += n;
            }
            for (unsigned int i=0;i<normals.size();++i)
                normals[i].normalize();
        }
        else
        {
            for (unsigned int i=0;i<normals.size();++i)
            {
                normals[i].clear();
                tangents[i].clear();
            }
            for (unsigned int i=0;i<triangles.size();++i)
            {
                TCoord A = positions[triangles[i][0]];
                TCoord B = positions[triangles[i][1]];
                TCoord C = positions[triangles[i][2]];
                B -= A;
                C -= A;
                TCoord n = cross(B,C);
                n.normalize();
                TReal Au = texcoords[triangles[i][0]][0];
                TReal Bu = texcoords[triangles[i][1]][0];
                TReal Cu = texcoords[triangles[i][2]][0];
                Bu -= Au;
                Cu -= Au;
                TCoord t = B * Cu - C * Bu;
                t.normalize();
                for (unsigned int j=0;j<3;++j)
                {
                    normals[triangles[i][j]] += n;
                    tangents[triangles[i][j]] += t;
                }
            }
            for (unsigned int i=0;i<normals.size();++i)
            {
                tangents[i] = cross(normals[i],tangents[i]);
                normals[i].normalize();
                tangents[i].normalize();
            }
        }
    }
}

//// MAIN METHOD ////

void simulation_mapping()
{
    if (!simulation_mapping_needed) return;

    simulation_mapping_needed = false;
    FEMMesh* mesh = fem_mesh;
    if (!mesh) return;
    for (unsigned int i = 0; i < render_meshes.size(); ++i)
    {
        render_meshes[i]->updatePositions(mesh);
        render_meshes[i]->updateNormals();
    }
}
