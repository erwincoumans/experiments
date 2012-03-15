#ifndef READ_MESH_NETGEN_H
#define READ_MESH_NETGEN_H
#include <iostream>
#include <fstream>

template<class VecCoord, class VecTetra, class VecTri>
bool read_mesh_netgen(const char* filename, VecCoord& points, VecTetra& tetrahedra, VecTri& triangles)
{
    std::ifstream in(filename);
    if (!in)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }
    std::cout << "Reading file " << filename << std::endl;
    unsigned int nbp = 0;
    in >> nbp;
    points.resize(nbp);
    std::cout << nbp << " points" << std::endl;
    for (unsigned int i=0;i<nbp;++i)
    {
        in >> points[i][0] >> points[i][1] >> points[i][2];
    }

    unsigned int nbe = 0;
    in >> nbe;
    tetrahedra.resize(nbe);
    std::cout << nbe << " tetrahedra" << std::endl;
    for (unsigned int i=0;i<nbe;++i)
    {
        int domain;
        int a,b,c,d;
        in >> domain >> a >> b >> c >> d;
        tetrahedra[i][0] = a-1;
        tetrahedra[i][1] = b-1;
        tetrahedra[i][2] = c-1;
        tetrahedra[i][3] = d-1;
        for (unsigned int j=0;j<4;++j)
            if ((unsigned)tetrahedra[i][j] >= nbp)
            {
                std::cerr << "ERROR: invalid index " << tetrahedra[i][j] << " in tetrahedron " << i << std::endl;
                tetrahedra[i][j] = 0;
            }
    }

    unsigned int nbt = 0;
    in >> nbt;
    triangles.resize(nbt);
    std::cout << nbt << " triangles" << std::endl;
    for (unsigned int i=0;i<nbt;++i)
    {
        int boundary;
        int a,b,c;
        in >> boundary >> a >> b >> c;
        triangles[i][0] = a-1;
        triangles[i][1] = b-1;
        triangles[i][2] = c-1;
        for (unsigned int j=0;j<3;++j)
            if ((unsigned)triangles[i][j] >= nbp)
            {
                std::cerr << "ERROR: invalid index " << triangles[i][j] << " in triangle " << i << std::endl;
                triangles[i][j] = 0;
            }
    }

    in.close();

    return true;
}

#endif
