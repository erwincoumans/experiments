#ifndef WRITE_MESH_OBJ_H
#define WRITE_MESH_OBJ_H

#include <iostream>
#include <fstream>
#include <map>

template<class VecCoord, class VecTri, class VecTexCoord>
bool write_mesh_obj(const char* filename, const char* mtlfile, const VecCoord& points, const VecTri& triangles, const VecTexCoord* texcoords, const std::vector<std::string>* groups = NULL, const std::vector<std::string>* materials = NULL, const std::vector<int>* smooth = NULL)
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Cannot write to file " << filename << std::endl;
        return false;
    }
    std::cout << "Writing file " << filename << std::endl;

    out << "# File exported by gpugems-cuda-fem" << std::endl;
    out << "#Vertex Count " << points.size() << std::endl;
    out << "#Face Count " << triangles.size() << std::endl;
    if (mtlfile && *mtlfile)
        out << "mtllib " << mtlfile << std::endl;

    for (unsigned int i=0; i<points.size(); ++i)
        out << "v " << points[i] << std::endl;

    if (texcoords)
        for (unsigned int i=0; i<texcoords->size(); ++i)
            out << "vt " << (*texcoords)[i] << std::endl;

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        if (groups && i < groups->size())
        {
            if (i==0 || (*groups)[i] != (*groups)[i-1])
                out << "g "<<(*groups)[i] << std::endl;
        }
        if (materials && i < materials->size())
        {
            if (i==0 || (*materials)[i] != (*materials)[i-1])
                out << "usemtl "<<(*materials)[i] << std::endl;
        }
        if (smooth && i < smooth->size())
        {
            if (i==0 || (*smooth)[i] != (*smooth)[i-1])
            {
                int s = (*smooth)[i];
                if (s < 0) out << "s off" << std::endl;
                else       out << "s " << s << std::endl;
            }
        }
        out << "f";
        for (unsigned int j=0;j<triangles[i].size();++j)
        {
            out << ' ' << triangles[i][j]+1;
            if (texcoords && texcoords->size() > (unsigned int) triangles[i][j])
                out << '/' << triangles[i][j]+1;
        }
        out << std::endl;
    }

    out.close();

    return true;
}

#endif
