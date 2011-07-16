#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_TETGEN_READ_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_TETGEN_READ_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <iostream>
#include <fstream>
#include <map>

namespace OpenTissue
{
  namespace t4mesh
  {
    /**
    * Read TetGen Method.
    *
    * @param filename      The path and filename of the tetgen file to be
    *                      read (without extensions).
    * @param mesh          The mesh which the file data is read into.
    *
    * @return              A boolean indicating success or failure.
    */
    template<typename point_container,typename t4mesh_type>
    bool tetgen_read(const std::string & filename,t4mesh_type & mesh,point_container & points)
    {
      typedef typename t4mesh_type::node_type              node_type;
      typedef typename node_type::real_type                real_type;
      typedef typename t4mesh_type::node_iterator          node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator   tetrahedron_iterator;

      mesh.clear();

      std::string node_filename = filename + ".node";
      std::string element_filename = filename + ".ele";

      std::ifstream file(node_filename.c_str());

      if(!file.is_open())
      {
        std::cerr << "Error unable to open file '" << node_filename << "'" << std::endl;
        return false;
      }

      // .node file
      //
      // First line: <# of points> <dimension (must be 3)> <# of attributes> <boundary markers (0 or 1)>
      // Remaining line: <point #> <x> <y> <z> [attributes] [boundary marker]
      int cnt_nodes, dimension, cnt_attributes;
      bool has_boundary;

      file >> cnt_nodes >> dimension >> cnt_attributes >> has_boundary;

      points.resize(cnt_nodes);

      std::map<int,node_iterator> internal_node_iterator;  //--- Auxiliary lookup data structure

      int external_index;
      real_type x, y, z;
      real_type attribute;

      int boundary_marker;
      for(int i=0; i<cnt_nodes; ++i)
      {
        file >> external_index >> x >> y >> z;

        for (int a = 0; a < cnt_attributes; ++a)
          file >> attribute; // Just skip them
        if (has_boundary)
          file >> boundary_marker;

        node_iterator node = mesh.insert();
        points[node->idx()](0) = x;
        points[node->idx()](1) = y;
        points[node->idx()](2) = z;

        internal_node_iterator[external_index] = node;
      }
      file.close();
      // .ele file
      //
      // First line: <# of tetrahedra> <points per tetrahedron> <# of attributes>
      // Remaining line: <tetrahedron #> <point> <point> <point> <point> [attributes]
      file.open(element_filename.c_str());
      if(!file.is_open())
      {
        std::cerr << "Error unable to open file '" << element_filename << "'" << std::endl;
        return false;
      }
      int cnt_tetrahedra, cnt_points;
      file >> cnt_tetrahedra >> cnt_points >> cnt_attributes;
      if (cnt_points != 4)
      {
        std::cerr << "We only support tetrahedra with 4 points!" << std::endl;
        file.close();
        return false;
      }
      //---
      //--- By convention the first attribute contains the region
      //--- number that a tetrahedra belongs to
      //---
      if(cnt_attributes>1)
      {
        std::cerr << "We only support tetrahedra with 1 region!" << std::endl;
        file.close();
        return false;
      }
      int external_node_idx0;
      int external_node_idx1;
      int external_node_idx2;
      int external_node_idx3;
      for(int i=0;i<cnt_tetrahedra;++i)
      {
        file >> external_index >> external_node_idx0 >> external_node_idx1 >> external_node_idx2 >> external_node_idx3;

        for (int a = 0; a < cnt_attributes; ++a)
          file >> attribute; // Just skip any attributes

        node_iterator n0 = internal_node_iterator[external_node_idx0];
        node_iterator n1 = internal_node_iterator[external_node_idx1];
        node_iterator n2 = internal_node_iterator[external_node_idx2];
        node_iterator n3 = internal_node_iterator[external_node_idx3];
        mesh.insert(n0,n1,n2,n3);
      }
      file.close();
      std::cout << "Read "
        << cnt_nodes
        << " nodes and "
        << cnt_tetrahedra
        << " tetrahedra"
        << std::endl;
      return true;
    }


    /**
    * Read TetGen Method.
    *
    * @param filename      The path and filename of the tetgen file to be
    *                      read (without extensions).
    * @param mesh          The mesh which the file data is read into.
    *
    * @return              A boolean indicating success or failure.
    */
    template<typename t4mesh_type>
    bool tetgen_read(const std::string & filename,t4mesh_type & mesh)
    {
      default_point_container<t4mesh_type> points(&mesh);
      return tetgen_read(filename,mesh,points);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_TETGEN_READ_H
#endif
