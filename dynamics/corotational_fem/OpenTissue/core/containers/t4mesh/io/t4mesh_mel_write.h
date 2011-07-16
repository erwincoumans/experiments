#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_MEL_WRITE_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_MEL_WRITE_H
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
#include <sstream>

namespace OpenTissue
{
  namespace t4mesh
  {

    namespace detail
    {

      /**
      * This convenience function createsa polygonal object with the specified
      * index value. The polygonal object corresponds to a uniform scaled version
      * of a tetrahedron.
      *
      * @param idx    The index of the polygonal mesh
      * @param p0     The first coordinate of the tetrahedron.
      * @param p1     The second coordinate of the tetrahedron.
      * @param p2     The third coordinate of the tetrahedron.
      * @param p3     The fourth coordinate of the tetrahedron.
      * @param scale  The scale value to be used.
      *
      */
      template<typename vector3_type,typename real_type>
      inline std::string mel_tetrahedron(  
        size_t const & idx
        , vector3_type const & p0
        , vector3_type const & p1
        , vector3_type const & p2
        , vector3_type const & p3
        , real_type scale
        )
      {
        std::stringstream mel;

        vector3_type c = (p0+p1+p2+p3)/real_type(4.0);
        vector3_type v0 = scale*(p0 - c) + c;
        vector3_type v1 = scale*(p1 - c) + c;
        vector3_type v2 = scale*(p2 - c) + c;
        vector3_type v3 = scale*(p3 - c) + c;


        mel << "polyCreateFacet -ch off"
          << " -p " << v0[0] << " " << v0[1] << " " << v0[2]
        << " -p " << v1[0] << " " << v1[1] << " " << v1[2]
        << " -p " << v2[0] << " " << v2[1] << " " << v2[2]
        << " -n idx" << idx << ";" 
          << std::endl;

        mel << "polyAppendVertex -ch off -v 0 -v 1"
          << " -p " 
          << v3[0] << " " << v3[1] << " " << v3[2]
        << ";"
          << std::endl;

        mel << "polyAppendVertex -ch off -v 1 -v 2 -v 3;" << std::endl;
        mel << "polyAppendVertex -ch off -v 2 -v 0 -v 3;" << std::endl;
        return mel.str();
      }

    } // end of namespace detail

    /**
    * MEL Write Routine.
    * This function is great for exporting a visual representation of a
    * tetrahedra mesh into a MEL (Maya Embedded Language) file. A MEL
    * file is a script file that can be run in Maya. The output file will
    * thus create polygonal meshes corresponding to the individual tetrahedrons
    * in the tetrahedron mesh.
    *
    * @param filename     The filename to write to.
    * @param mesh         The tetrahedra mesh that should be written
    * @param points       The coordinates of the nodes of the tetrahedra mesh.
    * @param scale        The individual scaling to be used upon each tetrahedron.
    *
    * @return             If succesfully written to the specified file then the
    *                     return value is true otherwise it is false.
    */
    template<typename point_container,typename t4mesh_type,typename real_type2>
    inline bool mel_write(std::string const & filename, t4mesh_type const & mesh, point_container & points, real_type2 scale)
    {
      typedef typename t4mesh_type::node_iterator node_iterator;
      typedef typename t4mesh_type::node_iterator const_node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator tetrahedron_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator const_tetrahedron_iterator;

      std::ofstream mel_file(filename.c_str(), std::ios::out);
      if (!mel_file) // TODO: henrikd - gcc says "always true"
      {
        std::cerr << "t4mesh::mel_write(): Error unable to create file: "<< filename << std::endl;
        return false;
      }

      for(const_tetrahedron_iterator tetrahedron=mesh.tetrahedron_begin();tetrahedron!=mesh.tetrahedron_end();++tetrahedron)
      {          
        std::string mel_string = detail::mel_tetrahedron(
            tetrahedron->idx()
          , points[tetrahedron->i()->idx()]
          , points[tetrahedron->j()->idx()]
          , points[tetrahedron->k()->idx()]
          , points[tetrahedron->m()->idx()]
          , scale
            );
        mel_file << mel_string << std::endl;
      }
      mel_file << "select -r \"idx*\" ;" << std::endl;
      mel_file << "group; xform -os -piv 0 0 0;" << std::endl;

      mel_file.flush();
      mel_file.close();

      std::cout << "done writting mel file.." << std::endl;

      return true;
    }

    /**
    * MEL Write Routine.
    * This function is great for exporting a visual representation of a
    * tetrahedra mesh into a MEL (Maya Embedded Language) file. A MEL
    * file is a script file that can be run in Maya. The output file will
    * thus create polygonal meshes corresponding to the individual tetrahedrons
    * in the tetrahedron mesh.
    *
    * @param filename     The filename to write to.
    * @param mesh         The tetrahedra mesh that should be written
    * @param scale        The individual scaling to be used upon each tetrahedron.
    *
    * @return             If succesfully written to the specified file then the
    *                     return value is true otherwise it is false.
    */
    template<typename t4mesh_type, typename real_type2>
    inline bool mel_write(std::string const & filename, t4mesh_type const & mesh, real_type2 scale)
    {
      default_point_container<t4mesh_type> points(const_cast<t4mesh_type*>(&mesh));
      return mel_write(filename,mesh,points,scale);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_MEL_WRITE_H
#endif
