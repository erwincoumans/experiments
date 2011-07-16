#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_READ_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_READ_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_vector3.h>
#include <OpenTissue/core/math/math_is_finite.h>
#include <OpenTissue/core/math/math_is_number.h>

#include <OpenTissue/utility/utility_tag_traits.h>

#include <TinyXML/tinyxml.h>

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

namespace OpenTissue
{
  namespace t4mesh
  {

    /**
     * Read XML Method, with point container for node positions.
     *
     * @param filename      The path and filename of the tetgen file to be
     *                      read (without extensions).
     * @param mesh          The mesh which the file data is read into.
     *
     * @param points        The node positions will be read into this
     *                      container.
     *
     * @return              A boolean indicating success or failure.
     */
    template<typename point_container,typename t4mesh_type>
    bool xml_read(std::string const & filename, t4mesh_type & mesh, point_container & points)
    {
      typedef typename point_container::value_type vector3_type;
      typedef typename t4mesh_type::node_iterator node_iterator;
      typedef typename t4mesh_type::node_iterator const_node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator tetrahedron_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator const_tetrahedron_iterator;

      mesh.clear();

#ifdef TIXML_USE_STL
      TiXmlDocument xml_document(filename);
#else
      TiXmlDocument xml_document(filename.c_str());
#endif

      if(!xml_document.LoadFile())
      {
        std::cerr << "file not found" << std::endl;
        return false;
      }
      TiXmlHandle document_handle( &xml_document );

      TiXmlElement * xml_t4mesh = document_handle.FirstChild( "T4MESH" ).Element();
      assert(xml_t4mesh || !"Oh no, could not find a T4MESH tag?");

      int cnt_nodes = 0;
      if(xml_t4mesh->Attribute("nodes"))
      {
        std::istringstream str_stream(xml_t4mesh->Attribute("nodes"));
        str_stream >> cnt_nodes;
      }
      assert(cnt_nodes>0 || !"Node count was not positive");

      for(int i=0;i< cnt_nodes;++i)
        mesh.insert();

      TiXmlElement * xml_tetrahedron = document_handle.FirstChild( "T4MESH" ).FirstChild( "TETRAHEDRON" ).Element();
      for( ; xml_tetrahedron; xml_tetrahedron=xml_tetrahedron->NextSiblingElement("TETRAHEDRON") )
      {
        int idx0=-1,idx1=-1,idx2=-1,idx3=-1;

        if(!xml_tetrahedron->Attribute("i"))
        {
          std::cerr << "t4mesh::xml_read(): Missing i index on tetrahedron" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_tetrahedron->Attribute("i"));
          str_stream >> idx0;
        }
        if(!xml_tetrahedron->Attribute("j"))
        {
          std::cerr << "t4mesh::xml_read(): Missing j index on tetrahedron" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_tetrahedron->Attribute("j"));
          str_stream >> idx1;
        }
        if(!xml_tetrahedron->Attribute("k"))
        {
          std::cerr << "t4mesh::xml_read(): Missing k index on tetrahedron" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_tetrahedron->Attribute("k"));
          str_stream >> idx2;
        }
        if(!xml_tetrahedron->Attribute("m"))
        {
          std::cerr << "t4mesh::xml_read(): Missing m index on tetrahedron" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_tetrahedron->Attribute("m"));
          str_stream >> idx3;
        }
        if((idx0<0 ||idx1<0 ||idx2<0 ||idx3<0)||(idx0>=cnt_nodes ||idx1>=cnt_nodes ||idx2>=cnt_nodes ||idx3>=cnt_nodes))
        {
          std::cerr << "t4mesh::xml_read(): Illegal node index on tetrahedron" << std::endl;
          return false;
        }
        tetrahedron_iterator T = mesh.insert(idx0,idx1,idx2,idx3);

        // kenny extended tetrahedra to support ``nico tags''
        int tag;
        if(xml_tetrahedron->Attribute("tag"))
        {
            std::istringstream str_stream(xml_tetrahedron->Attribute("tag"));
            str_stream >> tag;
            OpenTissue::utility::set_tag(*T, tag);
        }

      }


      points.resize(cnt_nodes);

      TiXmlElement * xml_point = document_handle.FirstChild( "T4MESH" ).FirstChild( "POINT" ).Element();
      for( ; xml_point; xml_point=xml_point->NextSiblingElement("POINT") )
      {
        int idx;

        if(!xml_point->Attribute("idx"))
        {
          std::cerr << "t4mesh::xml_read(): Missing index on point" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_point->Attribute("idx"));
          str_stream >> idx;
        }
        if(idx<0 || idx>=cnt_nodes)
        {
          std::cerr << "t4mesh::xml_read(): Illegal index on point" << std::endl;
          return false;
        }

        vector3_type value;

        if(!xml_point->Attribute("coord"))
        {
          std::cerr << "t4mesh::xml_read(): Missing coord on point" << std::endl;
          return false;
        }
        else
        {
          std::istringstream str_stream(xml_point->Attribute("coord"));
          str_stream >> value;
        }

        assert(is_number(value(0)) || !"First coordinate was not a number");
        assert(is_finite(value(0)) || !"First coordinate was not finite");
        assert(is_number(value(1)) || !"Second coordinate was not a number");
        assert(is_finite(value(1)) || !"Second coordinate was not finite");
        assert(is_number(value(2)) || !"Third coordinate was not a number");
        assert(is_finite(value(2)) || !"Third coordinate was not finite");

        points[idx](0) = value(0);
        points[idx](1) = value(1);
        points[idx](2) = value(2);

        // nico: tags
        int tag;
        if(xml_point->Attribute("tag"))
        {
            std::istringstream str_stream(xml_point->Attribute("tag"));
            str_stream >> tag;
            OpenTissue::utility::set_tag(*mesh.node(idx), tag);
        }
      }

      xml_document.Clear();
      return true;
    };

    /**
     * Check whether the xml document has tags assigned to the nodes
     *
     * @param filename   The path and filename of the xml file to be
     *                   checked.
     *
     * @return           true if tags present
     */
    inline bool xml_has_tags(std::string const & filename)
    {
#ifdef TIXML_USE_STL
      TiXmlDocument xml_document(filename);
#else
      TiXmlDocument xml_document(filename.c_str());
#endif

      if(!xml_document.LoadFile())
      {
        std::cerr << "file not found" << std::endl;
        return false;
      }
      TiXmlHandle document_handle( &xml_document );

      TiXmlElement * xml_t4mesh = document_handle.FirstChild( "T4MESH" ).Element();
      assert(xml_t4mesh || !"Oh no, could not find a T4MESH tag?");

      bool has_tags = false;

      TiXmlElement * xml_point = document_handle.FirstChild( "T4MESH" ).FirstChild( "POINT" ).Element();
      for( ; xml_point; xml_point=xml_point->NextSiblingElement("POINT") )
      {
        if(xml_point->Attribute("tag"))
        {
          has_tags = true;
          break;
        }
      }

      xml_document.Clear();
      return has_tags;
    }

    /**
     * Read XML Method, convenience call with coordinates read into
     * node->m_coord.
     *
     * @param filename      The path and filename of the xml file to be
     *                      read.
     * @param mesh          The mesh which the file data is read into.
     *
     * @return              A boolean indicating success or failure.
     */
    template<typename t4mesh_type>
    bool xml_read(std::string const & filename,t4mesh_type & mesh)
    {
      default_point_container<t4mesh_type> points(&mesh);
      return xml_read(filename, mesh, points);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_READ_H
#endif
