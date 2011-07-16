#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_WRITE_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_WRITE_H
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

#include <boost/type_traits.hpp>  //--- needed for remove_pointer type_traits

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

namespace OpenTissue
{
  namespace t4mesh
  {

    /**
     * XML Document creation routine. Useful lower­level routine for when
     * you want to keep a reference to the created doc.
     *
     * @param doc          Reference to XML doc to be written.
     * @param mesh         Mesh to be serialized.
     * @param points       Node coordinates will be read from this container.
     *
     * @return             If succesfully written to the specified file then the
     *                     return value is true otherwise it is false.
     *
     * @example
     * @code
     *
     * template<typename skin_type>
     * void skin_xml_embed_extra(skin_type const & skin, TiXmlDocument & doc)
     * {
     *   // iterate over 'POINT' nodes and add normals
     *   TiXmlElement * point = TiXmlHandle(&doc).FirstChild("T4MESH").FirstChild("POINT").Element();
     *   for( int i = 0; point; point=point->NextSiblingElement("POINT"), ++i )
     *   {
     *     skin_type::const_node_iterator n = skin.const_node(i);
     * 
     *     std::stringstream normal;
     *     normal << n->m_original_normal;
     * 
     *     point->SetAttribute("normal", normal.str());
     *   }
     * }
     * 
     * // write the tetrahedral mesh
     * mesh_type skin;
     * TiXmlDocument doc;
     * t4mesh::xml_make_doc(doc, skin, default_point_container<mesh_type>(&skin));
     * skin_xml_embed_extra(skin, doc);
     * doc.SaveFile("skin.xml");
     *
     * @endcode
     */
    template<typename point_container,typename t4mesh_type>
    bool xml_make_doc(TiXmlDocument & doc, t4mesh_type const & mesh, point_container & points)
    {
      typedef typename point_container::value_type vector3_type;
      typedef typename t4mesh_type::node_iterator node_iterator;
      typedef typename t4mesh_type::node_iterator const_node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator tetrahedron_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator const_tetrahedron_iterator;

      assert(points.size()==mesh.size_nodes() || !"node size mismatch between point container and mesh");

      TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
      TiXmlElement * meshelem = new TiXmlElement( "T4MESH" );
      meshelem->SetAttribute( "nodes", mesh.size_nodes() );

      doc.LinkEndChild(decl);
      doc.LinkEndChild(meshelem);

      for(const_tetrahedron_iterator tetrahedron=mesh.tetrahedron_begin();tetrahedron!=mesh.tetrahedron_end();++tetrahedron)
      {
        TiXmlElement * elem = new TiXmlElement( "TETRAHEDRON" );
        elem->SetAttribute( "i", tetrahedron->i()->idx() );
        elem->SetAttribute( "j", tetrahedron->j()->idx() );
        elem->SetAttribute( "k", tetrahedron->k()->idx() );
        elem->SetAttribute( "m", tetrahedron->m()->idx() );

        if (OpenTissue::utility::has_tag(*tetrahedron))
          elem->SetAttribute( "tag", OpenTissue::utility::tag_value(*tetrahedron));

        meshelem->LinkEndChild( elem );
      }

      for(unsigned int i=0;i<mesh.size_nodes();++i)
      {
        vector3_type coord;
        coord(0) = points[i](0);
        coord(1) = points[i](1);
        coord(2) = points[i](2);

        TiXmlElement * elem = new TiXmlElement( "POINT" );

        std::stringstream s;
        s.precision(15);
        s << coord;
        elem->SetAttribute( "idx", i );
        elem->SetAttribute( "coord", s.str() );
        if (OpenTissue::utility::has_tag(*mesh.const_node(i)))
          elem->SetAttribute( "tag", OpenTissue::utility::tag_value(*mesh.const_node(i)));

        meshelem->LinkEndChild( elem );
      }

      return true;
    }

    /**
     * Write t4mesh to XML file. 
     *
     * @param filename     File to write to.
     * @param mesh         Mesh to be serialized.
     * @param points       Node coordinates will be read from this container.
     *
     * @return             If succesfully written to the specified file then the
     *                     return value is true otherwise it is false.
     */
    template<typename point_container,typename t4mesh>
    bool xml_write(std::string const & filename,t4mesh const & mesh,point_container & points)
    {
      assert(points.size()==mesh.size_nodes() || !"node size mismatch between point container and input mesh");

      // build document
      TiXmlDocument doc;
      if (!OpenTissue::t4mesh::xml_make_doc(doc, mesh, points))
        return false;

      // write the document
#ifdef TIXML_USE_STL
      doc.SaveFile(filename);
#else
      doc.SaveFile(filename.c_str());
#endif

      return true;
    }

    /**
     * XML Write Routine, default version where node->m_coord is
     * assumed.
     *
     * @param filename     File to write to.
     * @param mesh         Mesh to be serialized.
     *
     * @return             If successfully written to the specified file then the
     *                     return value is true otherwise it is false.
     */
    template<typename t4mesh_type>
    bool xml_write(std::string const & filename,t4mesh_type const & mesh)
    {
      default_point_container<t4mesh_type> points(const_cast<t4mesh_type*>(&mesh));
      return xml_write(filename,mesh,points);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_IO_T4MESH_XML_WRITE_H
#endif
