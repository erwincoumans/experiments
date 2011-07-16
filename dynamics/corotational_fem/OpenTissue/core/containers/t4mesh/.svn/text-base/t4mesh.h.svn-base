#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/t4mesh/t4mesh_default_traits.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_default_point_container.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4node.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4tetrahedron.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4mesh.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4edges.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4boundary_faces.h>

#include <OpenTissue/core/math/math_basic_types.h>

namespace OpenTissue
{
  namespace t4mesh
  {
    /**
    * Basic (Simple) Tetrahedra Mesh.
    *
    * This tetrahedra mesh data structure is designed specially for
    * two purposes: It should maintain a valid topology of the mesh
    * at all times, that is the connectivity of nodes and tetrahedra
    * are always valid.
    *
    * The other purpose is to make sure that the global indexing of
    * nodes (0..N-1) and tetrahedra (0..T-1) always is a compact range
    * starting from zero to the maximum number minus one.
    *
    * Obviously removing entities (nodes or tetrahedra) alters the global
    * index ranges, thus end users can not trust previously stored indices
    * of entities in their own apps.
    *
    * The mesh takes three template arguments. The first specifies the
    * math_types used in the mesh. The following two arguments are node
    * traits and tetrahedron traits respectively. Currently these template
    * arguments are set to meaningfull defaults.
    *
    */
    template<
        typename M = OpenTissue::math::BasicMathTypes<double, size_t>
      , typename N = t4mesh::DefaultNodeTraits< M >
      , typename T = t4mesh::DefaultTetrahedronTraits
    >
    class T4Mesh 
      : public OpenTissue::t4mesh::detail::T4Mesh<M,N,T>
    {};

  } // end of namespace t4mesh
} // end of namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_T4MESH_H
#endif
