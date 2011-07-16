#ifndef OPENTISSUE_DYNAMICS_FEM_FEM_UPDATE_ORIGINAL_COORD_H
#define OPENTISSUE_DYNAMICS_FEM_FEM_UPDATE_ORIGINAL_COORD_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

namespace OpenTissue
{
  namespace fem
  {

    /**
    * Update original coords with world coords.
    *
    * Note: This completely destroys any prior initialization
    * of force offset (f0 in stiffness elements initialization).
    * Therefore these must be re-initialized after invocation
    * of this function.
    *
    * @param begin
    * @param end
    */
    template < typename node_iterator >
    inline void update_original_coord(node_iterator const & begin,node_iterator const & end)
    {
      for (node_iterator N = begin;N!=end;++N)
        N->m_model_coord = N->m_coord;
    }

  } // namespace fem
} // namespace OpenTissue

//OPENTISSUE_DYNAMICS_FEM_FEM_UPDATE_ORIGINAL_COORD_H
#endif
