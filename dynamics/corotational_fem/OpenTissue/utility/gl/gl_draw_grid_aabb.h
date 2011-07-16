#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_GRID_AABB_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_GRID_AABB_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/core/geometry/geometry_aabb.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw AABB box around the Grid.
    *
    * @param grid   The grid to draw.
    */
    template<typename grid_type>
    inline void DrawGridAABB(grid_type const & grid)
    {
      typedef typename grid_type::math_types       math_types;

      typedef geometry::AABB<math_types>          aabb_type;

      aabb_type aabb( grid.min_coord(), grid.max_coord() );

      DrawAABB( aabb, true );
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_GRID_AABB_H
#endif
