#ifndef OPENTISSUE_UTILITY_GK_GL_DRAW_GRID_PLANES_H
#define OPENTISSUE_UTILITY_GK_GL_DRAW_GRID_PLANES_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>

namespace OpenTissue
{

  namespace gl
  {

    template<typename grid_type>
    inline void DrawGridPlanes(grid_type const & grid,unsigned int const & k)
    {
      typedef typename grid_type::math_types    math_types;
      typedef typename math_types::real_type    real_type;

      size_t i,j;

      real_type kmin = grid.min_coord(2);
      //real_type kmax = grid.max_coord(2);
      real_type imin = grid.min_coord(0);
      real_type imax = grid.max_coord(0);
      real_type jmin = grid.min_coord(1);
      real_type jmax = grid.max_coord(1);

      real_type ival,jval,kval;
      
      kval = kmin + k*grid.dz();
      glBegin(GL_LINES);
      jval = grid.min_coord(1);
      for(j=0; j<grid.J(); ++j)
      {
        glVertex3f(imin,jval,kval);
        glVertex3f(imax,jval,kval);
        jval += grid.dy();
      }
      ival = grid.min_coord(0);
      for(i=0; i<grid.I(); ++i)
      {
        glVertex3f(ival,jmin,kval);
        glVertex3f(ival,jmax,kval);
        ival += grid.dx();
      }
      glEnd();
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GK_GL_DRAW_GRID_PLANES_H
#endif
