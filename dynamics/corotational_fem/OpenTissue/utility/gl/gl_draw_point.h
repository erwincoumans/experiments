#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_POINT_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_POINT_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/math/math_vector3.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Point.
    *
    */
    template <typename vector3_type>
    inline void DrawPoint( vector3_type const & p , double const radius = 0.1)
    {
      typedef typename vector3_type::value_type value_type;
      GLUquadric * qobj = gluNewQuadric();
      glPushMatrix();
      glTranslatef( p(0), p(1), p(2) );
      GLint slices = 8;
      GLint stacks = 8;
      gluSphere( qobj, radius, slices, stacks );
      glPopMatrix();
      gluDeleteQuadric( qobj );
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_POINT_H
#endif
