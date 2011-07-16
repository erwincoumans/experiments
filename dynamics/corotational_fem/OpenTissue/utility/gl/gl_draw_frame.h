#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_FRAME_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_FRAME_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

#include <OpenTissue/utility/gl/gl_color_picker.h>
#include <OpenTissue/utility/gl/gl_draw_point.h>
#include <OpenTissue/utility/gl/gl_draw_vector.h>

#include <OpenTissue/core/math/math_vector3.h>
#include <OpenTissue/core/math/math_matrix3x3.h>
#include <OpenTissue/core/math/math_quaternion.h>
#include <OpenTissue/core/math/math_coordsys.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Frame.
    *
    * @param p    The point where to place origo of the frame
    * @param Q    The orientation of the frame as a quaterion.
    */
    template <typename T>
    inline void DrawFrame( math::Vector3<T> const & p, math::Quaternion<T> const & Q )
    {
      typedef T real_type;
      math::Matrix3x3<real_type> R(Q);
      math::Vector3<real_type> axis1( R( 0, 0 ), R( 1, 0 ), R( 2, 0 ) );
      math::Vector3<real_type> axis2( R( 0, 1 ), R( 1, 1 ), R( 2, 1 ) );
      math::Vector3<real_type> axis3( R( 0, 2 ), R( 1, 2 ), R( 2, 2 ) );
      ColorPicker( 1.0, 0.0, 0.0 );
      DrawVector( p, axis1 );
      ColorPicker ( 0.0, 1.0, 0.0 );
      DrawVector( p, axis2 );
      ColorPicker( 0.0, 0.0, 1.0 );
      DrawVector( p, axis3 );
      GLUquadric * qobj = gluNewQuadric();
      glPushMatrix();
      glTranslatef( p(0), p(1), p(2) );
      GLint slices = 12;
      GLint stacks = 12;
      GLdouble radius = 0.1;
      ColorPicker( .7, .7, .7 );
      gluSphere( qobj, radius, slices, stacks );
      glPopMatrix();
      gluDeleteQuadric( qobj );
    }

    template <typename V>
    inline void DrawFrame( math::CoordSys<V> const & C)
    {
      DrawFrame(C.T(),C.Q());
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_FRAME_H
#endif
