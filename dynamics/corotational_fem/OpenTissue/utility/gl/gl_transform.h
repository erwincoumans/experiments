#ifndef OPENTISSUE_UTILITY_GL_GL_TRANSFORM_H
#define OPENTISSUE_UTILITY_GL_GL_TRANSFORM_H
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
#include <OpenTissue/core/math/math_quaternion.h>
#include <OpenTissue/core/math/math_coordsys.h>
#include <OpenTissue/core/math/math_constants.h>

#include <cmath>

namespace OpenTissue
{

  namespace gl
  {

    template<typename value_type>
    inline void  Transform( math::Vector3<value_type> const & r)
    {
      glTranslatef( ( GLfloat ) r( 0 ), ( GLfloat ) r( 1 ), ( GLfloat ) r( 2 ) );
    }

    /**
    * Apply Rigid Body Transformation to current Matrix Track.
    *
    * @param Q    Rotation.
    */
    template <typename value_type>
    inline void  Transform ( math::Quaternion<value_type> const & Q )
    {
      using std::acos;
      using std::sin;

      value_type angle = acos( Q.s() ) * 2.0;
      if ( angle )
      {
        value_type factor = sin( angle / 2.0 );
        value_type rx = Q.v()( 0 ) / factor;
        value_type ry = Q.v()( 1 ) / factor;
        value_type rz = Q.v()( 2 ) / factor;
        angle = ( angle / math::detail::pi<value_type>() ) * 180;  //--- Convert to degrees
        glRotatef( ( GLfloat ) angle, ( GLfloat ) rx, ( GLfloat ) ry, ( GLfloat ) rz );
      }
    }

    /**
    * Apply Rigid Body Transformation to current Matrix Track.
    *
    * That is rotate before translate!
    *
    * @param r    Translation.
    * @param Q    Rotation.
    */
    template <typename vector3_type, typename quaternion_type>
    inline void Transform ( vector3_type const & r, quaternion_type const & Q )
    {
      Transform(r);
      Transform(Q);
    }

    /**
    * Apply Coordsys Transformation to current Matrix Track.
    *
    * That is rotate before translate!
    *
    * @param C    Coordinate System Transform.
    */
    template <typename V>
    inline void  Transform ( math::CoordSys<V> const & C )
    {
      Transform(C.T());
      Transform(C.Q());
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_TRANSFORM_H
#endif
