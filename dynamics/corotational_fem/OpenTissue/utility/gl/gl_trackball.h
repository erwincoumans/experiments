#ifndef OPENTISSUE_UTILITY_GL_TRACKBALL_H
#define OPENTISSUE_UTILITY_GL_TRACKBALL_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/utility/trackball/trackball.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Higher-Level OpenGL Trackball Implementation.
    */
    template<typename math_types>
    class TrackBall
    {
    public:

      typedef typename math_types::real_type                real_type;

      typedef          OpenTissue::utility::trackball::Trackball<real_type>  trackball_type;

      typedef typename math_types::vector3_type             vector3_type;
      typedef typename math_types::matrix3x3_type           matrix3x3_type;
      typedef typename math_types::quaternion_type          quaternion_type;
      typedef typename trackball_type::gl_transform_type    gl_transform_type;

    protected:

      real_type                m_radius;
      trackball_type           m_trackball;
      gl_transform_type        m_trackball_matrix;    ///< Trackball rotation, stored in column major form.
    public:

      matrix3x3_type const & get_current_rotation()
      {
        return m_trackball.get_current_rotation();
      }

      gl_transform_type const & get_gl_current_transform()
      {
        return m_trackball.get_gl_current_transform();
      }

      matrix3x3_type const & get_incremental_rotation() const 
      {
        return m_trackball.get_incremental_rotation();
      }

    protected:

      void normalize( real_type const & sx, real_type const & sy, real_type & nx, real_type & ny  )
      {
        GLfloat viewport[4];
        glGetFloatv(GL_VIEWPORT,viewport);
        nx = 2.0 * sx / viewport[2] - 1.0;
        if ( nx < -1.0 )
          nx = -1.0;
        if ( nx > 1.0 )
          nx = 1.0;

        ny = -( 2.0 * sy / viewport[3] - 1.0 );
        if ( ny < -1.0 )
          ny = -1.0;
        if ( ny > 1.0 )
          ny = 1.0;
      };

    public:

      TrackBall()
        : m_radius(1.0)
        , m_trackball(m_radius)
      {
      };

      void mouse_down(real_type const &  sx,real_type const & sy)
      {
        real_type nx,ny;
        normalize(sx,sy,nx,ny);
        m_trackball.begin_drag( nx, ny );
      };

      void mouse_up(real_type const &  sx,real_type const & sy)
      {
        real_type nx,ny;
        normalize(sx,sy,nx,ny);
        m_trackball.end_drag( nx, ny );
      };

      void mouse_move(real_type const &  sx,real_type const & sy)
      {
        real_type nx,ny;
        normalize(sx,sy,nx,ny);
        m_trackball.drag( nx, ny );
      };

    };

    /**
    * Multiplies the top element of the current openGL stack with the trackball transformation matrix.
    */
    template<typename types>
    inline void MultTrackballMatrix( TrackBall<types> const & trackball )
    {
      glMultMatrixd( const_cast<TrackBall<types>*>(&trackball)->get_absolute_rotation() );
    }

    /**
    * Loads the Trackball transformation matrix onto the current opengl matrix stack.
    */
    template<typename types>
    inline void LoadTrackballMatrix( TrackBall<types> const & trackball )
    {
      glLoadMatrixd( const_cast<TrackBall<types>*>(&trackball)->get_absolute_rotation() );
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_TRACKBALL_H
#endif
