#ifndef OPENTISSUE_UTILITY_GL_GL_CAMERA_H
#define OPENTISSUE_UTILITY_GL_GL_CAMERA_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_trackball.h>
#include <vector>
#include <cassert>

namespace OpenTissue
{
  namespace gl
  {

    /**
    * Camera Class.
    * The camera class provides functionality for computing the coordinate
    * transformation that brings you from your world space into your local
    * camera coordinate frame.
    *
    * In terms of GLU this similar to what the function gluLookAt() does.
    * 
    * When you want to use a Camera, then in your display handler write
    *
    *  ...
    *  glMatrixMode( GL_MODELVIEW );
    *  LoadCameraMatrix( camera)
    *  ....
    *
    */
    template<typename math_types>
    class Camera : public TrackBall<math_types>
    {
    public:

      typedef TrackBall<math_types>            base_class;
      typedef typename math_types::real_type       real_type;
      typedef typename math_types::vector3_type    vector3_type;
      typedef typename math_types::matrix3x3_type  matrix3x3_type;

    protected:

      vector3_type  m_position;    ///< Viewer Position (Placement of Camera in Model Space.
      vector3_type  m_target;      ///< Viewer Target, the spot in the model space we are looking at.
      vector3_type  m_up;          ///< Viwer Up, the direction in model space that is upwards.
      vector3_type  m_dof;         ///< Direction of Flight, i.e. the direction in model space we are moving forward in....
      vector3_type  m_right;       ///< Direction to the right in model space.
      GLfloat m_modelview[ 16 ];   ///<  Model to Camera Space Transformation Matrix. In column-major order.

    protected:

      bool          m_orbit_mode;      ///< Boolean flag indicating whether camera is orbiting or rotation duing a mouse move.
      bool          m_target_locked;   ///< Boolean flag indicating whether target positions is locked (ie. we allways look at the same spot)
      vector3_type  m_tmp_position;    ///< Temporaries used for handling trackball.
      vector3_type  m_tmp_target;      ///< Temporaries used for handling trackball.
      vector3_type  m_tmp_up;          ///< Temporaries used for handling trackball.

    public:

      vector3_type const & up() const { return m_up; }
      vector3_type const & target() const { return m_target; }
      vector3_type const & position() const { return m_position; }
      vector3_type const & right() const { return m_right; }
      vector3_type const &  dof() const { return m_dof; }
      GLfloat const * get_modelview_matrix()const {return m_modelview; };
      bool & orbit_mode(){return m_orbit_mode;};
      bool const & orbit_mode()const {return m_orbit_mode;};
      bool & target_locked(){return m_target_locked;};
      bool const & target_locked()const {return m_target_locked;};

    public:

      Camera()
        : m_orbit_mode(true)
        , m_target_locked(false)
      {
        init(vector3_type(0,0,40),vector3_type(0,0,0),vector3_type(0,1,0));
      };

      /**
      *
      *
      *
      *
      * @param position_
      * @param target_
      * @param up_
      */
      void init(vector3_type const & position_, vector3_type const & target_, vector3_type const & up_)
      {
        m_position = position_;
        m_target = target_;
        m_up = up_;

        //--- First determine transform from local camera frame to world
        //--- coordinate system, i.e.
        //---
        //---     H = | R T |  =   | X Y Z T |=   | right  up' -dof position |
        //---         | 0 1 |      | 0 0 0 1 |    |   0     0    0    1      |
        //---
        //---
        //---   |q|       |p|
        //---   |1|  =  H |1|    or    q = R*p + T
        //---
        m_dof    = unit(m_target - m_position);
        m_right  = unit(m_dof % m_up);
        m_up     = m_right % m_dof;

        //---
        //--- When rendering we need to transform points from world space into
        //--- camera space, so we need the inverse transform
        //---
        //---   |p|            |1|
        //---   |1|  =  H^{-1} |1|
        //---
        //--- or
        //---
        //---   p =  R^T(  q - T)
        //---
        //--- Thus
        //---               | R^T    -R^T T |
        //---   H^{-1}  =   | 0         1   |
        //---
        m_modelview[ 0 ]  = m_right(0);  m_modelview[ 4 ]  = m_right(1);  m_modelview[ 8 ]  = m_right(2);  m_modelview[ 12 ] = -m_right*m_position;
        m_modelview[ 1 ]  = m_up(0);      m_modelview[ 5 ]  = m_up(1);      m_modelview[ 9 ]  = m_up(2);      m_modelview[ 13 ] = -m_up*m_position;
        m_modelview[ 2 ]  = -m_dof(0);   m_modelview[ 6 ]  = -m_dof(1);   m_modelview[ 10 ]  = -m_dof(2);  m_modelview[ 14 ] = m_dof*m_position;
        m_modelview[ 3 ]  = 0.0f;        m_modelview[ 7 ]  = 0.0f;        m_modelview[ 11 ] = 0.0f;        m_modelview[ 15 ] = 1.0f;
      };

      /**
      * Get Depth.
      *
      * @param r   Some position in world coordinate space
      *
      * @return    The depth, i.e. signed distance along dof wrt. camera position.
      */
      real_type depth(vector3_type const & r)const  {  return m_dof*(r-m_position);  }

    public:

      /**
      * Rotate Camera.
      *
      * @param R
      */
      void rotate(matrix3x3_type const & R)
      {
        matrix3x3_type Rc;
        Rc(0,0) = m_right(0); Rc(0,1) = m_right(1); Rc(0,2) = m_right(2);
        Rc(1,0) = m_up(0); Rc(1,1) = m_up(1); Rc(1,2) = m_up(2);
        Rc(2,0) = -m_dof(0); Rc(2,1) = -m_dof(1); Rc(2,2) = -m_dof(2);
        matrix3x3_type H  =  trans(Rc)*R*Rc;
        if(!m_target_locked)
        {
          m_target = H*(m_target - m_position) + m_position;
        }
        m_up = H*m_up;
        init(m_position,m_target,m_up);
      };

      /**
      * Orbit around target.
      *
      * @param R  The matrix indicating the rotation to orbit with.
      */
      void orbit(matrix3x3_type const & R)
      {
        matrix3x3_type Rc;
        Rc(0,0) = m_right(0); Rc(0,1) = m_right(1); Rc(0,2) = m_right(2);
        Rc(1,0) = m_up(0); Rc(1,1) = m_up(1); Rc(1,2) = m_up(2);
        Rc(2,0) = -m_dof(0); Rc(2,1) = -m_dof(1); Rc(2,2) = -m_dof(2);
        matrix3x3_type H  =  trans(Rc)*R*Rc;

        m_position =  H*(m_position-m_target) + m_target;
        m_up       =  H*m_up;
        init(m_position,m_target,m_up);
      };

      /**
      * Pan Camera.
      *
      * @param x    Horizontal camera displacement.
      * @param y    Vertical   camera displacement.
      */
      void pan(real_type const & x, real_type const & y)
      {
        vector3_type panx = x * m_right;
        vector3_type pany = y * unit(m_right % m_dof);//y*m_up;
        m_position += panx;
        m_position += pany;
        if(!m_target_locked)
        {
          m_target   += panx;
          m_target   += pany;
        }
        init(m_position,m_target,m_up);
      };

      /**
      *
      * @param zoom_factor     Example double size : 2, halfside : 0.5;
      */
      void move(real_type const & distance)
      {
        m_position +=  m_dof*distance;
        if(!m_target_locked)
          m_target +=  m_dof*distance;
        init(m_position,m_target,m_up);
      };

    public:

      /**
      *
      * @param sx
      * @param sy
      */
      void mouse_down(real_type const &  sx,real_type const & sy)
      {
        m_tmp_position = m_position;
        m_tmp_target   = m_target;
        m_tmp_up       = m_up;
        base_class::mouse_down(sx,sy);
      };

      /**
      *
      * @param sx
      * @param sy
      */
      void mouse_up(real_type const &  sx,real_type const & sy)
      {
        base_class::mouse_up(sx,sy);
        init(m_tmp_position, m_tmp_target, m_tmp_up);
        if(m_orbit_mode)
          orbit(  trans( this->get_incremental_rotation() ));
        else
          rotate( this->get_incremental_rotation() );
      };

      /**
      *
      * @param sx
      * @param sy
      */
      void mouse_move(real_type const &  sx,real_type const & sy)
      {
        base_class::mouse_move(sx,sy);
        init(m_tmp_position, m_tmp_target, m_tmp_up);
        if(m_orbit_mode)
          orbit(  trans( this->get_incremental_rotation() ));
        else
          rotate( this->get_incremental_rotation() );
      }

    };

    /**
    *
    * @param camera
    */
    template<typename types>
    inline void MultCameraMatrix(Camera<types> const & camera)
    {
      glMultMatrixf( camera.get_modelview_matrix() );
    }

    /**
    *
    * @param camera
    */
    template<typename types>
    inline void LoadCameraMatrix(Camera<types> const & camera)
    {
      glLoadMatrixf( camera.get_modelview_matrix() );
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_CAMERA_H
#endif
