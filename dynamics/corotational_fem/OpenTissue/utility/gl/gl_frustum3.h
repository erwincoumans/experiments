#ifndef OPENTISSUE_UTILITY_GL_GL_FRUSTUM_H
#define OPENTISSUE_UTILITY_GL_GL_FRUSTUM_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/geometry/geometry_plane.h>

#include <cassert>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * See http://www.opengl.org/resources/faq/technical/viewcull.c
    */
    template<typename types>
    class Frustum
    {
    public:

      typedef typename types::real_type       real_type;
      typedef typename types::vector3_type    vector3_type;
      typedef          geometry::Plane<types> plane_type;

      const static unsigned int RIGHT  = 0u; ///< The RIGHT side of the frustum
      const static unsigned int LEFT   = 1u; ///< The LEFT	 side of the frustum
      const static unsigned int BOTTOM = 2u; ///< The BOTTOM side of the frustum
      const static unsigned int TOP    = 3u; ///< The TOP side of the frustum
      const static unsigned int BACK   = 4u; ///< The BACK	side of the frustum (near plane)
      const static unsigned int FRONT  = 5u; ///< The FRONT side of the frustum (far plane)

    protected:

      plane_type  m_planes[6];  ///< The planes of the frustum. Normals are pointing inwards!!!

    public:

      plane_type & get_plane(unsigned int const & i) { return m_planes[i]; }
      plane_type const & get_plane(unsigned int const & i) const { return m_planes[i]; }

    protected:

      /**
      * Auxiliary method used to multiply opengl matrices.
      *
      *    C = A*B
      *
      * All matrices are represented in column major form.
      */
      void mul(GLfloat * A,GLfloat * B,GLfloat * C)
      {
        C[ 0] =  A[0]*B[ 0] + A[4]*B[ 1] + A[ 8]*B[ 2] + A[12]*B[ 3];
        C[ 1] =  A[1]*B[ 0] + A[5]*B[ 1] + A[ 9]*B[ 2] + A[13]*B[ 3];
        C[ 2] =  A[2]*B[ 0] + A[6]*B[ 1] + A[10]*B[ 2] + A[14]*B[ 3];
        C[ 3] =  A[3]*B[ 0] + A[7]*B[ 1] + A[11]*B[ 2] + A[15]*B[ 3];
        C[ 4] =  A[0]*B[ 4] + A[4]*B[ 5] + A[ 8]*B[ 6] + A[12]*B[ 7];
        C[ 5] =  A[1]*B[ 4] + A[5]*B[ 5] + A[ 9]*B[ 6] + A[13]*B[ 7];
        C[ 6] =  A[2]*B[ 4] + A[6]*B[ 5] + A[10]*B[ 6] + A[14]*B[ 7];
        C[ 7] =  A[3]*B[ 4] + A[7]*B[ 5] + A[11]*B[ 6] + A[15]*B[ 7];
        C[ 8] =  A[0]*B[ 8] + A[4]*B[ 9] + A[ 8]*B[10] + A[12]*B[11];
        C[ 9] =  A[1]*B[ 8] + A[5]*B[ 9] + A[ 9]*B[10] + A[13]*B[11];
        C[10] =  A[2]*B[ 8] + A[6]*B[ 9] + A[10]*B[10] + A[14]*B[11];
        C[11] =  A[3]*B[ 8] + A[7]*B[ 9] + A[11]*B[10] + A[15]*B[11];
        C[12] =  A[0]*B[12] + A[4]*B[13] + A[ 8]*B[14] + A[12]*B[15];
        C[13] =  A[1]*B[12] + A[5]*B[13] + A[ 9]*B[14] + A[13]*B[15];
        C[14] =  A[2]*B[12] + A[6]*B[13] + A[10]*B[14] + A[14]*B[15];
        C[15] =  A[3]*B[12] + A[7]*B[13] + A[11]*B[14] + A[15]*B[15];
      }

    public:

      /**
      * Computes the current clip planes in model space.
      */
      void  update()
      {
        //--- let x be the homegeneous point in model space and y the correspoding
        //--- canonical view volume point, then we have the 
        //---
        //---            y = P M x
        //---
        //--- where M is the model transform, and P the projection transform. Now
        //--- define the C matrix as:
        //---
        //---     C = P M      
        //---
        //--- Here C is in column major form,
        //---
        //---      | c0 c4 c8   c12 |
        //--- C =  | c1 c5 c9   c13 |  
        //---      | c2 c6 c10  c14 |  
        //---      | c3 c7 c11  c15 |
        //---
        GLfloat P[16];
        GLfloat M[16];
        GLfloat C[16];
        glGetFloatv( GL_PROJECTION_MATRIX, P );
        glGetFloatv( GL_MODELVIEW_MATRIX,  M );
        mul(P,M,C);
        //--- We define the plane equation as
        //---
        //---    n*p - w = 0, 
        //---
        //--- where given a point p0 on the plane we have
        //---
        //---    w = n*p0
        //---
        //--- We can represent such a plane by a 4 dimensional vector, 
        //---
        //---      [ n^T -w ]^T
        //---
        //--- In the canonical view volume we have 6 such vectors
        //---
        //---   right : [-1  0  0  1]
        //---   left  : [ 1  0  0  1]
        //---   bottom: [ 0  1  0  1]
        //---   top   : [ 0 -1  0  1]
        //---   back  : [ 0  0 -1  1]
        //---   front : [ 0  0  1  1]
        //---
        //--- We want to transform these back into model space, such that we can do
        //--- model space clipping.
        //---
        //---
        //---
        //---  Let a plane given by [ n^T -w ]^T be first rotated by R and then translated by T
        //---
        //---     p0' = R p0 + T
        //---      n' = R n
        //---
        //--- So the rotated plane equation is
        //---
        //---    R n *p -  R n *( R p0 + T ) = 0
        //---
        //---    R n *p -  R n * R p0 -  R n * T  = 0
        //---    R n *p -    n * p0   -  n * R^T T  = 0
        //---
        //--- Or using homegeneous coordinates
        //---
        //---               |R       0 |  | n|
        //---   | p^T  1 |  |-R^T T  1 |  |-w|   = 0
        //---
        //---
        //--- But how do we transform a plane with C?....
        //---
        m_planes[RIGHT].n()(0)  =    C[ 3] - C[0];
        m_planes[RIGHT].n()(1)  =    C[ 7] - C[4];
        m_planes[RIGHT].n()(2)  =    C[11] - C[8];
        m_planes[RIGHT].w()     =  -(C[15] - C[12]);

        m_planes[LEFT].n()(0)   =    C[ 3] + C[ 0];
        m_planes[LEFT].n()(1)   =    C[ 7] + C[ 4];
        m_planes[LEFT].n()(2)   =    C[11] + C[ 8];
        m_planes[LEFT].w()      =  -(C[15] + C[12]);

        m_planes[BOTTOM].n()(0) =    C[ 3] + C[ 1];
        m_planes[BOTTOM].n()(1) =    C[ 7] + C[ 5];
        m_planes[BOTTOM].n()(2) =    C[11] + C[ 9];
        m_planes[BOTTOM].w()    =  -(C[15] + C[13]);

        m_planes[TOP].n()(0)    =    C[ 3] - C[ 1];
        m_planes[TOP].n()(1)    =    C[ 7] - C[ 5];
        m_planes[TOP].n()(2)    =    C[11] - C[ 9];
        m_planes[TOP].w()       =  -(C[15] - C[13]);

        m_planes[BACK].n()(0)   =    C[ 3] - C[ 2];
        m_planes[BACK].n()(1)   =    C[ 7] - C[ 6];
        m_planes[BACK].n()(2)   =    C[11] - C[10];
        m_planes[BACK].w()      =  -(C[15] - C[14]);

        m_planes[FRONT].n()(0)  =    C[ 3] + C[ 2];
        m_planes[FRONT].n()(1)  =    C[ 7] + C[ 6];
        m_planes[FRONT].n()(2)  =    C[11] + C[10];
        m_planes[FRONT].w()     =  -(C[15] + C[14]);

        //--- Normalize plane normals
        for(unsigned int i=0;i<6;++i)
        {
          real_type tmp = std::sqrt(m_planes[i].n()*m_planes[i].n());
          if(tmp)
          {
            m_planes[i].n() /= tmp;
            m_planes[i].w() /= tmp;
          }
        }
      };

      /**
      *
      */
      template<typename aabb_type>
      const bool contains(aabb_type const & aabb) const
      {
        vector3_type center   = aabb.center();
        vector3_type halfdiag = aabb.max()-aabb.center();        
        for(int i = 0; i < 6; i++)
        {
          real_type m = m_planes[i].signed_distance(center);
          real_type n = halfdiag(0)*std::fabs(m_planes[i].n()(0)) + halfdiag(1)*std::fabs(m_planes[i].n()(1)) + halfdiag(2)*std::fabs(m_planes[i].n()(2));
          if (m + n < 0)
            return false;          
        }
        return true;
      }    
      /**
      *
      */
      const bool  contains(vector3_type const & p) const
      {
        for(unsigned int i=0;i<6;++i)
          if(m_planes[i].signed_distance(p)<=0)
            return false;
        return true;
      }

      /**
      *
      */
      const bool  contains(vector3_type const & p,real_type const & radius) const
      {
        for(unsigned int i=0;i<6;++i)
          if(m_planes[i].signed_distance(p)<=-radius)
            return false;
        return true;
      }

    };

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_FRUSTUM_H
#endif
