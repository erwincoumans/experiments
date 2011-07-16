#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_JOINT_LIMITS_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_JOINT_LIMITS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/kinematics/skeleton/skeleton_bone_access.h>

namespace OpenTissue
{
  namespace gl
  {

    /**
     * Draw Inverse Kinematics Joint Limits
     *
     * @param bone     A skeleton bone holding information about the joint limits that should be visualized.
     */
    template<typename bone_type>
    inline void DrawJointLimits(bone_type const & bone)
    {
      using std::cos;
      using std::sin;

      typedef typename bone_type::bone_traits      bone_traits;
      typedef typename bone_type::math_types       math_types;
      typedef typename math_types::value_traits    value_traits;
      typedef typename math_types::real_type       real_type;
      typedef typename math_types::vector3_type    vector3_type;
      typedef typename math_types::quaternion_type quaternion_type;

      float const r = length( bone.relative().T() )* 0.4f ;

      glPushMatrix();

      // Joint limits are specified (i.e. fixed constants) in the parent frame
      if(!bone.is_root())
      {
        Transform( bone.parent()->absolute() );
      }

      if(bone.type() == bone_traits::ball_type)
      {

        size_t const N = 10;

        std::vector<real_type> x;
        std::vector<real_type> y;
        std::vector<real_type> z;
        std::vector<real_type> nx;
        std::vector<real_type> ny;
        std::vector<real_type> nz;
        x.resize(N*N);
        y.resize(N*N);
        z.resize(N*N);
        nx.resize(N*N);
        ny.resize(N*N);
        nz.resize(N*N);
        bone_type const * child = OpenTissue::skeleton::BoneAccess::get_first_child_ptr( &bone );
        vector3_type const T_child = child ?  child->bind_pose().T() :  vector3_type(value_traits::zero(), value_traits::zero(), value_traits::zero() );        
        
        // R =  Rz( 0 = phi ) Ry( 1 = psi ) Rz(  2 = theta )
        real_type const & phi_min    = bone.box_limits().min_limit(0);
        real_type const & psi_min    = bone.box_limits().min_limit(1);
        real_type const & theta_min  = bone.box_limits().min_limit(2);

        real_type const & phi_max    = bone.box_limits().max_limit(0);
        real_type const & psi_max    = bone.box_limits().max_limit(1);
        real_type const & theta_max  = bone.box_limits().max_limit(2);




        quaternion_type orientation;

        real_type const delta_phi = (phi_max - phi_min)/(N -1u);
        real_type const delta_psi = (psi_max - psi_min)/(N -1u);
        real_type const delta_theta = (theta_max - theta_min)/(N -1u);

        glPushMatrix();
        Transform( bone.relative().T() );

        ColorPicker(0.5,1.0,0.5);

        if( phi_min < phi_max && psi_min < psi_max)
        {


       real_type theta = theta_min;
        for(size_t k=0;k<N;++k)
        {
          
          real_type phi = phi_min;
          for(size_t j=0;j<N;++j)
          {
            real_type psi = psi_min;
            for(size_t i=0;i<N;++i)
            {
              orientation = OpenTissue::math::Rz( phi ) * OpenTissue::math::Ry( psi )* OpenTissue::math::Rz( theta );
              vector3_type p = orientation.rotate( T_child );  
              
              //vector3_type p = orientation.rotate( vector3_type(1.0,0.0,0.0) );            
              vector3_type n = unit( p );
              //DrawPoint( p );
              nx[k] = n(0);
              ny[k] = n(1);
              nz[k] = n(2);
              
              x[k] = p(0);
              y[k] = p(1);
              z[k] = p(2);
              ++k;
              psi += delta_psi;
            }
            phi += delta_phi;
          }
          theta += delta_theta;
        }
        }


        //real_type phi = phi_min;
        //for(size_t j=0;j<N;++j)
        //{
        //  real_type psi = psi_min;
        //  for(size_t i=0;i<N;++i)
        //  {
        //    size_t k = j*N+i;
        //    orientation = OpenTissue::math::Rz( phi ) * OpenTissue::math::Ry( psi ) * OpenTissue::math::Rz( theta_min );
        //    vector3_type p = orientation.rotate( T_child );
        //    vector3_type n = unit( p );
        //    
        //    nx[k] = n(0);
        //    ny[k] = n(1);
        //    nz[k] = n(2);
        //    
        //    x[k] = p(0);
        //    y[k] = p(1);
        //    z[k] = p(2);
        //    psi += delta_psi;
        //  }
        //  phi += delta_phi;
        //}

        //
        //ColorPicker(1.0,0.5,0.5);

        glBegin(GL_QUADS);
        for(size_t j=0;j<(N-1u);++j)
        {
          for(size_t i=0;i<(N-1u);++i)
          {
            size_t i0 = j*N + i;
            size_t i1 = i0  + 1u;
            size_t i2 = i1  + N;
            size_t i3 = i2  - 1u;

            //glVertex3f(  x[i0],  y[i0],  z[i0] );
            glVertex3f(  x[i0],  y[i0],  z[i0] );
            glNormal3f( nx[i0], ny[i0], nz[i0] );

            //glVertex3f(x[i1],y[i1],z[i1]);
            glVertex3f( x[i1],y[i1],z[i1]);
            glNormal3f(nx[i1],ny[i1],nz[i1]);

            //glVertex3f(x[i2],y[i2],z[i2]);
            glVertex3f( x[i2],y[i2],z[i2]);
            glNormal3f(nx[i2],ny[i2],nz[i2]);

            //glVertex3f(x[i3],y[i3],z[i3]);
            glVertex3f( x[i3],y[i3],z[i3]);
            glNormal3f(nx[i3],ny[i3],nz[i3]);

          }
          
        }
        glEnd();

        glPopMatrix();
      }


      else if(bone.type() == bone_traits::hinge_type)
      {
        real_type const & theta_min  = bone.box_limits().min_limit(0);
        real_type const & theta_max  = bone.box_limits().max_limit(0);

       size_t const N = 10;

        float x[N];
        float y[N];
        float z[N];

        real_type delta_theta = (theta_max - theta_min)/(N -1u);

        bone_type * child = OpenTissue::skeleton::BoneAccess::get_first_child_ptr( &bone );
        vector3_type T = vector3_type(value_traits::zero(), value_traits::zero(), value_traits::zero() );        
        if(child)
          T = unit( child->bind_pose().T() );

        real_type theta = theta_min;
        for(size_t i=0;i<N;++i)
        {
          vector3_type p = OpenTissue::math::Ru( theta, bone.u() )*T;

          x[i] = r*p(0);
          y[i] = r*p(1);
          z[i] = r*p(2);

          theta += delta_theta;
        }

        glPushMatrix();
        Transform( bone.relative().T() );
        // draw the axis of rotation
        ColorPicker(1.0,0.0,0.0);
        DrawVector(vector3_type( 0, 0, 0 ), bone.u() , 0.5 , false);
        ColorPicker(0.0,0.0,1.0);
        //draw the fan of movement 
        vector3_type max_limit ;
        vector3_type min_limit ;
        vector3_type starting = vector3_type(0,1,0);
        
        //were using the rodriques formula to rotate the min and max vectors around the bones u axis  
        vector3_type w = cross( bone.u() , cross(bone.u() ,starting));
        w = dot(w,starting)*w;
        max_limit = unit( starting * cos(bone.box_limits().max_limit(0)) + cross(bone.u() ,starting)* sin(bone.box_limits().max_limit(0)) + w * (1 - cos(bone.box_limits().max_limit(0))));

        min_limit =  unit(starting * cos(bone.box_limits().min_limit(0)) + cross(bone.u() ,starting)* sin(bone.box_limits().min_limit(0)) + w * (1 - cos(bone.box_limits().min_limit(0))));

        vector3_type norman = - unit(max_limit + min_limit);
        /*DrawVector(vector3_type( 0, 0, 0 ), max_limit , 0.7 , false);
        ColorPicker(0.0,0.0,0.0);
        DrawVector(vector3_type( 0, 0, 0 ), min_limit , 0.7 , false);*/
        glBegin(GL_TRIANGLES);

        
        for(size_t i=0;i<(N-1u);++i)
        {
          glVertex3f(  x[i],  y[i],  z[i] );
          glNormal3f(  bone.u()(0), bone.u()(1), bone.u()(2) );

          size_t i1 = i + 1u;
          glVertex3f(  x[i1],  y[i1],  z[i1] );
          glNormal3f(  bone.u()(0), bone.u()(1), bone.u()(2) );

          glVertex3f( 0.0f, 0.0f, 0.0f );
          glNormal3f(  bone.u()(0), bone.u()(1), bone.u()(2) );
        }
        glEnd();
        glPopMatrix();
      }
      else if(bone.type() == bone_traits::slider_type)
      {
        real_type const & theta_min  = bone.box_limits().min_limit(0);
        real_type const & theta_max  = bone.box_limits().max_limit(0);
        DrawVector( bone.u()*theta_min, bone.u()*(theta_max-theta_min) );
      }
      else
      {
        assert( false || !"DrawJointLimits(): Unrecognized joint type");
      }

      glPopMatrix();
    }


  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_JOINT_LIMITS_H
#endif
