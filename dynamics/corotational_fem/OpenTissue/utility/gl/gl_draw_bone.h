#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/core/math/math_constants.h>

#include <cassert>


namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Bone.
    * Orientations and translations are simply drawn as arrows and coordinate frames.
    *
    * @param bone       A reference to the bone that should be drawn.
    */
    template<typename bone_type>
    inline void DrawBone(bone_type const & bone)
    {
      glPushMatrix();
      if(!bone.is_root())
      {
        Transform(bone.parent()->absolute());
      }
      DrawVector(bone.relative().T());
      DrawFrame(bone.relative());
      glPopMatrix();
    }

    /**
    * Draw Bone.
    * Mimics Maya's bone visualization. Translations are drawn
    * as pyramids and orientaions as circular arcs on a sphere.
    *
    * Observe that the pyramid for the root bone is not drawn. This
    * may cause one to think that a bone consist of a ``sphere-pyramid''
    * pairing. This is however misleadning the true pairing that match
    * the underlying bone transform is ``pyramid-sphere''.
    *
    *
    * @param bone       A reference to the bone that should be drawn.
    */
    template<typename bone_type>
    inline void DrawFancyBone(bone_type const & bone)
    {
      using std::sqrt;
      using std::acos;
      using std::cos;
      using std::sin;

      typedef typename bone_type::math_types       math_types;
      typedef typename math_types::real_type       real_type;
      typedef typename math_types::vector3_type    vector3_type;
      typedef typename math_types::quaternion_type quaternion_type;
      typedef typename math_types::value_traits    value_traits;

	  
      vector3_type const & T = bone.relative().T();
      quaternion_type Q;
      Q = bone.relative().Q();
      real_type length = sqrt( dot(T,T) );
      real_type radius = 0.075 * length;

      glPushMatrix();
      if(!bone.is_root())
      {
        Transform(bone.convert(bone.parent()->absolute()));
      }

      {
        glPushMatrix();
        Transform(T,Q);
        int const N = 24;
        real_type delta_theta = 2*value_traits::pi()/N;

        real_type theta = 0;

        ColorPicker(1,0,0,1);
        glBegin(GL_LINE_LOOP);
        for(int i=0;i<N;++i)
        {
          glNormal3f(1,0,0);
          glVertex3f( 0, radius*cos(theta),radius*sin(theta));
          theta += delta_theta;
        }
        glEnd();
        theta = 0;
        ColorPicker(0,1,0,1);
        glBegin(GL_LINE_LOOP);
        for(int i=0;i<N;++i)
        {
          glNormal3f(0,1,0);
          glVertex3f( radius*cos(theta),0,radius*sin(theta));
          theta += delta_theta;
        }
        glEnd();
        theta = 0;
        ColorPicker(0,0,1,1);
        glBegin(GL_LINE_LOOP);
        for(int i=0;i<N;++i)
        {
          glNormal3f(0,1,0);
          glVertex3f( radius*cos(theta), radius*sin(theta), 0);
          theta += delta_theta;
        }
        glEnd();
        glPopMatrix();
      }
      {
        if(!bone.is_root())
        {
          GLfloat angle = 0;
          GLfloat x = 1;
          GLfloat y = 0;
          GLfloat z = 0;

          if ( ( T(0) == 0 ) && ( T(1) == 0 ) )
          {
            if ( T(2) > 0 )
            {
              angle = 0;
            }
            else
            {
              angle = 180;
            }
          }
          else
          {
            vector3_type v_unit;
            vector3_type axis;

            v_unit = unit( T );
            axis = unit( cross( T, vector3_type(0,0,1 ) ) );
            angle = acos( dot(v_unit , vector3_type(0,0,1 ) ) );
            angle = -180.0 * angle / value_traits::pi();
            x = axis(0);
            y = axis(1);
            z = axis(2);
          }

          glPushMatrix();
          glRotatef( angle, x, y, z );
          glRotatef( 45, 0, 0, 1 );

          real_type base = radius / 2.0;

          ColorPicker(0,0,.5,1);
          glBegin(GL_LINE_LOOP);
          glNormal3f(0,0,-1);
          glVertex3f( -base, -base, radius);
          glNormal3f(0,0,-1);
          glVertex3f( -base,  base, radius);
          glNormal3f(0,0,-1);
          glVertex3f(  base,  base, radius);
          glNormal3f(0,0,-1);
          glVertex3f(  base, -base, radius);
          glEnd();

          glBegin(GL_LINE_LOOP);
          glNormal3f(0,-1,0);
          glVertex3f( -base, -base, radius);
          glNormal3f(0,-1,0);
          glVertex3f(  base, -base, radius);
          glNormal3f(0,-1,0);
          glVertex3f(  0, 0, length-radius);
          glEnd();

          glBegin(GL_LINE_LOOP);
          glNormal3f(1,0,0);
          glVertex3f(  base, -base, radius);
          glNormal3f(1,0,0);
          glVertex3f(  base,  base, radius);
          glNormal3f(1,0,0);
          glVertex3f(  0, 0, length-radius);
          glEnd();

          glBegin(GL_LINE_LOOP);
          glNormal3f(0,1,0);
          glVertex3f(  base, base, radius);
          glNormal3f(0,1,0);
          glVertex3f( -base, base, radius);
          glNormal3f(0,1,0);
          glVertex3f(  0, 0, length-radius);
          glEnd();

          glBegin(GL_LINE_LOOP);
          glNormal3f(-1,0,0);
          glVertex3f( -base,  base, radius);
          glNormal3f(-1,0,0);
          glVertex3f( -base, -base, radius);
          glNormal3f(-1,0,0);
          glVertex3f(  0,0, length-radius);
          glEnd();

          glPopMatrix();

        }
      }

      glPopMatrix();
    }

    /**
    * Draw Stick Bone.
    * Orientations and translations are simply drawn as arrows and coordinate frames.
    *
    * @param bone       A reference to the bone that should be drawn.
    */
    template<typename bone_type>
    inline void DrawStickBone(bone_type const & bone ,float red = 0.7,float green = 0.7,float blue = 0.7)
    {
      ColorPicker(red,green,blue,1.0);
      glPushMatrix();
      if(!bone.is_root())
      {
        Transform(bone.parent()->absolute());
      }
      DrawVector(bone.relative().T());

      glPopMatrix();
    }

    /**
    * Draw Fat Bone.
    * the bone is drawn as a capsule the color can be chosen or left at the default light grey.
    *
    * @param bone       A reference to the bone that should be drawn.
    */
    template<typename bone_type,typename capsule_type >
    inline void DrawFatBone(bone_type const & bone ,capsule_type capsule,float red = 0.7,float green = 0.7,float blue = 0.7,float thickness = 0.2)
    {
      typedef typename bone_type::math_types     math_types;
      typedef typename math_types::vector3_type  V;
      typedef typename math_types::real_type     T;
      typedef typename math_types::value_traits  value_traits;
      
      V const zero = V(value_traits::zero(), value_traits::zero(), value_traits::zero());

      glPushMatrix();
      if(!bone.is_root())
      {
        Transform(bone.parent()->absolute());
        ColorPicker(red,green,blue,1.0);

        //T thickness = 0.2;
        if(length(bone.relative().T()) < 0.001) 
          thickness = 0.3;

        capsule.set(zero, bone.relative().T(), thickness);

        DrawCapsule(capsule);
        //DrawVector(bone.relative().T());

      }
      glPopMatrix();
    }


  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_H
#endif
