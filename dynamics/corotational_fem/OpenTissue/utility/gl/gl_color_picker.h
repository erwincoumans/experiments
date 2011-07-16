#ifndef OPENTISSUE_UTILITY_GL_GL_COLOR_PICKER_H
#define OPENTISSUE_UTILITY_GL_GL_COLOR_PICKER_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * OpenGL Color Picker.
    */
    template<typename T1,typename T2,typename T3,typename T4>
    inline void ColorPicker( T1 const red, T2 const green, T3 const blue, T4 const alpha, unsigned int face = GL_FRONT_AND_BACK)
    {
      if ( glIsEnabled( GL_COLOR_MATERIAL ) || !glIsEnabled( GL_LIGHTING ) )
      {
        glColor4f( (GLfloat) red, (GLfloat) green, (GLfloat) blue, (GLfloat) alpha );
      }
      else
      {
        GLfloat color[] = {(GLfloat) red, (GLfloat) green, (GLfloat) blue, (GLfloat) alpha};
        glMaterialfv( face, GL_DIFFUSE, color );
      }
    }

    template<typename T1,typename T2,typename T3>
    inline void ColorPicker( T1 const red, T2 const green, T3 const blue)
    {
      ColorPicker(red,green,blue,0.0,GL_FRONT_AND_BACK);
    }

    template<typename vector3_type>
    inline void ColorPicker( vector3_type const rgb )
    {
      ColorPicker(rgb(0), rgb(1), rgb(2), 0.0, GL_FRONT_AND_BACK);
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_COLOR_PICKER_H
#endif
