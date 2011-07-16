#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_STRING_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_STRING_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

#include <string>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw String.
    * Writes a string at the specified location.
    *
    * Contributted by Adam Hasselblach and Sren Horn and tweaked by Kenny Erleben.
    *
    *  @param str     The string to be drawn
    *  @param T       Location in world coordinates
    *  @param width   Wanted width of text in world space length. If
    *                 zero no rescaling of text is done, this is default
    *                 behavior.
    */
    template<typename vector3_type>
    inline void DrawString(std::string const & str, vector3_type const & T,float width = 0.0f)
    {
      char const * text = str.c_str();
      unsigned int length = str.length();

      float total_width = 0;
      for (unsigned int i = 0;i<=length;++i)
        total_width += glutStrokeWidth(GLUT_STROKE_ROMAN, text[i]);//--- what units are this working in???

      float aspect = 1.0;
      if(width>0)
        aspect = width/total_width;

      glPushMatrix();
      glTranslatef(T(0),T(1),T(2));
      float magic = 0.005;  //--- some magic because I cant figure out the units!!!!
      glScalef(magic,magic,magic);
      for (unsigned int i = 0;i<=length;++i)
        glutStrokeCharacter( GLUT_STROKE_ROMAN, text[i]);
      glPopMatrix();
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_STRING_H
#endif
