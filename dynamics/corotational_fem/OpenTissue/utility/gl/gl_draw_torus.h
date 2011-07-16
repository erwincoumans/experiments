#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_TORUS_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_TORUS_H
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
    * Draw Torus
    *
    * @param torus      A reference to the torus that should be drawn
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename torus_type>
    inline void DrawTorus(torus_type const & torus, bool wireframe = false)
    {
      glPushMatrix();
      glTranslated(torus.center()[0], torus.center()[1], torus.center()[2]);
      if (wireframe)
        glutWireTorus(torus.tube(), torus.radius(), 32, 32);
      else
        glutSolidTorus(torus.tube(), torus.radius(), 32, 32);
      glPopMatrix();
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_TORUS_H
#endif
