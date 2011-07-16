#ifndef OPENTISSUE_GPU_TEXTURE_GL_IS_FLOAT_TEXTURE_SUPPORTED_H
#define OPENTISSUE_GPU_TEXTURE_GL_IS_FLOAT_TEXTURE_SUPPORTED_H
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
    * Test if floating point texture formats are supported.
    * This is a convenience function that makes it easier for
    * end users to determine whether floating point textures
    * are supported by their opengl drivers.
    *
    * @return  If supported then thereturn value is true
    *          otherwise it is false.
    */
    inline bool is_float_texture_supported()
    {
      // read more here http://opengl.org/registry/specs/ARB/texture_float.txt
      //if (glewIsSupported("GL_ARB_texture_float"))
      //{
      //  return true;
      //}
      // and here http://opengl.org/registry/specs/NV/float_buffer.txt
      if (glewIsSupported("GL_NV_float_buffer"))
      {
        return true;
      }
      return false;
    }

  } //namespace gl

} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_GL_IS_FLOAT_TEXTURE_SUPPORTED_H
#endif
