#ifndef OPENTISSUE_UTILITY_GL_GL_CHECK_ERRORS_H
#define OPENTISSUE_UTILITY_GL_GL_CHECK_ERRORS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include<iostream>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * A simple OpenGL error checking routine.
    * This compiles away to a no-op method if the NDEBUG preprocessor symbol
    * is defined during compilation.
    * @param location  Optional. A string that can be used to indicate the location
    *                  where the error check occurs.
    * @param ostr      Optional. Determines the destination of the error message.
    *                  Defaults to cerr, but could also be a file.
    */
#ifndef NDEBUG
    inline void gl_check_errors( const char* location = 0, std::ostream& ostr = std::cerr )
    {
      GLuint errnum;
      const char *errstr;
      while ( (errnum = glGetError()) != 0 )
      {
        errstr = reinterpret_cast<const char *>(gluErrorString(errnum));
        if (errstr)
          ostr << errstr;
        else
          ostr << "<unknown err: " << errnum << ">";

        if(location)
          ostr << " at " << location;
        ostr << std::endl;
      }
      return;
    }
#else
    inline void gl_check_errors(  )
    {}
    inline void gl_check_errors( const char* )
    {}
    inline void gl_check_errors( const char* , std::ostream&  )
    {}
#endif

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_CHECK_ERRORS_H
#endif
