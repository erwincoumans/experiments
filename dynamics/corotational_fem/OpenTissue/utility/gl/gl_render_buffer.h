#ifndef OPENTISSUE_UTILITY_GL_GL_RENDER_BUFFER_H
#define OPENTISSUE_UTILITY_GL_GL_RENDER_BUFFER_H
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
    * Render Buffer. 
    * This class encapsulates the render buffer OpenGL object described in the
    * frame buffer object (FBO) OpenGL spec. See the official spec at:
    * http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt
    * for complete details.
    *
    * A "render buffer" is a chunk of GPU memory used by frame buffer objects to
    * represent "traditional" framebuffer memory (depth, stencil, and color buffers).
    * By "traditional," we mean that the memory cannot be bound as a texture. 
    * With respect to GPU shaders, Render buffer memory is "write-only." Framebuffer
    * operations such as alpha blending, depth test, alpha test, stencil test, etc.
    * read from this memory in post-fragement-shader (ROP) operations.
    *
    * The most common use of render buffers is to create depth and stencil buffers.
    * Note that as of 7/1/05, NVIDIA drivers to do not support stencil Renderbuffers.
    *
    * Usage Notes:
    * 1) "internal_format" can be any of the following: Valid OpenGL internal formats
    * beginning with: RGB, RGBA, DEPTH_COMPONENT
    *
    * or a stencil buffer format (not currently supported  in NVIDIA drivers as of 7/1/05).
    * STENCIL_INDEX1_EXT 
    * STENCIL_INDEX4_EXT     
    * STENCIL_INDEX8_EXT     
    * STENCIL_INDEX16_EXT
    *
    * This implementation was inspired by the ideas in the FBO class from
    * GPUGP by Aaron Lefohn.
    */
    class Renderbuffer
    {
    protected:

      GLuint m_buffer_id;  ///<  The render buffer identifier of this render buffer.

    protected:

      /**
      * Create Unused Render Buffer identifier.
      *
      * @return    The unused identifier.
      */
      static GLuint create_buffer_id() 
      {
        GLuint id = 0;
        glGenRenderbuffersEXT(1, &id);
        gl::gl_check_errors("RenderBuffer::create_buffer_id(): glGenRenderbuffersEXT");
        return id;
      }

    public:

      Renderbuffer()
        : m_buffer_id( create_buffer_id() )
      {}

      /**
      * Specialized Constructor.
      *
      * @param internal_format   The format of the render buffer: RGB, RGBA, DEPTH_COMPONENT etc..
      * @param width             The number of pixels in a row.
      * @param height            The number of rows in the buffer.
      */
      Renderbuffer(GLenum internal_format, int width, int height)
        : m_buffer_id( create_buffer_id() )
      {
        set(internal_format, width, height);
      }

      ~Renderbuffer(){	glDeleteRenderbuffersEXT(1, &m_buffer_id); }

    public:

      void bind() 
      {	
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_buffer_id);
        gl::gl_check_errors("RenderBuffer::bind(): glBindRenderbufferEXT");
      }
      void unbind() 
      {	
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0); 
        gl::gl_check_errors("RenderBuffer::unbind(): glBindRenderbufferEXT");
      }

      /**
      * Set Render Buffer Format and Size.
      *
      * @param internal_format   The format of the render buffer: RGB, RGBA, DEPTH_COMPONENT etc..
      * @param width             The number of pixels in a row.
      * @param height            The number of rows in the buffer.
      */
      void set(GLenum internal_format, int width, int height)
      {
        int max_size = Renderbuffer::get_max_size();
        if (width > max_size || height > max_size ) 
        {
          std::cerr << "Renderbuffer.set(): " << "Size too big (" << width << ", " << height << ")" << std::endl;
          return;
        }
        //--- Guarded bind
        GLint tmp = 0;
        glGetIntegerv( GL_RENDERBUFFER_BINDING_EXT, &tmp );
        gl::gl_check_errors("RenderBuffer::set(): glGetIntegerv");
        GLuint saved_id = static_cast<GLuint>(tmp);
        if (saved_id != m_buffer_id) 
        {
          bind();
        }
        //--- Allocate memory for renderBuffer
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, internal_format, width, height );
        gl::gl_check_errors("RenderBuffer::set(): glRenderbufferStorageEXT");
        //--- Guarded unbind
        if (saved_id != m_buffer_id) 
        {
          unbind();
        }
      }

      /**
      * Get Identifier.
      *
      * @return    The identifier of the render buffer.
      */
      GLuint get_id() const {	return m_buffer_id; }

    public:

      /**
      * Get Maximum Size.
      *
      * @return   The maximum size of a render buffer. I.e. the maximum
      *           number of ``pixels'' in widht and height arguments.
      */
      static GLint get_max_size()
      {
        GLint max_size = 0;
        glGetIntegerv( GL_MAX_RENDERBUFFER_SIZE_EXT, &max_size );
        gl::gl_check_errors("RenderBuffer::get_max_size(): glGetIntegerv");
        return max_size;
      }

    };

    typedef Renderbuffer*  renderbuffer_pointer;

    /**
    * Create Depth Buffer.
    * This method is a convenience tool, it makes it easier to
    * create a render buffer representing a depth buffer.
    *
    * @param width     The number of pixels in a row of the depth buffer.
    * @param height    The number of rows in the depth buffer.
    */
    inline renderbuffer_pointer create_depth_buffer(unsigned int width, unsigned int height)
    {
      renderbuffer_pointer buffer;
      buffer = new Renderbuffer();
      buffer->set(GL_DEPTH_COMPONENT24,width, height);    
      return buffer;
    }

    /**
    * Create Stencil Buffer.
    * This method is a convenience tool, it makes it easier to
    * create a render buffer representing a stencil buffer.
    *
    * @param width     The number of pixels in a row of the stencil buffer.
    * @param height    The number of rows in the stencil buffer.
    */
    inline renderbuffer_pointer create_stencil_buffer(unsigned int width, unsigned int height)
    {
      renderbuffer_pointer buffer;
      buffer = new Renderbuffer();
      buffer->set(GL_STENCIL_INDEX16_EXT,width, height);
      return buffer;
    }


    /**
    * Create Stencil Buffer.
    * This method is a convenience tool, it makes it easier to
    * create a render buffer representing a packed depth and stencil buffer.
    *
    * @param width     The number of pixels in a row of the buffer.
    * @param height    The number of rows in the buffer.
    */
    inline renderbuffer_pointer create_packed_depth_and_stencil_buffer(unsigned int width, unsigned int height)
    {
      renderbuffer_pointer buffer;
      buffer = new Renderbuffer();
      //GL_DEPTH24_STENCIL8_EXT OR DEPTH_STENCIL_EXT?
      buffer->set(GL_DEPTH24_STENCIL8_EXT,width, height);
      return buffer;
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_RENDER_BUFFER_H
#endif
