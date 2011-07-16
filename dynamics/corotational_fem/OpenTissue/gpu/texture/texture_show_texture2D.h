#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_SHOW_TEXTURE2D_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_SHOW_TEXTURE2D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/gpu/cg/cg_program.h>

namespace OpenTissue
{
  namespace texture
  {

    /**
    * Show 2D Texture
    * This function displays the specified texture as a full-screen image.
    *
    * @param texture    A pointer to the 2D texture that should be displayed.
    */
    template<typename texture2D_pointer>
    inline void show_texture2D( texture2D_pointer texture )
    {
      static cg::Program  vp; 
      static cg::Program  fp; 

      static std::string path = opentissue_path + "/OpenTissue/gpu/texture";

      bool cull_face_enabled = glIsEnabled(GL_CULL_FACE)?true:false;
      bool depth_test_enabled = glIsEnabled(GL_DEPTH_TEST)?true:false;    
      glDisable( GL_CULL_FACE );
      glDisable( GL_DEPTH_TEST );

      if(!vp.is_program_loaded())
        vp.load_from_file(cg::Program::vertex_program,path + "/vp_show_texture2D.cg");
      vp.enable();

      if(!fp.is_program_loaded())
        fp.load_from_file(cg::Program::fragment_program,path + "/fp_show_texture2D.cg");
      fp.set_input_texture("texture", texture);
      fp.enable();

      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
      glBegin( GL_QUADS );
      {
        glTexCoord2f( 0, 0 );
        glVertex3f( -1, -1, 0 );
        glTexCoord2f( texture->width(), 0 );
        glVertex3f( 1, -1, 0 );
        glTexCoord2f( texture->width(), texture->height() );
        glVertex3f( 1, 1, 0 );
        glTexCoord2f( 0, texture->height() );
        glVertex3f( -1, 1, 0 );
      }
      glEnd();

      vp.disable();
      fp.disable();

      if(cull_face_enabled)
        glEnable(GL_CULL_FACE);
      if(depth_test_enabled)
        glEnable(GL_DEPTH_TEST);
    }

  } // namespace texture

} // namespace OpenTissue

// OPENTISSUE_GPU_TEXTURE_TEXTURE_SHOW_TEXTURE2D_H
#endif

