#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_PATH_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_PATH_H
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
    * Draw Bone Path.
    * Draw bone path as a sequence of points. This is a good tool for visualization motion samples.
    *
    * @param bone       A reference to the bone that should be drawn.
    * @param poses      A transform container, containing sampled poses of the bone motion.
    * @param radius     The radius size of the path being drawn. Default value is 0.02.
    */
    template<typename bone_type ,typename transform_container>
    inline void DrawBonePath(bone_type const & bone , transform_container const & poses, double const radius = 0.02)
    {
      typedef typename bone_type::vector3_type   V;

      V const draw_zone = bone.relative().T();

      glPushMatrix();
      if(!bone.is_root())
      {
        Transform(bone.parent()->absolute());
      }
      Transform(bone.relative().T());

      typename transform_container::const_iterator pose = poses.begin();
      typename transform_container::const_iterator end  = poses.end();

      size_t const N = std::distance(pose,end);

      float const intensity_increment = 1.0f / N;
      float       intensity           = 0.0f;

      for(;pose!=end;++pose)
      {
        glPushMatrix();
        Transform(pose->Q());        
        ColorPicker(intensity*0.5f, intensity, 1.0f);        
        DrawPoint(draw_zone,radius); 
        glPopMatrix();
        intensity += intensity_increment;
      }
      glPopMatrix();
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_BONE_PATH_H
#endif
