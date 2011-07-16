#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_CHAIN_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_CHAIN_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>

namespace OpenTissue
{
  namespace gl
  {

    /**
    * Draw Chain.
    *
    * @param chain       A reference to the chain that should be drawn.
    */
    template<typename chain_type>
    inline void DrawChain(chain_type const & chain)
    {
       typedef typename chain_type::math_types    math_types;
       typedef typename math_types::vector3_type  vector3_type;
       typedef typename math_types::real_type     real_type;

       vector3_type origin      = chain.get_root()->absolute().T();
       vector3_type destination = chain.get_end_effector()->absolute().T();
       vector3_type p_global    = chain.p_global();
       vector3_type x_global    = chain.x_global();
       vector3_type y_global    = chain.y_global();

       ColorPicker(1.0,0.6,0.0);
       DrawVector(origin,destination-origin,0.5);

       ColorPicker(1.0,0.0,1.0);
       DrawVector(origin,p_global-origin,0.5);
       
       ColorPicker(1.0,0.0,0.0);
       DrawVector(p_global,x_global,0.5);
       
       ColorPicker(0.0,1.0,0.0);
       DrawVector(p_global,y_global,0.5);       
    }
  
  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_CHAIN_H
#endif
