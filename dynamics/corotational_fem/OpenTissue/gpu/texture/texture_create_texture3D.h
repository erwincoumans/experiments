#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE3D_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE3D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/gpu/texture/texture_texture3D.h>
#include <OpenTissue/gpu/texture/texture_types.h>

namespace OpenTissue
{
  namespace texture
  {
    /**
    * Create 3D texture from Grid.
    * This function choices the lowest power-of 2 texture fitting the grid-dimensions best.
    * 
    * @param grid   The grid from which texture data should be loaded.
    *
    * @return      A pointer to the newly created 3D texture.
    */
    template<typename grid_type>
    inline texture3D_pointer create_texture3D_from_grid( grid_type & grid )
    {
      typedef typename grid_type::value_type T;
      unsigned int channels = 1;  //--- Grid only supports one-channel!!!
      unsigned int int_format = internal_format<T>( channels );
      unsigned int ext_format = external_format( channels );
      unsigned int ext_type   = external_type<T>();
      bool cubic = false;
      float scale = 16.0;
      texture3D_pointer tex(
        new Texture3D(
        int_format
        , grid.I()
        , grid.J()
        , grid.K()
        , ext_format
        , ext_type
        , grid.data()
        , cubic
        , scale
        )
        );
      return tex;
    }


    /** 
    * Create Empty Texture.
    * Example usage: 
    *
    *  texture3D_poitner tex = create_texture3D<unsigned short>(32,32,32);
    * 
    * @param texture_size_i   The number of pixels in a row of the texture.
    * @param texture_size_j   The number of rows in an image of the texture.
    * @param texture_size_k   The number of images in the texture.
    *
    * @return      A pointer to the newly created 3D texture.
    */
    template<typename T>
    inline texture3D_pointer create_texture3D(  int texture_size_i,  int texture_size_j,   int texture_size_k  );

    template<>
    inline texture3D_pointer create_texture3D<unsigned short>( int texture_size_i, int texture_size_j, int texture_size_k )
    {
      typedef unsigned short T;
      unsigned int channels = 1;
      unsigned int int_format = internal_format<T>( channels );
      unsigned int ext_format = external_format( channels );
      unsigned int ext_type   = external_type<T>();

      bool cubic = false;
      float scale = 16.0;
      texture3D_pointer tex(
        new Texture3D(
        int_format
        , texture_size_i
        , texture_size_j
        , texture_size_k
        , ext_format
        , ext_type
        , 0
        , cubic
        , scale
        )
        );
      return tex;
    }

  } // namespace texture
} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE3D_H
#endif
