#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CONNECTED_COMPONENTS_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CONNECTED_COMPONENTS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <boost/multi_array.hpp>

#include <vector>

namespace OpenTissue
{
  namespace grid
  {

    namespace detail
    {
      /**
      * Associate two adjacent components so that they represent the same component.
      * @param i Component label value of a pixel under consideration.
      * @param j Component label value of an adjacent pixel.
      * @param assoc Array containing associations of component label values.
      */
      template< typename unsigned_int_container>
      inline void associate(size_t i, size_t j, unsigned_int_container & assoc)
      {
        if ( j == 0 || j == i )
          return;

        size_t search = j;
        do
        {
          search = assoc[search];
        }
        while ( search != j && search != i );

        if ( search == j )
        {
          std::swap(assoc[i],assoc[j]);
        }
      }
    } // namespace detail

    /**
    * Compute the 8-connected components of a binary image.
    * The components in the returned image are labeled with positive integer values.
    * If the image is identically zero, then the components image is identically
    * zero and the returned quantity is zero.
    *
    * @see    grid/util/connected_components2D.h for details.
    *
    * @param  image      Input image. Should be binary, with 1 representing data.
    * @param  components Output image. Each components is labelled with an unique identifier.
    * @return Number of components found.
    */
    template < typename grid_type >
    inline size_t connected_components(grid_type const & image, grid_type & components)
    {
      components.create(image.I(), image.J(), image.K(), 1,1,1);

      // Create a temporary copy of image to store intermediate information during boundary extraction.
      // The original image is embedded in an image with two more rows, two more columns and two more slices
      // so that the image boundary pixels are properly handled.
      typedef boost::multi_array<size_t, 3> array_type;
      typedef array_type::index array_index;

      size_t width = image.I()+2;
      size_t height = image.J()+2;
      size_t depth = image.K()+2;
      
      array_type tmp(boost::extents[width][height][depth]);
      {
        array_index xp1, yp1, zp1;
        size_t x0,y0, z0;
        for (z0 = 0, zp1 = 1; z0 < image.K(); ++z0, ++zp1)
          for (y0 = 0, yp1 = 1; y0 < image.J(); ++y0, ++yp1)
            for (x0 = 0, xp1 = 1; x0 < image.I(); ++x0, ++xp1)
              tmp[xp1][yp1][zp1] = ( image(x0,y0,z0) ? 1 : 0 );
      }

      // label connected components in 1D array
      size_t initial_component_count = 0;

      size_t * val = tmp.data();
      for (size_t i = 0; i < tmp.num_elements(); ++i, ++val)
      {
        if ( *val )
        {
          initial_component_count++;
          while ( *val > 0 )
          {
            // loop terminates since tmp is zero on its boundaries
            *val = initial_component_count;
            ++i;
            ++val;
          }
        }
      }

      if ( initial_component_count == 0 )
      {
        // input image is identically zero
        std::fill(components.begin(), components.end(), 0);
        return 0;
      }

      // associative memory for merging
      std::vector<size_t> assoc(initial_component_count+1,0);
      for (size_t i = 0; i < initial_component_count + 1; ++i)
        assoc[i] = i;

      // Merge equivalent components.
      // Voxel (x,y,z) has previous neighbors:
      // (x-1,y-1,z-1), (x,y-1,z-1), (x+1,y-1,z-1),
      // (x-1,y,z-1), (x,y,z-1), (x+1,y,z-1),
      // (x-1,y+1,z-1), (x,y+1,z-1), (x+1,y+1,z-1),
      // (x-1,y-1,z), (x,y-1,z), (x+1,y-1,z),
      // (x-1,y,z)
      // [13 of 26 voxels visited before (x,y,z) is visited, get component labels from them].
      for (array_index z = 1; z < depth-1; ++z)
        for (array_index y = 1; y < height-1; ++y)
          for (array_index x = 1; x < width-1; ++x)
          {
            size_t value = tmp[x][y][z];
            if ( value > 0 )
            {
              detail::associate(value, tmp[x-1][y-1][z-1], assoc);
              detail::associate(value, tmp[x  ][y-1][z-1], assoc);
              detail::associate(value, tmp[x+1][y-1][z-1], assoc);
              detail::associate(value, tmp[x-1][y  ][z-1], assoc);
              detail::associate(value, tmp[x  ][y  ][z-1], assoc);
              detail::associate(value, tmp[x+1][y  ][z-1], assoc);
              detail::associate(value, tmp[x-1][y+1][z-1], assoc);
              detail::associate(value, tmp[x  ][y+1][z-1], assoc);
              detail::associate(value, tmp[x+1][y+1][z-1], assoc);
              detail::associate(value, tmp[x-1][y-1][z  ], assoc);
              detail::associate(value, tmp[x  ][y-1][z  ], assoc);
              detail::associate(value, tmp[x+1][y-1][z  ], assoc);
              detail::associate(value, tmp[x-1][y  ][z  ], assoc);
            }
          }

          // replace each cycle of equivalent labels by a single label
          size_t component_count = 0;
          for (size_t i = 1; i <= initial_component_count; ++i)
          {
            if ( i <= assoc[i] )
            {
              component_count++;
              size_t current = i;
              while ( assoc[current] != i )
              {
                size_t next = assoc[current];
                assoc[current] = component_count;
                current = next;
              }
              assoc[current] = component_count;
            }
          }

          // pack a relabeled image in smaller size output
          {
            array_index xp1, yp1, zp1;
            size_t x0,y0,z0;
            for (z0 = 0, zp1 = 1; z0 < components.K(); ++z0, ++zp1)
              for (y0 = 0, yp1 = 1; y0 < components.J(); ++y0, ++yp1)
                for (x0 = 0, xp1 = 1; x0 < components.I(); ++x0, ++xp1)
                  components(x0,y0,z0) = assoc[ tmp[xp1][yp1][zp1] ];
          }

          return component_count;
    }

  } // namespace grid
} // namespace OpenTissue

//  OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CONNECTED_COMPONENTS_H
#endif
