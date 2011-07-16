#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_RESCALE_RANGE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_RESCALE_RANGE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <cmath>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Rescale input range to output range.
    *
    * @param grid      The grid to re-scale.
    * @param src_min Scale factor in the I-direction of the grid.
    * @param src_max Scale factor in the J-direction of the grid.
    * @param dst_min Scale factor in the K-direction of the grid.
    * @param dst_max Scale factor in the K-direction of the grid.
    */
    template <typename grid_type>
    inline void rescale_range(
      grid_type & grid
      , typename grid_type::value_type src_min
      , typename grid_type::value_type src_max
      , typename grid_type::value_type dst_min
      , typename grid_type::value_type dst_max
      )
    {
      // TODO: Replace float with boost::type-traits magic "next-higher datatype"
      typedef typename grid_type::value_type value_type;

      std::cout << "-- Normalizing from range [" << src_min << ", " << src_max
        << "] to range [" << dst_min << ", " << dst_max << "]" << std::endl;
      float src_scale = 1.0f / ( src_max - src_min );
      float val;

      value_type * t = grid.data();
      for ( size_t i = 0; i < grid.size(); ++i, ++t )
      {
        val = static_cast<float>(*t);
        val = (val<src_min) ? src_min : ( (val>src_max) ? src_max : val );
        val = src_scale * ( val - src_min );
        val = dst_min + val*( dst_max - dst_min );
        *t = static_cast<value_type>(val);
      }
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_RESCALE_RANGE_H
#endif
