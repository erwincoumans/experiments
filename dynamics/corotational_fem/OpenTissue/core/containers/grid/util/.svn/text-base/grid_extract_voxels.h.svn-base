#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_EXTRACT_VOXELS_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_EXTRACT_VOXELS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Extract Voxels using Transfer Function.
    * The transfer function grids voxels to colors, this
    * function creates a new voxel grid containining all
    * visible colored voxels.
    *
    * @param in       A reference to a input voxel grid.
    * @param func     A reference to the transfer function that should be used.
    * @param out      Upon return this argument holds the colored voxels.
    */
    template < typename grid_type_in, typename transfer_function, typename grid_type_out >
    inline void extract_voxels(
      grid_type_in       const& in
      , transfer_function const& func
      , grid_type_out           & out
      )
    {
      typedef typename grid_type_in::value_type       value_type_in;
      typedef typename grid_type_out::value_type      value_type_out;
      typedef typename grid_type_in::math_types_in    math_types_in;
      typedef typename math_types_in::vector3_type    vector3_type_in;

      typedef typename grid_type_in::const_index_iterator const_index_iterator_in;

      std::cout << "-- OpenTissue::extract_voxels()" << std::endl;

      int entries = func.getSize();

      // TODO: henrikd 2005-06-30 - Find proper way to convert from range in grid
      //       to range in table.
      value_type_in data_min = OpenTissue::grid::min_element(in);
      value_type_in data_max = OpenTissue::grid::max_element(in);

      if ( data_min < 0 )  // TODO: This check is always false when the datatype is unsigned! Workaround needed!
        std::cout << "OpenTissue::extract_voxels(): WARNING, data-range not compatible with Colortable (min<0)!" << std::endl;

      if ( data_max > entries )
        std::cout << "OpenTissue::extract_voxels(): WARNING, data-range not compatible with Colortable (max>entries)!"  << std::endl;

      size_t count = 0;
      const_index_iterator_in voxel;
      for ( voxel = in.begin(); voxel != in.end(); ++voxel )
      {
        int index = static_cast<int>( *voxel );
        float r = *static_cast<float*>( func.get( index, 0 ) );
        float g = *static_cast<float*>( func.get( index, 1 ) );
        float b = *static_cast<float*>( func.get( index, 2 ) );
        float alpha = *static_cast<float*>( func.get( index, 3 ) );
        // TODO: Comparing floats with == or != is not safe
        if ( alpha && ( r || g || b ) )
        {
          out( voxel.get_index() ) = value_type_out(0);
          ++count;
        }
        else
          out( voxel.get_index() ) = out.infinity();
      }
      
      std::cout << "OpenTissue::extract_voxels(): Found "
        << count
        << " voxels from "
        << in.size()
        << " possible"
        << std::endl;

      std::cout << "OpenTissue::extract_voxels(): Done" << std::endl;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_EXTRACT_VOXELS_H
#endif
