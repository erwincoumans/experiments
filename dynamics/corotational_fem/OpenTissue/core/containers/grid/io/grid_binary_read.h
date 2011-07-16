#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_READ_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_READ_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <iostream>
#include <string>

#ifdef WIN32
#  include <io.h> //--- KE 31-05-2003: Windows specific...
#endif

namespace OpenTissue
{
  namespace grid
  {
    template <typename grid_type>
    inline bool binary_read(std::string const & filename, grid_type & grid)
    {
      typedef typename grid_type::value_type     value_type;
      typedef typename grid_type::math_types     math_types;
      typedef typename math_types::vector3_type  vector3_type;
      typedef typename math_types::real_type     real_type;

      using std::min;
      using std::max;

      FILE *stream;
      // TODO: fopen is deprecated in VC++8
      if((stream = fopen( filename.c_str(), "rb" )) == NULL )
      {
        std::cerr << "binary_read(): unable to open file " << filename << std::endl;
        return false;
      }

      size_t I,J,K,N;
      if( fread( &I, sizeof( size_t ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): did not read I");
      if( fread( &J, sizeof( size_t ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): did not read J");
      if( fread( &K, sizeof( size_t ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): did not read K");
      if( fread( &N, sizeof( size_t ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): did not read N");

      real_type min_x,min_y,min_z;

      if( fread( &min_x, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read min x");
      if( fread( &min_y, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read min y");
      if( fread( &min_z, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read min z");

      real_type max_x,max_y,max_z;
      if( fread( &max_x, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read max x");
      if( fread( &max_y, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read max y");
      if( fread( &max_z, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read max z");

      real_type dx,dy,dz;
      if( fread( &dx, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read dx");
      if( fread( &dy, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read dy");
      if( fread( &dz, sizeof( real_type ), 1, stream ) != 1)
        throw std::logic_error("binary_read(): could not read dz");

      grid.create(vector3_type(min_x,min_y,min_z), vector3_type(max_x, max_y, max_z), I, J, K);
      if (grid.dx() != dx || grid.dy() != dy || grid.dz() != dz)
      {
        std::cout << "binary_read(): warning: spacing differs between calculated and stored!" << std::endl;
        grid.dx() = dx;
        grid.dy() = dy;
        grid.dz() = dz;
      }

      if( fread( grid.data(), sizeof( value_type ), grid.size(), stream ) != grid.size() )
        throw std::logic_error("binary_read(): could not read data");

      fclose( stream );

      std::cout << "binary_read(): Completed reading file: " << filename
        << ", min = " << OpenTissue::grid::min_element(grid) << " max = " << OpenTissue::grid::max_element(grid) << std::endl;
      return true;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_READ_H
#endif
