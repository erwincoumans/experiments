#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_WRITE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_WRITE_H
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
    inline bool binary_write(std::string const & filename, grid_type & grid)
    {
      typedef typename grid_type::value_type    value_type;
      typedef typename grid_type::math_types    math_types;
      typedef typename math_types::vector3_type vector3_type;
      typedef typename math_types::real_type    real_type;

      FILE *stream;
      // TODO: fopen is deprecated in VC++8
      if( (stream = fopen( filename.c_str(), "wb" )) == NULL )
      {
        std::cerr << "binary_write() : unable to open file " << filename << std::endl;
        return false;
      }

      size_t I = grid.I();
      size_t J = grid.J();
      size_t K = grid.K();
      size_t N = grid.size();
      fwrite( &I, sizeof( size_t ), 1, stream );
      fwrite( &J, sizeof( size_t ), 1, stream );
      fwrite( &K, sizeof( size_t ), 1, stream );
      fwrite( &N, sizeof( size_t ), 1, stream );

      vector3_type min_coord = grid.min_coord();
      real_type min_x = min_coord(0);
      real_type min_y = min_coord(1);
      real_type min_z = min_coord(0);
      fwrite( &min_x, sizeof( real_type ), 1, stream );
      fwrite( &min_y, sizeof( real_type ), 1, stream );
      fwrite( &min_z, sizeof( real_type ), 1, stream );

      vector3_type max_coord = grid.max_coord();
      real_type max_x = max_coord(0);
      real_type max_y = max_coord(1);
      real_type max_z = max_coord(0);
      fwrite( &max_x, sizeof( real_type ), 1, stream );
      fwrite( &max_y, sizeof( real_type ), 1, stream );
      fwrite( &max_z, sizeof( real_type ), 1, stream );

      real_type dx = grid.dx();
      real_type dy = grid.dy();
      real_type dz = grid.dz();
      fwrite( &dx, sizeof( real_type ), 1, stream );
      fwrite( &dy, sizeof( real_type ), 1, stream );
      fwrite( &dz, sizeof( real_type ), 1, stream );

      fwrite( grid.data(), sizeof( value_type ), grid.size(), stream );

      fclose( stream );
      std::cout << "binary_write(): Completed writing file: " << filename << std::endl;
      return true;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_BINARY_WRITE_H
#endif
