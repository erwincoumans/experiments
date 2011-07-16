#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_WRITE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_WRITE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <iostream>
#include <cstdio>
#include <string>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Write Raw file
    *
    * @param filename The path and filename of the raw file.
    * @param grid      The grid that should be written.
    *
    * @return         If file was succesfully written then the return value is true otherwise it is false.
    */
    template <typename grid_type>
    inline bool  raw_write(std::string const & filename, grid_type & grid)
    {
      typedef typename grid_type::value_type value_type;
      FILE *stream;
      if( (stream = fopen( filename.c_str(), "wb" )) == NULL ){
        std::cerr << "raw_write():  Unable to open file" << filename << std::endl;
        return false;
      }
      fwrite( grid.data(), sizeof( value_type ), grid.size(), stream );
      fclose( stream );
      std::cout << "raw_write(): Completed writing grid file " << filename << std::endl;
      return true;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_WRITE_H
#endif
