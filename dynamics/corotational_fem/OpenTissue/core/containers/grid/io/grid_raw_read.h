#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_READ_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_READ_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#ifdef WIN32
#  include <io.h> //--- KE 31-05-2003: Windows specific...
#endif

#include <OpenTissue/core/containers/grid/grid.h>

#include <iostream>
#include <iomanip>  // for setfill(), setw()
#include <sstream>
#include <list>
#include <cstdio>
#include <cmath>
#include <string>
#include <cassert>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Read Raw File.
    * Grid must be initialized to prober dimensions before reading
    *
    * @param filename   The path and filename of the raw file
    * @param grid        Upon return holds the grid.
    * @return           If grid was succesfully read then the return value is true otherwise it is false.
    */
    template <typename grid_type>
    inline bool  raw_read(std::string const & filename, grid_type & grid)
    {
      using std::min;
      using std::max;

      assert(grid.data()     || !"raw_read(): invalid data pointer");
      assert(grid.size()>0   || !"raw_read(): no space for reading data");
      typedef typename grid_type::value_type value_type;

      FILE *stream;
      if((stream = fopen( filename.c_str(), "rb" )) == NULL ){
        std::cerr << "raw_read(): Unable to open file" << filename << std::endl;
        return false;
      }
      fread( grid.data(), sizeof( value_type ), grid.size(), stream );
      fclose( stream );


      std::cout << "raw_read(): Completed reading grid file: "
        << filename
        << ", min = " 
        << static_cast<double>( OpenTissue::grid::min_element(grid) )
        << " max = "
        << static_cast<double>( OpenTissue::grid::max_element(grid) ) 
        << std::endl;
      return true;
    }

    /**
    * Read 8 bit data
    */
    template <typename grid_type>
    inline void raw_read_8bit(std::string const & filename, grid_type & target)
    {
      typedef OpenTissue::grid::Grid<unsigned char, typename grid_type::math_types > grid_8bit_type;
      raw_read( filename, target );
    }

    /**
    * Read 8 bit data and convert it to 16 bit
    */
    template <typename grid_type>
    inline void raw_read_8bit_to_16bit(std::string const & filename, grid_type & target)
    {
      typedef OpenTissue::grid::Grid<unsigned char, typename grid_type::math_types > grid_8bit_type;

      grid_8bit_type grid8bit;

      grid8bit.create( target.min_coord(), target.max_coord(), target.I(), target.J(), target.K() );
      raw_read( filename, grid8bit );

      unsigned char* cval = grid8bit.data();
      unsigned short* usval = target.data();
      for (size_t i=0; i<grid8bit.size(); i++, cval++, usval++)
        *usval = static_cast< unsigned short>(*cval)*16;
    }

    /**
    * Read 16 bit data (mostly just 12 bit)
    */
    template <typename grid_type>
    inline void raw_read_16bit(std::string const & filename, grid_type & target)
    {
      typedef OpenTissue::grid::Grid<unsigned short, typename grid_type::math_types> grid_16bit_type;
      raw_read( filename, target );
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_RAW_READ_H
#endif
