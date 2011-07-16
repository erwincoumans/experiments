#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_MATLAB_WRITE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_MATLAB_WRITE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <iostream>
#include <fstream>
#include <string>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Writes a Grid as a Matlab Array.
    *
    * @param mfile   The path and filename of the generated matlab file.
    * @param grid     The grid that should be written as a matlab array.
    * 
    * @return        If succesfull written the return value is true otherwise it is false.
    */
    template <typename grid_type>
    inline bool matlab_write(std::string const & mfile, grid_type const & grid)
    {
      typedef typename grid_type::value_type    value_type;
      typedef typename grid_type::math_types    math_types;
      typedef typename math_types::vector3_type vector3_type;
      typedef typename math_types::real_type    real_type;

      std::ofstream file(mfile.c_str());
      file.precision(30);

      value_type * value = grid.data();
      for (size_t k=0; k<grid.K(); ++k)
      {
        file << "mfile(:,:," << k+1 << ") = [ " ;
        for (size_t j=0; j<grid.J(); ++j)
        {
          file << "[ " ;

          for (size_t i=0; i<grid.I()-1; ++i, value++)
          {
            file << static_cast<double>(*value) << ", " ;
          }
          file << static_cast<double>(*value++) << "];" << std::endl;
        }
        file << "];" << std::endl;
      }
      file.flush();
      file.close();
      return true;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_MATLAB_WRITE_H
#endif
