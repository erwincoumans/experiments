#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_IMAGE_READ_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_IMAGE_READ_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/gpu/image/image.h>
#include <OpenTissue/gpu/image/io/image_read.h>

#include <iostream>
#include <list>
#include <cstdio>
#include <cmath>
#include <string>

#ifdef WIN32
#  include <io.h> //--- KE 31-05-2003: Windows specific...
#endif

namespace OpenTissue
{
  namespace grid
  {
    /**
    * Read images into Grid.
    *
    * Reads a stack of images into a grid data structure. The order of the images are
    * determined by the alphabetic sorting of the filenames and images are stored
    * in increasing k-planes.
    *
    * It is assumed that all images being read have the same (I,J) dimension, the
    * number of images determined the K-dimension of the resulting grid.
    *
    * @param directory   The location of a folder containing image files
    * @param filter      A file filter, for instance "*.bmp" or "*.png" etc.
    * @param grid         Upon return holds the grid.
    *
    * @return            If images were succesfully read into grid then the return value is true otherwise it is false.
    */
    template <typename grid_type>
    inline bool image_read(std::string const & directory, std::string const & filter, grid_type & grid)
    {
      typedef typename grid_type::value_type    value_type;
      typedef typename grid_type::math_types    math_types;
      typedef typename math_types::vector3_type vector3_type;
      typedef typename math_types::real_type    real_type;

#ifdef WIN32
      std::string filter = directory + '/' + filetype;

      std::list<std::string> filenames;

      struct _finddata_t file;
      long hFile;

      //--- Find first file specified by the given filter
      if( (hFile = _findfirst( filter.c_str() , &file )) == -1L )
      {
        std::cerr << "-- Invalid filter " << filter << " No such files in that directory!" << std::endl;
        _findclose( hFile );
        return false;
      }
      else
      {
        size_t rdonly = file.attrib & _A_RDONLY;
        size_t system = file.attrib & _A_SYSTEM;
        size_t hidden = file.attrib & _A_HIDDEN;
        size_t archiv = file.attrib & _A_ARCH;
        std::string filename = file.name;
        filenames.push_back(filename);

        //--- Find the rest of the files
        while( _findnext( hFile, &file ) == 0 )
        {
          rdonly = file.attrib & _A_RDONLY;
          system = file.attrib & _A_SYSTEM;
          hidden = file.attrib & _A_HIDDEN;
          archiv = file.attrib & _A_ARCH;
          filename = file.name;
          filenames.push_back(filename);
        }
        _findclose( hFile );
      }

      filenames.sort();
      int K = (int)filenames.size();
      std::string image_filename = directory + '/' + (*filenames.begin());

      image_type image;
      bool success = OpenTissue::image::read(image_filename,image);
      if(!success)
      {
        std::cout << "-- Could not read " << dibname1 << std::endl;
      }
      int I = image.width();
      int J = image.height();
      // Center view
      vector3_type min_coord(-I/2.0,-J/2.,-K/2.);
      vector3_type max_coord(I/2.,J/2.,K/2.);
      grid.create(min_coord,max_coord,I,J,K);

      int k=K-1;
      std::list<string>::iterator it = filenames.begin();
      for(;it!=filenames.end();++it,--k)
      {
        std::string filename = directory + '/' + (*it);
        std::cout << "image_read(): Reading " << filename << std::endl;

        success = OpenTissue::image::read(filename,image);
        if(!success)
        {
          std::cout << "image_read(): could not read " << filename << std::endl;
        }

        for(size_t j=0; j<grid.J(); ++j)
        {
          for(size_t i=0; i<grid.I(); ++i)
          {
            unsigned char red   = image.get(i, j,0);
            unsigned char green = image.get(i, j,1);
            unsigned char blue  = image.get(i, j,2);
            unsigned char alpha = image.get(i, j,3);
            value_type grid_value = static_cast<value_type>( red );
            long idx = (k*grid.I()*grid.J()) + (j*grid.I()) + i;
            grid(idx) = grid_value;
          }
        }

      }
      std::cout << "image_read(): Read map of size (" << I << "x" <<  J << "x" << K << ")" << std::endl;
#else
      std::cout<< "image_read(): Sorry windows only!" << std::endl;
#endif // WIN32
      return true;
    }

  } // namespace grid
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_GRID_IO_GRID_IMAGE_READ_H
#endif
