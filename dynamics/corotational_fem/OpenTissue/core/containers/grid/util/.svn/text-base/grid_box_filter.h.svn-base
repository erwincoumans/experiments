#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_BOX_FILTER_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_BOX_FILTER_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <boost/lambda/lambda.hpp>
#include <cmath> // for pow()

namespace OpenTissue
{
  namespace grid
  {
    /**
    * Fast convolution by an integer sized box signal.
    * Variance of the filter is 1/12*(size*size-1).
    *
    * @param src   Source grid to be convolved.
    * @param size  Size of box filter.
    * @param dst   Upon return, contains the filtered grid.
    */
    template <typename grid_type>
    inline void box_filter(grid_type const& src, size_t size, grid_type & dst)
    {
      typedef typename grid_type::value_type value_type;

      grid_type tmp=src;

      size_t center=size/2;
      for(size_t x=0; x<src.I(); ++x)
        for(size_t y=0; y<src.J(); ++y)
        {
          value_type sum= value_type(0);

          for(size_t z=0; z<src.K()+center; ++z)
          {
            if(z<src.K())
              sum+=tmp(x,y,z);
            if(z>=size)
              sum-=tmp(x,y,z-size);
            if(z>=center)
              dst(x,y,z-center)=sum;
          }
        }
        tmp = dst;
        for(size_t x=0; x<src.I(); ++x)
          for(size_t z=0; z<src.K(); ++z)
          {
            value_type sum= value_type(0);
            for(size_t y=0; y<src.J()+center; ++y)
            {
              if(y<src.J())
                sum+=tmp(x,y,z);
              if(y>=size)
                sum-=tmp(x,y-size,z);
              if(y>=center)
                dst(x,y-center,z)=sum;
            }
          }
          tmp=dst;
          for(size_t y=0; y<src.J(); ++y)
            for(size_t z=0; z<src.K(); ++z)
            {
              value_type sum= value_type(0);
              for(size_t x=0; x<src.I()+center; ++x)
              {
                if(x<src.I())
                  sum+=tmp(x,y,z);
                if(x>=size)
                  sum-=tmp(x-size,y,z);
                if(x>=center)
                  dst(x-center,y,z)=sum;
              }
            }

            std::for_each( 
              dst.begin()
              , dst.end()
              , boost::lambda::_1 *=  boost::lambda::make_const(  1.0/pow(static_cast<double>(size),3) ) 
              );
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_BOX_FILTER_H
#endif
