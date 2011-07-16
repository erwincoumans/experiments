#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GAUSS_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GAUSS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_constants.h>


#include <boost/cast.hpp> //--- Needed for boost::numeric_cast 

#include <valarray>
#include <vector>
#include <iostream>
#include <algorithm>

namespace OpenTissue
{
  namespace grid
  {

    namespace detail
    {
      /**
      * Compute Gaussian Kernel.
      * This method computes a normalized 1D Gaussian kernel.
      *
      * The Gaussian is centered in the middel of G, hence it is useful to
      * allocate a valarray of size 4*s+1.
      *
      * @param G     A pre-allocated valarray of scalars for the result.
      * @param s     The standard deviation of the Gaussian.
      */
      template <typename array_type, typename real_type>
      inline void compute_gaussian_kernel( array_type & G, real_type const & s )
      {
        typedef typename array_type::value_type    value_type; //--- entries of array may be of different type than s!!!!

        using std::exp;
        real_type x    = real_type(); //--- default construction on integral types means zero!!!
        value_type sum = real_type(); //--- default construction on integral types means zero!!!

        size_t size = boost::numeric_cast<size_t>(  G.size() );

        if ( ( size % 2 ) == 0 )
        {
          ++size;
          std::cout << "compute_gaussian_kernel(): Oups, size of gaussian kernel was even!!!" << std::endl;
        }

        real_type center = boost::numeric_cast<real_type>(   floor( size / 2.0 )  );

        //--- We calculate the unnormalized Gaussian in the interval -ms:ms,
        //--- where m = (G.size()-1)/2;  We assume G.size() is uneven.
        for ( size_t i = 0; i < size; ++i )
        {
          x = i - center;
          G[ i ] = boost::numeric_cast<value_type>(  exp( - ( x * x ) / ( 2.0 * s * s ) ) );
          sum += G[ i ];
        }

        //--- We normalize sum to 1.
        for ( size_t i = 0; i < size; ++i )
          G[i] = G[i] / sum;
      }

      /**
      * One-dimensional convolution with a Gaussian kernel.
      *
      * @param output      The resulting image.
      * @param input      The original image.
      * @param xdim   The size of I along the first dimenison.
      * @param ydim   The size of I along the second dimenison.
      * @param zdim   The size of I along the third dimenison.
      * @param dim    Which dimension to convolve along.
      * @param s      The standard deviation of the Gaussian to be convolved with.
      */
      template <typename value_type, typename real_type>
      inline void convolution1D(
        value_type * output
        , value_type const * input
        , int const & xdim
        , int const & ydim
        , int const & zdim
        , int const & dim
        , real_type const & s
        )
      {
        int size = xdim * ydim * zdim;

        //--- First take care of trivial case
        if ( s <= 0 )
        {
          std::copy( input, input+size, output );
          return;
        }

        //--- Now we allocate space for a gaussian on the interval -2s:2s.
        int N = boost::numeric_cast<int>( ceil( 4 * s ) + 1 );
        int N_half = boost::numeric_cast<int>( N/2.0);
        std::valarray<real_type> G( N );
        compute_gaussian_kernel( G, s );

        for ( int k = 0; k < zdim; ++k )
        {
          for ( int j = 0; j < ydim; ++j )
          {
            for ( int i = 0; i < xdim; ++i )
            {
              //--- Check if we should happen to be in the ``void'' of the image.
              //--- In this case we simply choose to ignore the gaussian bluring.
              int idx = ( i * ydim + j ) * zdim + k;
              if ( input[ idx ] == OpenTissue::math::detail::highest<value_type>() )
              {
                continue;
              }
              //--- Now we perform the convolution
              real_type sum = real_type();  //--- default constructed integral types are zero!!!
              for (int a = 0; a < N; ++a )
              {
                //--- We implement Neuman boundary condition: -1,-2,... -> 1,2,... (mirror)
                //--- We assume that G has uneven length!
                int newi = i;
                int newj = j;
                int newk = k;
                switch ( dim )
                {
                case 0:
                  newi -= a - N_half;
                  if ( newi < 0 )
                    newi = -newi;
                  if ( newi >= xdim )
                    newi = 2 * xdim - newi - 1; //--- zdim - (ind - zdim) = 2*zdim - ind;
                  break;
                case 1:
                  newj -= a - N_half;
                  if ( newj < 0 )
                    newj = -newj;
                  if ( newj >= ydim )
                    newj = 2 * ydim - newj - 1; //--- zdim - (ind - zdim) = 2*zdim - ind;
                  break;
                default:  //2 or higher
                  newk -= a - N_half;
                  if ( newk < 0 )
                    newk = -newk;
                  if ( newk >= zdim )
                    newk = 2 * zdim - newk - 1; //--- zdim - (ind - zdim) = 2*zdim - ind;
                  break;
                }

                //--- sum += J[newi*size[0]*size[1] +newj*size[0] +newk]*G[a];
                int idx2 = ( newi * ydim + newj ) * zdim + newk;

                //--- Void-values should not effect the convolution!!!
                if ( input[ idx2 ] != OpenTissue::math::detail::highest<value_type>() )
                  sum += boost::numeric_cast<real_type>(   input[ idx2 ] * G[ a ]  );
              }
              //--- Finally we update the destination image with the result of the convolution.
              output[ idx ] = boost::numeric_cast<value_type>(sum);
            }
          }
        }
      }

      /**
      * Three-dimensional convolution with a Gaussian kernel.
      *
      * @param output The resulting image.  Should be preallocated to same size as I.
      * @param input  The original image.
      * @param xdim   The size of I along the first dimenison.
      * @param ydim   The size of I along the second dimenison.
      * @param zdim   The size of I along the third dimenison.
      * @param sx     The standard deviation of the Gaussian along the first direction.
      * @param sy     The standard deviation of the Gaussian along the second direction.
      * @param sz     The standard deviation of the Gaussian along the third direction.
      */
      template <typename value_type, typename real_type>
      inline void  convolution3D(
        value_type * output
        , value_type const * input
        , int const & xdim
        , int const & ydim
        , int const & zdim
        , real_type const & sx
        , real_type const & sy
        , real_type const & sz
        )
      {
        int size = xdim * ydim * zdim;

        //--- Temporary workspace
        std::vector<value_type> tmp( size );

        //--- First the z-dimension:
        std::copy( input, input+size, tmp.begin() );
        convolution1D( output, &(tmp[0]), xdim, ydim, zdim, 2, sz );

        //--- Then the y-dimension:
        std::copy( output, output+size, tmp.begin() );
        convolution1D( output, &(tmp[0]), xdim, ydim, zdim, 1, sy );

        //--- Finally the x-dimension:
        std::copy( output, output+size, tmp.begin() );
        convolution1D( output, &(tmp[0]) , xdim, ydim, zdim, 0, sx );
      }

    }//namespace detail

    /**
    * Gaussian Convolution.
    *
    * @param src    Input image
    * @param dst    Upon return holds the resulting output image.
    * @param sx     Standard deviation in the x-axis direction.
    * @param sy     Standard deviation in the y-axis direction.
    * @param sz     Standard deviation in the z-axis direction.
    */
    template<typename grid_type,typename real_type>
    inline void gaussian_convolution(grid_type const & src, grid_type & dst, real_type sx, real_type sy, real_type sz )
    {
      std::cout << "gaussian_convolution(): dimensions: "
        << src.I()
        << "x"
        << src.J()
        << "x"
        << src.K()
        << std::endl;

      assert( dst.I() == src.I() || !"gaussian_convolution(): dst and src have differnet I dimension");
      assert( dst.J() == src.J() || !"gaussian_convolution(): dst and src have differnet I dimension");
      assert( dst.K() == src.K() || !"gaussian_convolution(): dst and src have differnet I dimension");
      int xdim = boost::numeric_cast< int>( src.I() );
      int ydim = boost::numeric_cast< int>( src.J() );
      int zdim = boost::numeric_cast< int>( src.K() );
      detail::convolution3D( dst.data(), src.data(), xdim, ydim, zdim, sx, sy, sz );
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GAUSS_H
#endif
