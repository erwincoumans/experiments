#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CURVATURE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CURVATURE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <cmath>
#include <OpenTissue/core/math/math_eigen_system_decomposition.h>
#include <OpenTissue/core/containers/grid/util/grid_gradient.h> 
#include <OpenTissue/core/containers/grid/util/grid_hessian.h>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Get Curvature.
    *
    * @param  i           Node coordinate
    * @param  j           Node coordinate
    * @param  k           Node coordinate
    * @param H            Upon return this value contains the mean curvature.
    * @param K            Upon return this value contains the gauss curvature.
    * @param k1            Upon return this value contains the first principal curvature.
    * @param k2            Upon return this value contains the second principal curvature.
    */
    template<typename grid_type, typename real_type>
    inline void curvature(
      grid_type const & grid
      , size_t i
      , size_t j
      , size_t k
      , real_type & K
      , real_type & G
      , real_type & k1
      , real_type & k2
      )
    {
      using std::min;
      using std::max;
      using std::pow;
      using std::sqrt;

      typedef OpenTissue::math::Vector3<real_type>          vector3_type;
      typedef OpenTissue::math::Matrix3x3<real_type>        matrix3x3_type;

      real_type limit_K = boost::numeric_cast<real_type>(   1. / min( grid.dx(), min( grid.dy(), grid.dz() ) )    );

      vector3_type g;
      matrix3x3_type H;
      gradient( grid, i, j, k, g );
      hessian( grid, i, j, k, H );
      real_type h = g * g;

      //--- Test whether the gradient was zero, if so we simply imagine it has norm one, a better
      //--- solution would proberly be to pick a random node and compute the curvature information
      //--- herein (this is suggest by Oscher and Fedkiw).
      if ( h == 0 )
        h = 1;
      //--- Compute Mean curvature, defined as: kappa = \nabla \cdot (\nabla \phi / \norm{\nabla \phi}  )
      const static real_type exponent = boost::numeric_cast<real_type>( 3. / 2. );
      K = ( 1.0 / pow( h, exponent ) ) * (
        g( 0 ) * g( 0 ) * ( H( 1, 1 ) + H( 2, 2 ) ) - 2. * g( 1 ) * g( 2 ) * H( 1, 2 ) +
        g( 1 ) * g( 1 ) * ( H( 0, 0 ) + H( 2, 2 ) ) - 2. * g( 0 ) * g( 2 ) * H( 0, 2 ) +
        g( 2 ) * g( 2 ) * ( H( 0, 0 ) + H( 1, 1 ) ) - 2. * g( 0 ) * g( 1 ) * H( 0, 1 )
        );
      //--- Clamp Curvature, it does not make sense if we compute
      //--- a curvature value that can not be representated with the
      //--- current grid resolution.
      K = min( K, limit_K  );
      K = max( K, -limit_K );

      //--- Compute Gaussian Curvature
      G = ( 1.0 / ( h * h ) ) * (
        g( 0 ) * g( 0 ) * ( H( 1, 1 ) * H( 2, 2 ) - H( 1, 2 ) * H( 1, 2 ) ) + 2. * g( 1 ) * g( 2 ) * ( H( 0, 2 ) * H( 0, 1 ) - H( 0, 0 ) * H( 1, 2 ) ) +
        g( 1 ) * g( 1 ) * ( H( 0, 0 ) * H( 2, 2 ) - H( 0, 2 ) * H( 0, 2 ) ) + 2. * g( 0 ) * g( 2 ) * ( H( 1, 2 ) * H( 0, 1 ) - H( 1, 1 ) * H( 0, 2 ) ) +
        g( 2 ) * g( 2 ) * ( H( 0, 0 ) * H( 1, 1 ) - H( 0, 1 ) * H( 0, 1 ) ) + 2. * g( 0 ) * g( 1 ) * ( H( 1, 2 ) * H( 0, 2 ) - H( 2, 2 ) * H( 0, 1 ) ) );
      //---- Clamp Curvature
      G = min( G, limit_K );
      G = max( G, -limit_K );
      //--- According to theory we can compute principal curvatures form Gaussian and Mesn curvature
      ral_type d = sqrt( K * K - G );
      k1 = K + d;
      k2 = K - d;
    }

    /**
    * Compute principal curvatures based on eigen value decomposition of the shape matrix.
    *
    *  Based on:
    *
    *     Bridson, R., Teran, J., Molino, N. and Fedkiw, R.,
    *     "Adaptive Physics Based Tetrahedral Mesh Generation Using Level Sets",
    *     Engineering with Computers, (in press).
    *
    * Earlier vesion appears to be (unpublished)
    *
    *    Tetrahedral Mesh Generation for Deformable Bodies
    *    Neil Molino (Stanford University)
    *    Robert Bridson (Stanford University)
    *    Ronald Fedkiw (Stanford University)
    *    Submitted to SCA 2003
    *
    * @param k1            Upon return this value contains the first principal curvature.
    * @param k2            Upon return this value contains the second principal curvature.
    */
    template<typename grid_type, typename real_type>
    inline void eigen_curvature(
      grid_type const & grid
      , size_t i
      , size_t j
      , size_t k
      , real_type & k1
      , real_type & k2
      )
    {
      using std::min;
      using std::max;
      using std::pow;
      using std::sqrt;

      typedef OpenTissue::math::Vector3<real_type>          vector3_type;
      typedef OpenTissue::math::Matrix3x3<real_type>        matrix3x3_type;

      //real_type limit_K = boost::numeric_cast<real_type>(   1. / min( grid.dx(), min( grid.dy(), grid.dz() ) )    );

      vector3_type g;
      matrix3x3_type H;
      gradient( grid, i, j, k, g );
      hessian( grid, i, j, k, H );

      real_type recip_norm_g = 1. / sqrt( g * g );
      //--- compute outward normal vector
      vector3_type n = g * recip_norm_g;
      //--- compute projection matrix
      matrix3x3_type I( 1, 0, 0, 0, 1, 0, 0, 0, 1 );
      matrix3x3_type nnt = outer_prod( n, n );
      matrix3x3_type P = I - nnt;
      //--- The shape matrix
      matrix3x3_type S = -P * H * P * recip_norm_g;
      //--- compute eigen values of the shape matrix
      matrix3x3_type V;
      vector3_type d;
      OpenTissue::math::eigen( S, V, d );
      size_t order[ 3 ];
      vector3_type abs_d = OpenTissue::fabs( d );
      get_increasing_order( abs_d, order );
      k1 = d( order[ 0 ] );
      k2 = d( order[ 1 ] );
      //dir1 = vector3_type(V(0,order[0]),V(1,order[0]),V(2,order[0]));
      //dir2 = vector3_type(V(0,order[1]),V(1,order[1]),V(2,order[1]));
    }



    /**
    * Computes principal curvatures based on linear algebra.
    *
    * This is similar to the approach used in the paper:
    *
    *   @inproceedings{museth.breen.ea02,
    *      author = {Ken Museth and David E. Breen and Ross T. Whitaker and Alan H. Barr},
    *      title = {Level set surface editing operators},
    *      booktitle = {Proceedings of the 29th annual conference on Computer graphics and interactive techniques},
    *      year = {2002},
    *      isbn = {1-58113-521-1},
    *      pages = {330--338},
    *      location = {San Antonio, Texas},
    *      doi = {http://doi.acm.org/10.1145/566570.566585},
    *      publisher = {ACM Press},
    *   }
    *
    *
    * @param k1            Upon return this value contains the first principal curvature.
    * @param k2            Upon return this value contains the second principal curvature.
    */
    template<typename grid_type, typename real_type>
    inline void algebra_curvature(
      grid_type const & grid
      , size_t i
      , size_t j
      , size_t k
      , real_type & k1
      , real_type & k2
      )
    {
      using std::min;
      using std::max;
      using std::pow;
      using std::sqrt;

      typedef OpenTissue::math::Vector3<real_type>          vector3_type;
      typedef OpenTissue::math::Matrix3x3<real_type>        matrix3x3_type;
      //real_type limit_K = boost::numeric_cast<real_type>(   1. / min( grid.dx(), min( grid.dy(), grid.dz() ) )    );

      vector3_type g;
      matrix3x3_type H;
      gradient( grid, i, j, k, g );
      hessian( grid, i, j, k, H );
      real_type recip_norm_g = 1. / sqrt( g * g );
      //--- compute outward normal vector
      vector3_type n = g * recip_norm_g;
      //--- compute projection matrix
      matrix3x3_type I( 1, 0, 0, 0, 1, 0, 0, 0, 1 );
      matrix3x3_type nnt = outer_prod( n, n );
      matrix3x3_type P = I - nnt;
      //--- The shape matrix
      matrix3x3_type S = -P * H * P * recip_norm_g;
      //--- Compute trace of S matrix
      real_type M = trace(S);
      matrix3x3_type ST = trans(S);
      matrix3x3_type SST = S * ST;
      real_type K = trace(SST);
      k1 = 0.5 * ( M + sqrt( 2 * K * K - M * M ) );
      k2 = 0.5 * ( M - sqrt( 2 * K * K - M * M ) );
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_CURVATURE_H
#endif
