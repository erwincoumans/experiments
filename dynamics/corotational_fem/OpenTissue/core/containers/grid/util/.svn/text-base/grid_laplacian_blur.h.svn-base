#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_LAPLACIAN_BLUR_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_LAPLACIAN_BLUR_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/grid/util/grid_poisson_solver.h>

#include <boost/lambda/lambda.hpp>
#include <algorithm>

namespace OpenTissue
{
  namespace grid
  {
    /**
    * Solves the PDE
    *
    *   \nu \nabla^2 \phi = phi(t=0)
    *
    * @param image            The image to blur.
    * @param diffusion        The value of the diffusion coefficient.
    * @param max_iterations   The maximum number of iterations allowed. Default is 30 iterations.
    */
    template<typename grid_type>
    inline void laplacian_blur(
      grid_type & image
      , double diffusion=1.0
      , size_t max_iterations = 10
      )
    {
      using std::fabs;

      grid_type rhs = image;
      if(fabs(diffusion-1.0)>0.0)
      {
        double inv_diffusion = 1.0/diffusion;

        typename grid_type::iterator r_end = rhs.end();
        typename grid_type::iterator r     = rhs.begin();
        for(;r!=r_end;++r)
          (*r) *= inv_diffusion;
        // TODO: The below one-liner doesn't work on windows?
        // NOTE: Converted above to the below one-liner.
        //{
        //  using namespace boost::lambda; /// NOTE: Are we allowed to use namespaces like this without polluting?
        //  std::for_each(rhs.begin(), rhs.end(), _1 *= inv_diffusion );
        //}

      }
      else
        std::fill( rhs.begin(), rhs.end(), typename grid_type::value_type(0.0) );

      poisson_solver(image,rhs,max_iterations);
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_LAPLACIAN_BLUR_H
#endif
