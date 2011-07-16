#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_ISOSURFACE_PROJECTION_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_ISOSURFACE_PROJECTION_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/grid/util/grid_value_at_point.h>
#include <OpenTissue/core/containers/grid/util/grid_gradient_at_point.h>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Project Point Cloud Onto Iso-surface.
    *
    * @param phi              The (signed distance grid) level set grid.
    * @param begin            An iterator to the first point that should
    *                         be projected.
    * @param end              An iterator to one position past the last point
    *                         that should be projected.
    * @param project_inside   Boolan flag indicating whether points lying inside
    *                         should be projected.
    * @param project_outside  Boolan flag indicating whether points lying outside
    *                         should be projected.
    * @param isovalue         The isovalue of the iso-surface onto which the points
    *                         should be projected. WARNING if value is larger or
    *                         smaller than what is actual in stored in phi, then
    *                         the projection algorithm may run into an infinite loop.
    */
    template<typename grid_type,typename vector3_iterator>
    inline void isosurface_projection(
      grid_type const & phi
      , vector3_iterator begin
      , vector3_iterator end
      , bool project_inside  = true
      , bool project_outside = true
      , double isovalue = 0.0
      )
    {
      typedef typename grid_type::math_types      math_types;
      typedef typename math_types::vector3_type   vector3_type;
      typedef typename math_types::value_type     real_type;

      using std::min;
      using std::max;
      using std::fabs;

      assert(isovalue > OpenTissue::grid::min_element(phi) || !"isosurface_projection(): isovalue exceeded minimum value");
      assert(isovalue < OpenTissue::grid::max_element(phi) || !"isosurface_projection(): isovalue exceeded maximum value");

      real_type threshold  = 0.0001;  //--- hmm, I just picked this one!!!

      for(vector3_iterator v = begin; v!=end; ++v)
      {
        vector3_type & p = (*v);

        real_type distance = value_at_point( phi, p ) - isovalue;

        if(!project_inside && distance < -threshold)
          continue;

        if(!project_outside && distance > threshold )
          continue;

        while ( fabs(distance) >  threshold)
        {
          vector3_type g = gradient_at_point( phi, p );
          while ( g*g < threshold )
          {
            vector3_type noise;
            random(noise, -threshold, threshold );
            p += noise;
            g = gradient_at_point(phi, p );
          }
          g = unit(g);
          p -= g * distance;
          distance = value_at_point( phi, p ) - isovalue;
        }
      }
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_ISOSURFACE_PROJECTION_H
#endif
