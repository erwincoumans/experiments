#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_TRANSFORM_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_TRANSFORM_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

namespace OpenTissue
{
  namespace grid
  {
    /**
    * Transform Grid.
    *
    * This is great for generating maps from other maps, say we want a map, G, with
    * the gradient magnitude of another map, U. Then simply write:
    *
    *   transform( 
    *       U.begin()
    *     , U.end()
    *     , G.begin()
    *     , boost::bind( gradient_magnitude<intensity_iterator> , _1 )
    *     );
    *
    * This is very similar to the std::transform. The major difference is that std::transform
    * dereference an iterator before invoking the functor. The transform version simply
    * invokes the functor by passing the iterator as argument.
    *
    * This is important because map iterators contain spatial information about the
    * location of the iterator. This location information is for instance needed for
    * finite difference computations etc..
    *
    * @param in_begin
    * @param in_end
    * @param out_begin
    * @param func
    */
    template<typename input_iterator, typename output_iterator ,typename function_type>
    inline void transform(input_iterator in_begin,input_iterator in_end, output_iterator out_begin, function_type func)
    {
      for ( input_iterator i = in_begin; i != in_end; ++i,++out_begin)
        *out_begin  = func( i );
    }



  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_FUNCTIONS_H
#endif
