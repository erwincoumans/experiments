#ifndef OPENTISSUE_UTILITY_UTILITY_COPY_H
#define OPENTISSUE_UTILITY_UTILITY_COPY_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>


#include <cstring> // for memcpy()

namespace OpenTissue { namespace utility
{
  // Special implementation of copy, that uses memcpy for compatible types
  // (based on example from boost::type_traits)
  namespace detail
  {
    template <bool b>
    struct copier
    {
      template <typename I1, typename I2>
      static I2 do_copy( I1 first, I1 last, I2 out )
      {
        std::cout << "-- OpenTissue::utility::copy (normal)" << std::endl;
        while ( first != last )
        {
          *out = *first;
          ++out;
          ++first;
        }
        return out;
      }
    };

    template <>
    struct copier<true>
    {
      template <typename I1, typename I2>
      static I2 do_copy( I1 first, I1 last, I2 out )
      {
        //std::cout << "-- OpenTissue::utility::copy (fast)" << std::endl;
        memcpy( &( *out ), &( *first ), ( last - first ) * sizeof( typename I2::value_type ) );
        return out + ( last - first );
      }
    };
  } // namespace detail

  template <typename I1, typename I2>
  I2 copy( I1 first, I1 last, I2 out )
  {
    typedef typename boost::remove_cv<typename std::iterator_traits<I1>::value_type>::type v1_t;
    typedef typename boost::remove_cv<typename std::iterator_traits<I2>::value_type>::type v2_t;

    return detail::copier <
           ::boost::type_traits::ice_and <
           ::boost::is_same<v1_t, v2_t>::value,
           //             ::boost::is_pointer<I1>::value,
           //             ::boost::is_pointer<I2>::value,
           ::boost::has_trivial_assign<v1_t>::value
           > ::value > ::do_copy( first, last, out );
  }
}} // namespace OpenTissue::utility

// OPENTISSUE_UTILITY_UTILITY_COPY_H
#endif
