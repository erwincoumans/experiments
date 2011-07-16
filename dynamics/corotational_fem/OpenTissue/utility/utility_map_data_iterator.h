#ifndef OPENTISSUE_UTILITY_UTILITY_MAP_DATA_ITERATOR_H
#define OPENTISSUE_UTILITY_UTILITY_MAP_DATA_ITERATOR_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <cassert>

namespace OpenTissue
{
  namespace utility
  {

    /**
    * Map Data Iterator.
    * This iteartor is useful for iterating over all elements stored
    * in a map, without getting bothered by key-values.
    *
    * Example Usage:
    *
    *  std::map< int, char > A;
    *  ... add some data...
    *  map_data_iterator begin( A.begin() );
    *  map_data_iterator end( A.end() );
    *  map_data_iterator a = begin;
    *  for(;a!=end;++a)
    *    *a = 'a';
    *
    */
    //template<typename map_iterator>
    //class map_data_iterator
    //{
    //public:
    //  typedef typename map_iterator::value_type   pair_type;
    //  typedef typename pair_type::second_type    value_type;
    //protected:
    //  map_iterator m_iter;
    //public:
    //  map_data_iterator() : m_iter( map_iterator() ){}
    //  map_data_iterator( map_iterator iter): m_iter( iter ) {}
    //public:
    //  bool operator== ( const map_data_iterator & other ) const {  return ( m_iter == other.m_iter ); }
    //  bool operator!= ( const map_data_iterator & other ) const {  return !( *this == other);     }
    //public:
    //  value_type & operator*() { return (m_iter->second); }
    //  value_type * operator->() { return &(m_iter->second); }
    //  map_data_iterator & operator++() { ++m_iter; return *this; }  // prefix only
    //};

    template <class map_iterator> 
    class map_data_iterator  
      : public boost::iterator_adaptor< map_data_iterator<map_iterator>, map_iterator, typename map_iterator::value_type::second_type > 
    { 
    public:

      typedef typename map_data_iterator::iterator_adaptor_ super_t;     
      typedef typename map_iterator::value_type        pair_type; 
      typedef typename pair_type::second_type      second_type; 

      friend class boost::iterator_core_access; 

    public: 

      map_data_iterator() {} 

      explicit map_data_iterator(map_iterator x) 
        : super_t(x) {} 

      template<class other_iterator> 
      map_data_iterator( 
        map_data_iterator<other_iterator> const& r 
        , typename boost::enable_if_convertible<other_iterator, map_iterator>::type* = 0 
        ) 
        : super_t(r.base()) 
      {} 

    private: 

      second_type & dereference() const { return this->base()->second; } 

    }; 

  } //End of namespace utility
} //End of namespace OpenTissue

// OPENTISSUE_UTILITY_UTILITY_MAP_DATA_ITERATOR_H
#endif
