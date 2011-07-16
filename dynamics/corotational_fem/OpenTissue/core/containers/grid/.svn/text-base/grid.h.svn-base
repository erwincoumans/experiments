#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_GRID_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_GRID_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/grid/util/grid_iterators.h>
#include <OpenTissue/utility/utility_copy.h>
#include <OpenTissue/core/math/math_constants.h>

#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>

namespace OpenTissue
{
  namespace grid
  {
    template < typename T, typename math_types_  >
    class Grid
    {
    public:

      typedef OpenTissue::grid::Grid<T, math_types_>			grid_type;
      typedef T                                           value_type;
      typedef math_types_					   	    			          math_types;

      typedef typename math_types::index_vector3_type  index_vector; 

      typedef OpenTissue::grid::detail::Iterator< grid_type, value_type &, value_type *>                        iterator;
      typedef OpenTissue::grid::detail::Iterator< const grid_type, value_type const &, value_type const *>      const_iterator;

      typedef OpenTissue::grid::detail::IndexIterator< grid_type, value_type &, value_type *>                   index_iterator;
      typedef OpenTissue::grid::detail::IndexIterator< const grid_type, value_type const &, value_type const *> const_index_iterator;

    protected:

      typedef typename math_types::vector3_type	     vector3_type;
      typedef typename math_types::real_type         real_type;
      typedef typename math_types::value_traits      value_traits;

    protected:

      vector3_type m_min_coord;
      vector3_type m_max_coord;
      size_t   m_N;            ///< Total number of nodes in grid.
      size_t   m_I;            ///< Number of nodes along x-axis.
      size_t   m_J;            ///< Number of nodes along y-axis.
      size_t   m_K;            ///< Number of nodes along z-axis.
      vector3_type m_delta;        ///< Internode spacing along coordinate axes.
      value_type * m_value;
      value_type * m_value_end;
      value_type   m_infinity;     ///< Value specifying unused grid nodes.

    public:

      iterator       begin()       { return       iterator( this, m_value ); }
      const_iterator begin() const { return const_iterator( this, m_value ); }
      iterator       end()         { return       iterator( this, m_value_end ); }
      const_iterator end()   const { return const_iterator( this, m_value_end ); }

      Grid()
        : m_min_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_max_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_N(0)
        , m_I(0)
        , m_J(0)
        , m_K(0)
        , m_delta(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_value(0)
        , m_value_end(0)
        , m_infinity( math::detail::highest<T>() )
      {}

      /**
      * Specialized Constructor.
      * Should be used for more complex data types, which
      * do not have a default numeric_limits
      */
      Grid(value_type const & unused_val)
        : m_min_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_max_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_I(0)
        , m_J(0)
        , m_K(0)
        , m_delta(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_value(0)
        , m_value_end(0)
      {
        m_infinity = unused_val;
      }

      Grid(grid_type const & G)
        : m_min_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_max_coord(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_I(0)
        , m_J(0)
        , m_K(0)
        , m_delta(value_traits::zero(),value_traits::zero(),value_traits::zero())
        , m_value(0)
        , m_value_end(0)
        , m_infinity( G.unused() )
      {
        *this = G;
      }

      ~Grid() 
      { 
        if(m_value) 
          delete [] m_value; 
      }

      grid_type & operator=(grid_type const & G)
      {
        bool should_allocate = false;

        if(m_I != G.m_I)
          should_allocate = true;
        if(m_J != G.m_J)
          should_allocate = true;
        if(m_K != G.m_K)
          should_allocate = true;

        if(should_allocate)
          create(G.m_min_coord, G.m_max_coord, G.m_I, G.m_J, G.m_K);

        OpenTissue::utility::copy(G.begin(), G.end(), begin());

        return (*this);
      }

    public:

      /**
      * Create Grid.
      * This method allocates internal data structures.
      *
      * @param min_coord
      * @param max_coord
      * @param Ival
      * @param Jval
      * @param Kval
      */
      void create(
        vector3_type const & min_coord
        , vector3_type const & max_coord
        , size_t const & Ival
        , size_t const & Jval
        , size_t const & Kval
        )
      {
        size_t new_size = Ival*Jval*Kval;

        if(m_N != new_size && m_value)
        {
          delete [] m_value;
          m_value = 0;
        }
        m_I = Ival;
        m_J = Jval;
        m_K = Kval;
        m_N = new_size;
        m_min_coord = min_coord;
        m_max_coord = max_coord;
        m_delta(0) = (m_max_coord(0)-m_min_coord(0))/(m_I-1);
        m_delta(1) = (m_max_coord(1)-m_min_coord(1))/(m_J-1);
        m_delta(2) = (m_max_coord(2)-m_min_coord(2))/(m_K-1);
        if(!m_value)
        {
          std::cout << "-- OpenTissue::Grid::Create() : Trying to allocate " << m_I << "x" << m_J << "x" << m_K << " map of size " << m_N*sizeof(T) << " bytes... ";
          m_value = new T[m_N];
          m_value_end = m_value+m_N;
          std::cout << "Success!" << std::endl;
        }
        clear();
      }

      /**
      * Create a map with given dimensions and place it in center of world.
      *
      * @param I_val  Number of elements in x-direction.
      * @param J_val  Number of elements in y-direction.
      * @param K_val  Number of elements in z-direction.
      * @param dx_val World-coord spacing between elements in x-direction.
      * @param dy_val World-coord spacing between elements in y-direction.
      * @param dz_val World-coord spacing between elements in z-direction.
      */
      void create(
        size_t I_val
        , size_t J_val
        , size_t K_val
        , real_type dx_val
        , real_type dy_val
        , real_type dz_val
        )
      {
        using std::max;

        real_type w = I_val * dx_val / value_traits::two();
        real_type h = J_val * dy_val / value_traits::two();
        real_type d = K_val * dz_val / value_traits::two();
        vector3_type min_coord_value( -w, -h, -d );
        vector3_type max_coord_value( w, h, d );

        real_type factor = max( w, max( h, d ) );

        min_coord_value /= factor;
        max_coord_value /= factor;

        create( min_coord_value, max_coord_value, I_val, J_val, K_val );
      }

      /**
      * Clears the map to infinity/unused values.
      */
      void clear()
      {
        assert(m_value || !"Grid::clear(): data value pointer was NULL");
        std::fill(begin(), end(), m_infinity);
      }

      void set_spacing(real_type const & new_dx, real_type const & new_dy, real_type const & new_dz)
      {
        real_type span_x = (m_I-1) * new_dx;
        m_max_coord(0)   = span_x / value_traits::two();
        m_min_coord(0)   = -m_max_coord(0);
        m_delta(0)       = new_dx;

        real_type span_y = (m_J-1)*new_dy;
        m_max_coord(1)   = span_y/value_traits::two();
        m_min_coord(1)   = -m_max_coord(1);
        m_delta(1)       = new_dy;

        real_type span_z = (m_K-1)*new_dz;
        m_max_coord(2)   = span_z/value_traits::two();
        m_min_coord(2)   = -m_max_coord(2);
        m_delta(2)       = new_dz;
      }

    public:

      value_type & get_value(size_t const & i, size_t const & j, size_t const & k) const
      {
        int ii = (i%m_I);
        if(ii<0)
          ii += m_I;
        int jj = (j%m_J);
        if(jj<0)
          jj += m_J;
        int kk = (k%m_K);
        if(kk<0)
          kk += m_K;
        long idx = (kk*m_J + jj)*m_I + ii;
        return m_value[idx];
      }

      value_type & get_value(size_t const & linear_index) const
      {
        assert(linear_index>=0           || !"Grid::get_value(): index was out of range");
        assert(linear_index<this->size() || !"Grid::get_value(): index was out of range");
        return m_value[linear_index];
      }

      value_type       & operator() (size_t const & i,size_t const & j,size_t const & k)       {  return this->get_value(i,j,k); }      
      value_type const & operator() (size_t const & i,size_t const & j,size_t const & k) const {  return this->get_value(i,j,k); }

      value_type       & operator() (size_t const & linear_index)       { return this->get_value( linear_index ); }
      value_type const & operator() (size_t const & linear_index) const { return this->get_value( linear_index ); }

      value_type       & operator() (index_vector const& iv)       { return this->get_value( iv(0), iv(1), iv(2) ); }

      value_type const & operator() (index_vector const& iv) const { return this->get_value( iv(0), iv(1), iv(2) ); }


      real_type width()  const { return m_max_coord(0) - m_min_coord(0); }
      real_type height() const { return m_max_coord(1) - m_min_coord(1); }
      real_type depth()  const { return m_max_coord(2) - m_min_coord(2); }

      real_type & dx() { return m_delta(0); }
      real_type & dy() { return m_delta(1); }
      real_type & dz() { return m_delta(2); }

      real_type const & dx() const { return m_delta(0); }
      real_type const & dy() const { return m_delta(1); }
      real_type const & dz() const { return m_delta(2); }

      size_t I() const { return m_I; }
      size_t J() const { return m_J; }
      size_t K() const { return m_K; }
      size_t size() const { return m_N; }

      vector3_type const & min_coord()                       const { return m_min_coord; }
      vector3_type const & max_coord()                       const { return m_max_coord; }
      real_type    const & min_coord(size_t const & idx) const { return m_min_coord(idx); }
      real_type    const & max_coord(size_t const & idx) const { return m_max_coord(idx); }

      value_type unused() const { return m_infinity; }

      value_type infinity() const { return m_infinity; }

      bool valid() const { return (m_value!=0); }

      bool empty() const { return ( size()==0 ); }

      value_type       * data()       { return m_value; }

      value_type const * data() const { return m_value; }

    };

    /**
    * Get Minimum Value.
    *
    * @param map   The map from which the minimum value is wanted.
    *
    * @return      The minimum value stored in grid.
    */
    template< typename T, typename M >
    T min_element(OpenTissue::grid::Grid<T,M> const & G)
    {
      T const * value = G.data();
      T min_value = OpenTissue::math::detail::highest<T>();
      for(size_t idx=0; idx<G.size(); ++idx, ++value)
      {
        if( *value!=G.unused() && *value<min_value )
          min_value = *value;
      }
      return min_value;
    }

    /**
    * Get Maximum Value.
    *
    * @param map   The map from which the maximum value is wanted.
    *
    * @return      The maximum value stored in grid.
    */
    template< typename T, typename M >
    T max_element(OpenTissue::grid::Grid<T,M>  const & G)
    {
      T const * value = G.data();
      T max_value = OpenTissue::math::detail::lowest<T>();

      for(size_t idx=0;idx<G.size();++idx,++value)
      {
        if(*value!=G.unused() && *value>max_value)
          max_value = *value;
      }
      return max_value;
    }

    /**
    * Get absolute value.
    * Note this function changes the values stored in the arugment.
    *
    * @param map   The map from which the maximum value is wanted.
    *
    * @return      A map containing the absolute value.
    */
    template<typename T, typename M>
    Grid<T,M> & fabs(Grid<T,M> & G)
    {
      using std::fabs;

      T * value = G.data();
      for(size_t idx=0;idx<G.size();++idx,++value)
      {
        if(*value!=G.unused())
          *value = fabs(*value);
      }
      return G;
    }

    /**
    * Negate using '-' operator of class T.
    * This function is thus only valid when T implements unary negation
    * Note this function changes the values stored in the arugment.
    *
    *
    * @param map   The map to be negated.
    *
    * @return      A map containing the negated values.
    */
    template<typename T,typename M>
    Grid<T,M> & negate(Grid<T,M> & G)
    {
      T * value = G.data();
      for(size_t idx=0;idx<G.size();++idx,++value)
      {
        if(*value!=G.unused())
          *value = -(*value);
      }
      return G;
    }

    template<typename T,typename M>
    Grid<T,M> &  scale(Grid<T,M> & G, T fac)
    {
      T * value = G.data();
      for(size_t idx=0;idx<G.size();++idx,++value)
      {
        if(*value!=G.unused())
          *value = fac*(*value);
      }
      return G;
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_GRID_H
#endif
