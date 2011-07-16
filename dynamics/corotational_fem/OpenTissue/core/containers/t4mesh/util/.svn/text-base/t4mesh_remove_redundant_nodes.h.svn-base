#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_REMOVE_REDUDANT_NODES_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_REMOVE_REDUDANT_NODES_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/collision/spatial_hashing/spatial_hashing.h>

namespace OpenTissue
{
  namespace t4mesh
  {

    namespace detail
    {

      /**
      * T4Mesh Point Queury Collision Policy.
      * This policy can be used to setup collision queries to find collisions between tetrahedral nodes.
      *
      * 
      */
      template<typename volume_mesh, typename point_container>
      class t4mesh_node_collision_policy
      {
      public:

        typedef typename volume_mesh::node_type         node_type;
        typedef typename node_type::vector3_type        vector3_type;
        typedef typename vector3_type::value_type       real_type;
        typedef int                                     index_type;
        typedef node_type                               data_type;
        typedef node_type                               query_type;
        typedef OpenTissue::spatial_hashing::GridHashFunction            hash_function; 

        typedef OpenTissue::spatial_hashing::Grid< vector3_type, OpenTissue::math::Vector3<int>, data_type, hash_function>  hash_grid;

        typedef std::list< data_type* >  result_container;

        real_type m_precision;       //--- precision when testing node coordinates for collision. Default value is 0.01
        point_container * m_points;  //--- A pointer to a container containing the coordinates of the tetrahedra mesh nodes.

      public:

        t4mesh_node_collision_policy()
          : m_precision(0.01)
        {}

        //vector3_type position ( data_type  const & data  ) const { return data.m_coord;                                   }
        //vector3_type min_coord( query_type const & query ) const { return vector3_type(query.m_coord - vector3_type(m_precision,m_precision,m_precision)); }
        //vector3_type max_coord( query_type const & query ) const { return vector3_type(query.m_coord + vector3_type(m_precision,m_precision,m_precision)); }
        //void         reset    ( result_container & results )     {    results.clear();                                    }
        //
        //void report(data_type const & data, query_type const & query, result_container & results)
        //{
        //  static real_type threshold = m_precision*m_precision + math::working_precision<real_type>( );
        //
        //  if(data.idx()==query.idx()) //--- No self-pairs!
        //    return;
        //
        //  vector3_type d = query.m_coord - data.m_coord;
        //  real_type tst = d*d;
        //  if( tst < threshold )
        //    results.push_back( const_cast<data_type*>(&data) );
        //}

        vector3_type position ( data_type  const & data  ) const 
        { 
          assert( m_points || !"t4mesh_node_collision_policy::position(): points container was null ");
          return (*m_points)[data.idx()];                                   
        }

        vector3_type min_coord( query_type const & query ) const 
        { 
          assert( m_points || !"t4mesh_node_collision_policy::min_coord(): points container was null ");
          return vector3_type((*m_points)[query.idx()] - vector3_type(m_precision,m_precision,m_precision)); 
        }

        vector3_type max_coord( query_type const & query ) const 
        {
          assert( m_points || !"t4mesh_node_collision_policy::max_coord(): points container was null ");
          return vector3_type((*m_points)[query.idx()] + vector3_type(m_precision,m_precision,m_precision)); 
        }

        void         reset    ( result_container & results )     {    results.clear();                                    }

        void report(data_type const & data, query_type const & query, result_container & results)
        {
          assert( m_points || !"t4mesh_node_collision_policy::report(): points container was null ");

          real_type threshold = m_precision*m_precision + math::working_precision<real_type>( );

          if(data.idx()==query.idx()) //--- No self-pairs!
            return;

          vector3_type d = (*m_points)[query.idx()] - (*m_points)[data.idx()];
          real_type tst = d*d;
          if( tst < threshold )
            results.push_back( const_cast<data_type*>(&data) );
        }

      };

    } // namespace detail

    /**
    * Remove Redundant Nodes.
    *
    * This function assumes that default traits are available, i.e. a node has an m_coord member of vector3_type.
    *
    * @param A          The volume mesh from which redundant nodes are to be removed.
    * @param B          The result volume mesh without any redundant nodes.
    * @param precision  Precision is maximum distance between two nodes that should be merged into one.
    */
    template<typename volume_mesh, typename point_container>
    void remove_redundant_nodes(
      volume_mesh & A
      , point_container const & pointsA
      , volume_mesh & B
      , point_container & pointsB
      , double precision = 10e-7
      )
    {

      typedef typename detail::t4mesh_node_collision_policy<volume_mesh,point_container>     policy;
      typedef typename volume_mesh::node_type                                 node_type;
      typedef typename volume_mesh::node_iterator                             node_iterator;
      typedef typename volume_mesh::tetrahedron_iterator                      tetrahedron_iterator;
      typedef typename policy::result_container                               result_container;
      typedef typename result_container::iterator                             result_iterator;
      typedef OpenTissue::spatial_hashing::PointDataQuery<typename policy::hash_grid,policy>   point_query_type;

      std::cout << "remove_redundant_nodes(): Initially  "
        << A.size_nodes()
        << " nodes and " 
        << A.size_tetrahedra()
        << " tetrahedra" 
        << std::endl;

      B.clear();

      //--- Create nodes in output mesh.
      unsigned int N = A.size_nodes();
      std::vector<unsigned int> node_lut(N,~0u);

      node_iterator a_begin = A.node_begin();
      node_iterator a_end   = A.node_end();

      point_query_type point_query;

      point_query.m_precision = precision;
      point_query.m_points = const_cast<point_container*>(&pointsA);

      point_query.auto_init_settings( a_begin, a_end );
      point_query.init_data( a_begin, a_end);

      node_iterator a = a_begin;
      node_lut[a->idx()] = B.insert(a->m_coord)->idx();
      ++a;
      for(; a!=a_end; ++a)
      {
        result_container results;

        point_query( *a , results, typename point_query_type::no_collisions_tag());

        result_iterator result = results.begin();
        result_iterator r_end  = results.end();
        unsigned int b = ~0u;
        for(;result!=r_end;++result)
        {
          //--- See if we already have created a node in B corresponding to the node we hit in A
          b = node_lut[ (*result)->idx() ];          
          //--- If a corresponding node already exist in B then we stop looking for more colliding nodes in A
          if(b != ~0u)
            break;
        }
        //--- If we did find a existing node in B then we simply let the
        //--- node from A point to the corresponding node in B. That is we
        //--- have multiple nodes in A pointing to a single node in B.
        //---
        //--- If we did not find any colliding nodes in A then it means that
        //--- the node we are looking at from A lies sufficently far away from
        //--- any other nodes, so we simply create a corresponding node in B. Or
        //--- if we did find collisions but no corresponding nodes in B then it
        //--- means we are looking at the first node from A lying in some ``cluster''
        //--- of nodes (that all will be merged into one single corresponding node
        //--- in B). In both of the above cases we must create a new corresponding
        //--- node in B.
        if(b != ~0u)
          node_lut[ a->idx() ] = b;
        else
          node_lut[ a->idx() ] = B.insert(a->m_coord)->idx();
      }

      //--- create coordinate points in B
      {
        pointsB.clear();
        pointsB.resize( B.size_nodes() );
        for(node_iterator a = a_begin; a!=a_end; ++a)
          pointsB[  node_lut[a->idx()]  ] = pointsA[ a->idx() ];
      }

      //--- create tetrahedra in output mesh
      tetrahedron_iterator T   = A.tetrahedron_begin();
      tetrahedron_iterator end = A.tetrahedron_end();
      for(;T!=end;++T)
      {
        unsigned int idx_i = node_lut[ T->i()->idx() ];
        unsigned int idx_j = node_lut[ T->j()->idx() ];
        unsigned int idx_k = node_lut[ T->k()->idx() ];
        unsigned int idx_m = node_lut[ T->m()->idx() ];

        if(idx_i == idx_j)
          continue;
        if(idx_i == idx_k)
          continue;
        if(idx_i == idx_m)
          continue;

        if(idx_j == idx_k)
          continue;
        if(idx_j == idx_m)
          continue;

        if(idx_k == idx_m)
          continue;

        B.insert(idx_i,idx_j,idx_k,idx_m);
      }

      std::cout << "remove_redundant_nodes(): removed "
        << (A.size_nodes() - B.size_nodes()) 
        << " nodes and " 
        << (A.size_tetrahedra() - B.size_tetrahedra()) 
        << " tetrahedra" 
        << std::endl;
    }

    template<typename volume_mesh>
    void remove_redundant_nodes(
      volume_mesh & A
      , volume_mesh & B
      , double precision = 10e-7
      )
    {

      typedef default_point_container<volume_mesh> point_container;
      point_container pointsA(&A);
      point_container pointsB(&B);
      remove_redundant_nodes(A,pointsA,B,pointsB,precision);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_REMOVE_REDUDANT_NODES_H
#endif
