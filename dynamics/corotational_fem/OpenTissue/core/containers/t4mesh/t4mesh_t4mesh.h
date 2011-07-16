#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_T4MESH_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_T4MESH_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/t4mesh/t4mesh_t4node.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_t4tetrahedron.h>
#include <OpenTissue/core/containers/t4mesh/t4mesh_core_access.h>
#include <OpenTissue/utility/utility_index_iterator.h>
#include <OpenTissue/core/math/math_constants.h>

#include <vector>
#include <list>
#include <cassert>

namespace OpenTissue
{
  namespace t4mesh
  {
    namespace detail
    {

      /**
      * Basic (Simple) Tetrahedra Mesh.
      *
      * This tetrahedra mesh data structure is designed specially for
      * two purposes: It should maintain a valid topology of the mesh
      * at all times, that is the connectivity of nodes and tetrahedra
      * are always valid.
      *
      * The other purpose is to make sure that the global indexing of
      * nodes (0..N-1) and tetrahedra (0..T-1) always is a compact range
      * starting from zero to the maximum number minus one.
      *
      * Obviously removing entities (nodes or tetrahedra) alters the global
      * index ranges, thus end users can not trust previously stored indices
      * of entities in their own apps.
      *
      * The mesh takes three template arguments. The first specifies the
      * math_types used in the mesh. The following two arguments are node
      * traits and tetrahedron traits respectively.
      */
      template<
        typename M
        , typename N
        , typename T
      >
      class T4Mesh
      {
      public:

        typedef M                        math_types;
        typedef N                        node_traits;
        typedef T                        tetrahedron_traits;
        typedef T4Mesh<M,N,T>            mesh_type;
        typedef T4Node<mesh_type>        node_type;
        typedef T4Tetrahedron<mesh_type> tetrahedron_type;
        typedef size_t                   index_type;

        /**
        * Undefined index value. 
        *
        * @return   An index value that means that the index value is undefined. The
        *          largest possible value of the index type is used for this purpose.
        */
        static index_type const & undefined() 
        {
          static index_type value = FLT_MAX;//math::detail::highest<index_type>();
          return value;
        }

      protected:

        typedef std::vector< node_type >        node_container;
        typedef std::vector< tetrahedron_type > tetrahedra_container;

	  public:

        node_container           m_nodes;            ///< Internal node storage.
        tetrahedra_container     m_tetrahedra;       ///< Internal tetrahedra storage.

      public:

        
      public:

        T4Mesh() 
          : m_nodes()
          , m_tetrahedra() 
        {}


        T4Mesh( T4Mesh const & cpy) 
        {
          *this = cpy;
        }

        T4Mesh & operator=(T4Mesh const & rhs) 
        {
          this->m_nodes = rhs.m_nodes;
          this->m_tetrahedra = rhs.m_tetrahedra;
          for(node_iterator n = this->node_begin();n!=this->node_end();++n)
            t4mesh_core_access::set_owner( (*n) , this );
          for(tetrahedron_iterator t = this->tetrahedron_begin();t!=this->tetrahedron_end();++t)
            t4mesh_core_access::set_owner( (*t) , this );
          return (*this);
        }


      public:

        void clear()
        {
          m_nodes.clear();
          m_tetrahedra.clear();
        }


      public:

        /**
        * Add New Node.
        * New node will have index value equal to number of nodes prior to insertion.
        *
        * @return            A iterator to the new node
        */
        void insert()
        {
          m_nodes.push_back( node_type() );
          node_type & nd = m_nodes.back();
          t4mesh_core_access::set_index( nd, this->m_nodes.size()-1 );
          t4mesh_core_access::set_owner( nd, this );
          //return node_iterator(m_nodes, nd.idx());
        }

        /**
        * Add New Node.
        * This method do not watch against creating multiple nodes with same coordinates.
        *
        * @param coord    The new coordinate of the node
        *
        * @return          An iterator to the newly created node.
        */
        template<typename vector3_type>
        void	insert(vector3_type const & coord)
        {
          node_iterator node = insert();
          node->m_coord =  coord ;
          //return node;
        }

        /**
        * Overloaded insert method for tetrahedron, support index-based insertion.
        *
        * This is a bit slower than using the iterator-based insertion method directly.
        * But it makes it easier to code....
        *
        */
        void	insert(
          index_type i,
          index_type j,
          index_type k,
          index_type m
          )
        {

//          assert(find(i,j,k,m)==tetrahedron_end() || !"T4Mesh::insert(): Tetrahedron already exists in mesh");

          m_tetrahedra.push_back( tetrahedron_type() );
          tetrahedron_type & t = m_tetrahedra.back();

          t4mesh_core_access::set_index( t, m_tetrahedra.size() - 1 );
          t4mesh_core_access::set_owner( t, this );
          t4mesh_core_access::set_node0( t, i );
          t4mesh_core_access::set_node1( t, j );
          t4mesh_core_access::set_node2( t, k );
          t4mesh_core_access::set_node3( t, m );

          t4mesh_core_access::tetrahedra_push_back( m_nodes[i], t.idx() );
          t4mesh_core_access::tetrahedra_push_back( m_nodes[j], t.idx() );
          t4mesh_core_access::tetrahedra_push_back( m_nodes[k], t.idx() );
          t4mesh_core_access::tetrahedra_push_back( m_nodes[m], t.idx() );

          //return tetrahedron_iterator(m_tetrahedra, t.idx());
        }

        /**
        * Find tetrahedron with the given nodes
        *
        * @param i           The global index of i'th node
        * @param j           The global index of j'th node
        * @param k           The global index of k'th node
        * @param m           The global index of m'th node
        *
        * @return        A iterator to the tetrahedron if it exist
        *                or end position otherwise.
        */
        index_type find(
          index_type i,
          index_type j,
          index_type k,
          index_type m
          ) const
        {


			for (int t=0;t<m_tetrahedra.size();t++)
			{
				if (m_tetrahedra[t].node_idx(0) == i &&
					m_tetrahedra[t].node_idx(1) == j &&
					m_tetrahedra[t].node_idx(2) == k &&
					m_tetrahedra[t].node_idx(3) == m)
				{
	              return tit->idx();
				}
			}
			return m_tetrahedra.size();
        }

        /**
        * Erase Tetrahedron at specified Position.
        *
        * @param where
        *
        * @return
        */
        index_type	erase(index_type& where)
        {
			index_type last = m_tetrahedra.size()-1;
			if (where != last)
			{
				this->swap(where,last);
			}

          this->unlink(last);
          //--- This might be a bit stupid, it would
          //--- proberly be better to keep track of last unused
          //--- entry and only resize vector when there is no
          //--- more space
          m_tetrahedra.pop_back();            
          return where;
        }

      protected:


        /**
        * Create new nodal connections for the specified Tetrahedron.
        */
        void link(
          index_type& tetIndex
          , index_type nodeIndexI
          , index_type nodeIndexJ
          , index_type nodeIndexK
          , index_type nodeIndexM
          )
        {

          t4mesh_core_access::tetrahedra_push_back( m_nodex[nodeIndexI],tetIndex );
          t4mesh_core_access::tetrahedra_push_back( m_nodex[nodeIndexJ],tetIndex );
          t4mesh_core_access::tetrahedra_push_back( m_nodex[nodeIndexK],tetIndex );
          t4mesh_core_access::tetrahedra_push_back( m_nodex[nodeIndexM],tetIndex );

          t4mesh_core_access::set_node0( m_tetrahedra[tetIndex],nodeIndexI );
          t4mesh_core_access::set_node0( m_tetrahedra[tetIndex],nodeIndexJ );
          t4mesh_core_access::set_node0( m_tetrahedra[tetIndex],nodeIndexK );
          t4mesh_core_access::set_node0( m_tetrahedra[tetIndex],nodeIndexM );
        }

        /**
        * Swap any internal data and nodal connections between
        * the two specified Tetrahedra.
        *
        * This metod is intended for internal usage only.
        */
        void swap(index_type& Aidx,index_type& Bidx)
        {
          if(Aidx==Bidx)
            throw std::invalid_argument("T4Mesh::swap(A,B): A and B were the same");

		  node_type* A = &m_tetrahedra[Aidx];
		node_type* B = &m_tetrahedra[Bidx];

          node_iterator Ai   = A->i();
          node_iterator Aj   = A->j();
          node_iterator Ak   = A->k();
          node_iterator Am   = A->m();

          node_iterator Bi   = B->i();
          node_iterator Bj   = B->j();
          node_iterator Bk   = B->k();
          node_iterator Bm   = B->m();

          //--- Remove all connections to nodes, that is i,j,k, and
          //--- m are set to -1 on both tetrahedra.
          unlink(A);
          unlink(B);
          //--- Swap any internal data stored in the tetrahedra,
          //--- notice that this is done using the trait class.
          tetrahedron_traits * TA = &(*A);
          tetrahedron_traits tmp;
          tetrahedron_traits * TB = &(*B);
          tmp = (*TA);
          (*TA) = (*TB);
          (*TB) = tmp;
          //--- Just to be on the safe side we reestablish tetrahedron
          //--- indices...
          t4mesh_core_access::set_index(*A, Aidx);
          t4mesh_core_access::set_index(*B, Bidx);
          //--- Finally we set up the nodal connections again
          link(A,Bi,Bj,Bk,Bm);
          link(B,Ai,Aj,Ak,Am);
        }

      protected:


      };

    } // namespace detail
  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_T4MESH_H
#endif
