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
          static index_type value = math::detail::highest<index_type>();
          return value;
        }

      protected:

        typedef std::vector< node_type >        node_container;
        typedef std::vector< tetrahedron_type > tetrahedra_container;

      protected:

        node_container           m_nodes;            ///< Internal node storage.
        tetrahedra_container     m_tetrahedra;       ///< Internal tetrahedra storage.

      public:

        typedef OpenTissue::utility::IndexIterator<index_type, tetrahedra_container> tetrahedron_iterator;
        typedef OpenTissue::utility::IndexIterator<index_type, tetrahedra_container> const_tetrahedron_iterator;

        tetrahedron_iterator       tetrahedron_begin()       { return tetrahedron_iterator( m_tetrahedra, 0 ); }
        tetrahedron_iterator       tetrahedron_end()         { return tetrahedron_iterator( m_tetrahedra, size_tetrahedra() ); }
        const_tetrahedron_iterator tetrahedron_begin() const { return const_tetrahedron_iterator( m_tetrahedra, 0 ); }
        const_tetrahedron_iterator tetrahedron_end()   const { return const_tetrahedron_iterator( m_tetrahedra, size_tetrahedra() ); }

        typedef OpenTissue::utility::IndexIterator<index_type, node_container> node_iterator;
        typedef OpenTissue::utility::IndexIterator<index_type, node_container> const_node_iterator;

        node_iterator       node_begin()       { return node_iterator( m_nodes, 0 ); }
        node_iterator       node_end()         { return node_iterator( m_nodes, size_nodes() ); }
        const_node_iterator node_begin() const { return const_node_iterator( m_nodes, 0 ); }
        const_node_iterator node_end()   const { return const_node_iterator( m_nodes, size_nodes() ); }

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

        node_iterator node(index_type idx)
        {
          if(!(idx>=0 && idx<size_nodes()))
            throw std::out_of_range("T4Mesh::node(idx): idx out of range");
          return node_iterator(m_nodes,idx);
        }

        const_node_iterator const_node(index_type idx) const
        {
          if(!(idx>=0 && idx<size_nodes()))
            throw std::out_of_range("T4Mesh::const_node(idx): idx out of range");
          return const_node_iterator(m_nodes,idx);
        }

        tetrahedron_iterator tetrahedron(index_type idx)
        {
          if(!(idx>=0 && idx<size_tetrahedra()))
            throw std::out_of_range("T4Mesh::tetrahedron(idx): idx out of range");
          return tetrahedron_iterator(m_tetrahedra,idx);
        }

        const_tetrahedron_iterator tetrahedron(index_type idx)const
        {
          if(!(idx>=0 && idx<size_tetrahedra()))
            throw std::out_of_range("T4Mesh::tetrahedron(idx): idx out of range");
          return const_tetrahedron_iterator(m_tetrahedra,idx);
        }

        size_t size_nodes()      const { return m_nodes.size(); }
        size_t size_tetrahedra() const { return m_tetrahedra.size(); }

      public:

        /**
        * Add New Node.
        * New node will have index value equal to number of nodes prior to insertion.
        *
        * @return            A iterator to the new node
        */
        node_iterator insert()
        {
          m_nodes.push_back( node_type() );
          node_type & nd = m_nodes.back();
          t4mesh_core_access::set_index( nd, size_nodes()-1 );
          t4mesh_core_access::set_owner( nd, this );
          return node_iterator(m_nodes, nd.idx());
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
        node_iterator insert(vector3_type const & coord)
        {
          node_iterator node = insert();
          node->m_coord =  coord ;
          return node;
        }

        /**
        * Overloaded insert method for tetrahedron, support index-based insertion.
        *
        * This is a bit slower than using the iterator-based insertion method directly.
        * But it makes it easier to code....
        *
        */
        tetrahedron_iterator insert(
          index_type i,
          index_type j,
          index_type k,
          index_type m
          )
        {
          return insert(
            node_iterator(m_nodes,i),
            node_iterator(m_nodes,j),
            node_iterator(m_nodes,k),
            node_iterator(m_nodes,m)
            );
        }

        tetrahedron_iterator insert(
          node_iterator  i,
          node_iterator  j,
          node_iterator  k,
          node_iterator  m
          )
        {
          verify_nodes(i,j,k,m);

          assert(find(i,j,k,m)==tetrahedron_end() || !"T4Mesh::insert(): Tetrahedron already exists in mesh");

          m_tetrahedra.push_back( tetrahedron_type() );
          tetrahedron_type & t = m_tetrahedra.back();

          t4mesh_core_access::set_index( t, size_tetrahedra() - 1 );
          t4mesh_core_access::set_owner( t, this );
          t4mesh_core_access::set_node0( t, i->idx() );
          t4mesh_core_access::set_node1( t, j->idx() );
          t4mesh_core_access::set_node2( t, k->idx() );
          t4mesh_core_access::set_node3( t, m->idx() );

          t4mesh_core_access::tetrahedra_push_back( *i, t.idx() );
          t4mesh_core_access::tetrahedra_push_back( *j, t.idx() );
          t4mesh_core_access::tetrahedra_push_back( *k, t.idx() );
          t4mesh_core_access::tetrahedra_push_back( *m, t.idx() );

          return tetrahedron_iterator(m_tetrahedra, t.idx());
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
        tetrahedron_iterator find(
          node_iterator i,
          node_iterator j,
          node_iterator k,
          node_iterator m
          ) const
        {
          verify_nodes(i,j,k,m);

          typename node_type::tetrahedron_circulator tit = i->begin();
          for(;tit!=i->end();++tit)
          {
            if(
              tit->node_idx(0) == i->idx() &&
              tit->node_idx(1) == j->idx() &&
              tit->node_idx(2) == k->idx() &&
              tit->node_idx(3) == m->idx() 
              )
              return tetrahedron_iterator(m_tetrahedra,tit->idx());
          }
          return tetrahedron_iterator(m_tetrahedra,size_tetrahedra());
        }

        /**
        * Erase Tetrahedron at specified Position.
        *
        * @param where
        *
        * @return
        */
        tetrahedron_iterator erase(tetrahedron_iterator & where)
        {
          verify_tetrahedron(where);

          tetrahedron_iterator I = tetrahedron(size_tetrahedra()-1);
          tetrahedron_iterator last(m_tetrahedra,I->idx());
          if(where!=last)
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
        * Remove all nodal connections from specified Tetrahedron.
        */
        void unlink(tetrahedron_iterator & I)
        {
          verify_tetrahedron(I);

          node_iterator i = I->i();
          node_iterator j = I->j();
          node_iterator k = I->k();
          node_iterator m = I->m();

          verify_nodes(i,j,k,m);

          t4mesh_core_access::tetrahedra_remove( *i, I->idx() );
          t4mesh_core_access::tetrahedra_remove( *j, I->idx() );
          t4mesh_core_access::tetrahedra_remove( *k, I->idx() );
          t4mesh_core_access::tetrahedra_remove( *m, I->idx() );

          t4mesh_core_access::set_node0( *I, this->undefined());
          t4mesh_core_access::set_node1( *I, this->undefined());
          t4mesh_core_access::set_node2( *I, this->undefined());
          t4mesh_core_access::set_node3( *I, this->undefined());
        }

        /**
        * Create new nodal connections for the specified Tetrahedron.
        */
        void link(
          tetrahedron_iterator & I
          , node_iterator & i
          , node_iterator & j
          , node_iterator & k
          , node_iterator & m
          )
        {
          verify_tetrahedron(I);

          if(I->node_idx(0)!=this->undefined())
            throw std::invalid_argument("T4Mesh::link(): node 0 on T was not undefined");
          if(I->node_idx(1)!=this->undefined())
            throw std::invalid_argument("T4Mesh::link(): node 1 on T was not undefined");
          if(I->node_idx(2)!=this->undefined())
            throw std::invalid_argument("T4Mesh::link(): node 2 on T was not undefined");
          if(I->node_idx(3)!=this->undefined())
            throw std::invalid_argument("T4Mesh::link(): node 3 on T was not undefined");

          verify_nodes(i,j,k,m);

          t4mesh_core_access::tetrahedra_push_back( *i, I->idx() );
          t4mesh_core_access::tetrahedra_push_back( *j, I->idx() );
          t4mesh_core_access::tetrahedra_push_back( *k, I->idx() );
          t4mesh_core_access::tetrahedra_push_back( *m, I->idx() );

          t4mesh_core_access::set_node0( *I, i->idx() );
          t4mesh_core_access::set_node1( *I, j->idx() );
          t4mesh_core_access::set_node2( *I, k->idx() );
          t4mesh_core_access::set_node3( *I, m->idx() );
        }

        /**
        * Swap any internal data and nodal connections between
        * the two specified Tetrahedra.
        *
        * This metod is intended for internal usage only.
        */
        void swap(tetrahedron_iterator & A,tetrahedron_iterator & B)
        {
          if(A==B)
            throw std::invalid_argument("T4Mesh::swap(A,B): A and B were the same");

          verify_tetrahedron(A);
          verify_tetrahedron(B);

          index_type      Aidx = A->idx();
          index_type      Bidx = B->idx();

          node_iterator Ai   = A->i();
          node_iterator Aj   = A->j();
          node_iterator Ak   = A->k();
          node_iterator Am   = A->m();

          verify_nodes(Ai,Aj,Ak,Am);

          node_iterator Bi   = B->i();
          node_iterator Bj   = B->j();
          node_iterator Bk   = B->k();
          node_iterator Bm   = B->m();

          verify_nodes(Bi,Bj,Bk,Bm);

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

        void verify_nodes(
          node_iterator const & i
          , node_iterator const & j
          , node_iterator const & k
          , node_iterator const & m
          ) const
        {
          if(i->owner()!=this)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): i did not belong to this mesh");
          }
          if(j->owner()!=this)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): j did not belong to this mesh");
          }
          if(k->owner()!=this)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): k did not belong to this mesh");
          }
          if(m->owner()!=this)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): l did not belong to this mesh");
          }
          if(!( i->idx()>=0 && i->idx()< size_nodes() ) )
          {
            throw std::out_of_range("T4MEsh::verify_nodes(i,j,k,m): index value of i was out of range");
          }
          if(!( j->idx()>=0 && j->idx()< size_nodes() ) )
          {
            throw std::out_of_range("T4MEsh::verify_nodes(i,j,k,m): index value of j was out of range");
          }
          if(!( k->idx()>=0 && k->idx()< size_nodes() ) )
          {
            throw std::out_of_range("T4MEsh::verify_nodes(i,j,k,m): index value of k was out of range");
          }
          if(!( m->idx()>=0 && m->idx()< size_nodes() ) )
          {
            throw std::out_of_range("T4MEsh::verify_nodes(i,j,k,m): index value of m was out of range");
          }
          if(i==j)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): i and j was equal");
          }
          if(i==k)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): i and k was equal");
          }
          if(i==m)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): i and m was equal");
          }
          if(j==k)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): j and k was equal");
          }
          if(j==m)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): j and m was equal");
          }
          if(k==m)
          {
            throw std::logic_error("T4MEsh::verify_nodes(i,j,k,m): j and m was equal");
          }
        }

        void verify_tetrahedron(tetrahedron_iterator const & I)const
        {
          if( !(   (I->idx() >=0)&&(I->idx() < size_tetrahedra())  )  )
            throw std::out_of_range("T4MEsh::verify_tetrahedron(I): index was out of range");

          if(I->owner()!=this)
          {
            throw std::logic_error("T4MEsh::verify_tetrahedron(I): I did not belong to this mesh");
          }
        }

      };

    } // namespace detail
  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_T4MESH_H
#endif
