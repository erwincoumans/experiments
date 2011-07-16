#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_T4TETRAHEDRON_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_T4TETRAHEDRON_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <boost/array.hpp>
#include <cassert>

namespace OpenTissue
{
  namespace t4mesh
  {

    template< typename mesh_type_>
    class T4Tetrahedron : public mesh_type_::tetrahedron_traits
    {
    public:
      typedef          mesh_type_                        mesh_type;
      typedef typename mesh_type::node_type              node_type;
      typedef typename mesh_type::tetrahedron_type       tetrahedron_type;
      typedef typename mesh_type::index_type             index_type;

      typedef typename mesh_type::node_iterator          node_iterator;
      typedef typename mesh_type::const_node_iterator    const_node_iterator;

      typedef typename mesh_type::tetrahedron_iterator   tetrahedron_iterator;
      typedef typename node_type::tetrahedron_circulator tetrahedron_circulator;

    protected:

      index_type                m_idx;      ///< Global index of tetrahedron
      mesh_type               * m_owner;    ///< Pointer to mesh which the node belongs to.
      boost::array<index_type, 4> m_nodes;  ///< Global index of node i,j,k and m

    private:

      friend class t4mesh_core_access;

      void set_index(index_type idx)    { m_idx = idx;      }
      void set_owner(mesh_type * owner) { m_owner = owner;  }
      void set_node0(index_type idx)    { m_nodes[0] = idx; }
      void set_node1(index_type idx)    { m_nodes[1] = idx; }
      void set_node2(index_type idx)    { m_nodes[2] = idx; }
      void set_node3(index_type idx)    { m_nodes[3] = idx; }

    public:

      T4Tetrahedron() 
        : m_idx( mesh_type::undefined() )
        , m_owner(0) 
      { 
        m_nodes.assign( mesh_type::undefined() ); 
      }

    public:

      index_type            idx()   const { return m_idx; }

      index_type            node_idx(index_type const & local_idx) const 
      { 
        assert(0<=local_idx);
        assert(local_idx<=3);

        return m_nodes[local_idx]; 
      }

      node_iterator       i()         { return m_owner->node(m_nodes[0]);       }
      const_node_iterator i() const   { return m_owner->const_node(m_nodes[0]); }
      node_iterator       j()         { return m_owner->node(m_nodes[1]);       }
      const_node_iterator j() const   { return m_owner->const_node(m_nodes[1]); }
      node_iterator       k()         { return m_owner->node(m_nodes[2]);       }
      const_node_iterator k() const   { return m_owner->const_node(m_nodes[2]); }
      node_iterator       m()         { return m_owner->node(m_nodes[3]);       }
      const_node_iterator m() const   { return m_owner->const_node(m_nodes[3]); }

      tetrahedron_iterator jkm() const
      {
        node_iterator a = j();
        node_iterator b = k();
        node_iterator c = m();
        for(tetrahedron_circulator it = a->begin();it!=a->end();++it)
          if(it->has_face(c,b,a))
            return m_owner->tetrahedron(it->idx());
        return m_owner->tetrahedron_end();
      }

      tetrahedron_iterator ijm()const
      {
        node_iterator a = i();
        node_iterator b = j();
        node_iterator c = m();
        for(tetrahedron_circulator it = a->begin();it!=a->end();++it)
          if(it->has_face(c,b,a))
            return m_owner->tetrahedron(it->idx());
        return m_owner->tetrahedron_end();
      }

      tetrahedron_iterator kim()const
      {
        node_iterator a = k();
        node_iterator b = i();
        node_iterator c = m();
        for(tetrahedron_circulator it = a->begin();it!=a->end();++it)
          if(it->has_face(c,b,a))
            return m_owner->tetrahedron(it->idx());
        return m_owner->tetrahedron_end();
      }

      tetrahedron_iterator ikj()const
      {
        node_iterator a = i();
        node_iterator b = k();
        node_iterator c = j();
        for(tetrahedron_circulator it = a->begin();it!=a->end();++it)
          if(it->has_face(c,b,a))
            return m_owner->tetrahedron(it->idx());
        return m_owner->tetrahedron_end();
      }

      mesh_type       * owner()       { return m_owner; }
      mesh_type const * owner() const { return m_owner; }


      node_iterator node(index_type local_idx)       
      {
        return m_owner->node( this->local2global( local_idx) );       
      }

      const_node_iterator node(index_type local_idx) const 
      {
        return m_owner->const_node( this->local2global( local_idx ) ); 
      }

      index_type local2global(index_type local_idx) const
      {
        assert(0<=local_idx);
        assert(local_idx<=3);
        return m_nodes[local_idx];
      }

      index_type global2local(index_type global_idx) const
      {
        if(global_idx==m_nodes[0])
          return 0;
        if(global_idx==m_nodes[1])
          return 1;
        if(global_idx==m_nodes[2])
          return 2;
        if(global_idx==m_nodes[3])
          return 3;
        return mesh_type::undefined();
      }

      bool has_face(node_iterator a,node_iterator b,node_iterator c) const
      {
        index_type i = m_nodes[0];
        index_type j = m_nodes[1];
        index_type k = m_nodes[2];
        index_type m = m_nodes[3];
        if(
          (  i == a->idx()   &&   j == b->idx()  &&   m == c->idx() )||
          (  j == a->idx()   &&   k == b->idx()  &&   m == c->idx() )||
          (  k == a->idx()   &&   i == b->idx()  &&   m == c->idx() )||
          (  i == a->idx()   &&   k == b->idx()  &&   j == c->idx() )||
          (  i == b->idx()   &&   j == c->idx()  &&   m == a->idx() )||
          (  j == b->idx()   &&   k == c->idx()  &&   m == a->idx() )||
          (  k == b->idx()   &&   i == c->idx()  &&   m == a->idx() )||
          (  i == b->idx()   &&   k == c->idx()  &&   j == a->idx() )||
          (  i == c->idx()   &&   j == a->idx()  &&   m == b->idx() )||
          (  j == c->idx()   &&   k == a->idx()  &&   m == b->idx() )||
          (  k == c->idx()   &&   i == a->idx()  &&   m == b->idx() )||
          (  i == c->idx()   &&   k == a->idx()  &&   j == b->idx() )
          )
          return true;
        return false;
      }

    };

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_T4TETRAHEDRON_H
#endif
