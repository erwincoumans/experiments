#ifndef OCTREE_H
#define OCTREE_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <limits>

template<class vec>
class BBox : public sofa::helper::fixed_array<vec,2>
{
public:
    typedef vec Coord;
    typedef typename Coord::value_type Real;
    enum { N = Coord::static_size };
    typedef sofa::helper::fixed_array<Coord,2> Inherit;

    static Real rmin() { return std::numeric_limits<Real>::min(); }
    static Real rmax() { return std::numeric_limits<Real>::max(); }
    
    BBox(const Coord& a, const Coord& b) : Inherit(a,b) {}
    BBox() : Inherit(Coord(rmax(),rmax(),rmax()),Coord(rmin(),rmin(),rmin())) {}
    const Coord& a() const { return this->at(0); }
          Coord& a()       { return this->at(0); }
    const Coord& b() const { return this->at(1); }
          Coord& b()       { return this->at(1); }
    bool empty() const { return a()[0] > b()[0]; }
    void set(const Coord& bmin, const Coord& bmax) { a() = bmin; b() = bmax; }
    void clear() { set(Coord(rmax(),rmax(),rmax()), Coord(rmin(),rmin(),rmin())); }
    void add(const Coord& p)
    {
        for (int c=0;c<N;++c)
        {
            if (p[c] < a()[c]) a()[c] = p[c];
            if (p[c] > b()[c]) b()[c] = p[c];
        }
    }
    void add(const BBox<Coord>& bb)
    {
        for (int c=0;c<N;++c)
        {
            if (bb.a()[c] < a()[c]) a()[c] = bb.a()[c];
            if (bb.b()[c] > b()[c]) b()[c] = bb.b()[c];
        }
    }
    template<class Iter>
    void add(Iter begin, Iter end)
    {
        for(;begin != end;++begin)
            add(*begin);
    }
    template<class Iter, class Container>
    void add(Iter begin, Iter end, const Container& points)
    {
        for(;begin != end;++begin)
            add(points[*begin]);
    }
    bool isIn(const Coord& v) const
    {
        bool r = true;
        for (int c=0;c<N;++c)
            r = r && v[c] >= a()[c] && v[c] <= b()[c];
        return r;
    }
    bool intersect(const BBox<Coord>& bb) const
    {
        bool r = true;
        for (int c=0;c<N;++c)
            r = r && a()[c] <= bb.b()[c] && b()[c] >= bb.a()[c];
        return r;
    }
};

template<class vec>
class Octree
{
public:
    typedef vec Coord;
    typedef typename Coord::value_type Real;
    enum { N = Coord::static_size };
    enum { NCHILDREN = 1 << N };
    typedef BBox<Coord> TBBox;
    typedef Octree<Coord> TOctree;
    TBBox m_bbox;
    TOctree* m_children; // NCHILDREN child nodes, if not leaf
    sofa::helper::vector<int> m_elems; // contained elements, if leaf
    Octree() : m_children(NULL) {}
    ~Octree() { delete[] m_children; }
    const TBBox& bbox() const { return m_bbox; }
    const sofa::helper::vector<int>& elems() const { return m_elems; }
    bool isLeaf() const { return m_children==NULL; }
    void init(const TBBox& bbox, const sofa::helper::vector<TBBox>& elemBB, const sofa::helper::vector<int>& elems, int elemsPerLeaf = 8, int maxLevel = 6)
    {
        m_bbox = bbox;
        sofa::helper::vector<int> myelems;
        myelems.reserve(elems.size());
        for (std::size_t i=0;i<elems.size();++i)
            if (bbox.intersect(elemBB[elems[i]]))
                myelems.push_back(elems[i]);
        if (maxLevel <= 0 || (int)myelems.size() <= elemsPerLeaf)
        { // leaf
            m_elems.swap(myelems);
        }
        else
        { // recursion
            m_children = new TOctree[NCHILDREN];
            Coord center = (bbox[0] + bbox[1])*0.5f;
            for (int i=0;i<NCHILDREN;++i)
            {
                TBBox bb = bbox;
                for (int c=0;c<N;++c)
                    bb[((i>>c)&1)^1][c] = center[c];
                m_children[i].init(bb, elemBB, myelems, elemsPerLeaf, maxLevel-1);
            }
        }
    }
    void init(const sofa::helper::vector<TBBox>& elemBB, int elemsPerLeaf = 8, int maxLevel = 6)
    {
        TBBox bbox;
        bbox.clear();
        bbox.add(elemBB.begin(), elemBB.end());
        sofa::helper::vector<int> elems;
        elems.resize(elemBB.size());
        for (std::size_t i=0;i<elemBB.size();++i)
            elems[i] = (int)i;
        init(bbox, elemBB, elems, elemsPerLeaf, maxLevel);
    }
    /// Find the leaf cell containing p
    TOctree* find(const Coord& p)
    {
        TOctree* n = this;
        while (!n->isLeaf())
        {
            n = n->m_children;
            Coord center = n->bbox()[1];
            for (int c=0;c<N;++c)
                if (p[c] > center[c]) n += 1<<c;
        }
        return n;
    }
    /// Find a non-empty leaf cell close to p
    TOctree* findNear(const Coord& p)
    {
        TOctree* n = this;
        while (!n->isLeaf())
        {
            TOctree* n2 = n->m_children;
            Coord center = n2->bbox()[1];
            int child = 0;
            for (int c=0;c<N;++c)
                if (p[c] > center[c]) child += 1<<c;
            if (n2[child].isLeaf() && n2[child].elems().empty())
            { // we need to look at the other children
                Coord diff2 = p - center;
                for (int c=0;c<N;++c) diff2[c] = diff2[c]*diff2[c];
                int bestchild = -1;
                Real bestdist = TBBox::rmax();
                for (int i=0;i<NCHILDREN;++i)
                {
                    if (n2[i].isLeaf() && n2[i].elems().empty()) continue;
                    Real dist = 0;
                    for (int c=0;c<N;++c)
                        if ((i ^ child) & (1 << c)) dist += diff2[c];
                    if (dist < bestdist)
                    {
                        bestchild = i;
                        bestdist = dist;
                    }
                }
                child = bestchild;
            }
            n = n2 + child;
        }
        return n;
    }
    /// Find all leaf cells within a given distance of p
    void findAllAround(sofa::helper::vector<TOctree*>& cells, const Coord& p, Real dist)
    {
        TOctree* n = this;
        if (n->isLeaf())
        {
            cells.push_back(n);
            return;
        }
        while (n) // && !n->isLeaf())
        {
            TOctree* n2 = n->m_children;
            Coord center = n2->bbox()[1];
            n = NULL;
            for (int child = 0; child < NCHILDREN; ++child)
            {
                bool out = false;
                for (int c=0;c<N;++c)
                    if (child & (1<<c))
                    {
                        if (p[c] + dist < center[c]) out = true;
                    }
                    else
                    {
                        if (p[c] - dist > center[c]) out = true;
                    }
                if (out) continue;
                if (n2[child].isLeaf())
                    cells.push_back(n2+child);
                else if (n == NULL)
                    n = n2 + child;
                else // we must use recursion
                    n2[child].findAllAround(cells, p, dist);
            }
        }
    }
};

#endif
