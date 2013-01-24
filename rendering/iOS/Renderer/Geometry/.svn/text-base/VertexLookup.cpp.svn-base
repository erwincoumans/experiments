#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//#pragma warning(disable:4786)

#include <vector>

/*!
**
** Copyright (c) 2007 by John W. Ratcliff mailto:jratcliff@infiniplex.net
**
** Portions of this source has been released with the PhysXViewer application, as well as
** Rocket, CreateDynamics, ODF, and as a number of sample code snippets.
**
** If you find this code useful or you are feeling particularily generous I would
** ask that you please go to http://www.amillionpixels.us and make a donation
** to Troy DeMolay.
**
** DeMolay is a youth group for young men between the ages of 12 and 21.  
** It teaches strong moral principles, as well as leadership skills and 
** public speaking.  The donations page uses the 'pay for pixels' paradigm
** where, in this case, a pixel is only a single penny.  Donations can be
** made for as small as $4 or as high as a $100 block.  Each person who donates
** will get a link to their own site as well as acknowledgement on the
** donations blog located here http://www.amillionpixels.blogspot.com/
**
** If you wish to contact me you can use the following methods:
**
** Skype Phone: 636-486-4040 (let it ring a long time while it goes through switches)
** Skype ID: jratcliff63367
** Yahoo: jratcliff63367
** AOL: jratcliff1961
** email: jratcliff@infiniplex.net
** Personal website: http://jratcliffscarab.blogspot.com
** Coding Website:   http://codesuppository.blogspot.com
** FundRaising Blog: http://amillionpixels.blogspot.com
** Fundraising site: http://www.amillionpixels.us
** New Temple Site:  http://newtemple.blogspot.com
**
**
** The MIT license:
**
** Permission is hereby granted, free of charge, to any person obtaining a copy 
** of this software and associated documentation files (the "Software"), to deal 
** in the Software without restriction, including without limitation the rights 
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
** copies of the Software, and to permit persons to whom the Software is furnished 
** to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all 
** copies or substantial portions of the Software.

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/



// CodeSnippet provided by John W. Ratcliff
// on March 23, 2006.
//
// mailto: jratcliff@infiniplex.net
//
// Personal website: http://jratcliffscarab.blogspot.com
// Coding Website:   http://codesuppository.blogspot.com
// FundRaising Blog: http://amillionpixels.blogspot.com
// Fundraising site: http://www.amillionpixels.us
// New Temple Site:  http://newtemple.blogspot.com
//
// This snippet shows how to 'hide' the complexity of
// the STL by wrapping some useful piece of functionality
// around a handful of discrete API calls.
//
// This API allows you to create an indexed triangle list
// from a collection of raw input triangles.  Internally
// it uses an STL set to build the lookup table very rapidly.
//
// Here is how you would use it to build an indexed triangle
// list from a raw list of triangles.
//
// (1) create a 'VertexLookup' interface by calling
//
//     VertexLook vl = Vl_createVertexLookup();
//
// (2) For each vertice in each triangle call:
//
//     unsigned int i1 = Vl_getIndex(vl,p1);
//     unsigned int i2 = Vl_getIndex(vl,p2);
//     unsigned int i3 = Vl_getIndex(vl,p3);
//
//     save the 3 indices into your triangle list array.
//
// (3) Get the vertex array by calling:
//
//     const float *vertices = Vl_getVertices(vl);
//
// (4) Get the number of vertices so you can copy them into
//     your own buffer.
//     unsigned int vcount = Vl_getVcount(vl);
//
// (5) Release the VertexLookup interface when you are done with it.
//     Vl_releaseVertexLookup(vl);
//
// Teaches the following lessons:
//
//    How to wrap the complexity of STL and C++ classes around a
//    simple API interface.
//
//    How to use an STL set and custom comparator operator for
//    a complex data type.
//
//    How to create a template class.
//
//    How to achieve significant performance improvements by
//    taking advantage of built in STL containers in just
//    a few lines of code.
//
//    You could easily modify this code to support other vertex
//    formats with any number of interpolants.



#define USE_KDTREE 1

#include "VertexLookup.h"


//namespace Vlookup
//{


#if USE_KDTREE

class KdTreeNode;

typedef std::vector< KdTreeNode * > KdTreeNodeVector;

enum Axes
{
  X_AXIS = 0,
  Y_AXIS = 1,
  Z_AXIS = 2
};

class KdTreeFindNode
{
public:
  KdTreeFindNode(void)
  {
    mNode = 0;
    mDistance = 0;
  }
  KdTreeNode  *mNode;
  float        mDistance;
};

class KdTreeNode
{
public:

  KdTreeNode(float x,float y,float z,float radius,void *userData,unsigned int index)
  {
    mX = x;
    mY = y;
    mZ = z;
    mRadius = radius;
    mUserData = userData;
    mIndex = index;
    mLeft = 0;
    mRight = 0;
  };

	~KdTreeNode(void)
  {
  }


  void add(KdTreeNode *node,Axes dim)
  {
    switch ( dim )
    {
      case X_AXIS:
        if ( node->getX() <= getX() )
        {
          if ( mLeft )
            mLeft->add(node,Y_AXIS);
          else
            mLeft = node;
        }
        else
        {
          if ( mRight )
            mRight->add(node,Y_AXIS);
          else
            mRight = node;
        }
        break;
      case Y_AXIS:
        if ( node->getY() <= getY() )
        {
          if ( mLeft )
            mLeft->add(node,Z_AXIS);
          else
            mLeft = node;
        }
        else
        {
          if ( mRight )
            mRight->add(node,Z_AXIS);
          else
            mRight = node;
        }
        break;
      case Z_AXIS:
        if ( node->getZ() <= getZ() )
        {
          if ( mLeft )
            mLeft->add(node,X_AXIS);
          else
            mLeft = node;
        }
        else
        {
          if ( mRight )
            mRight->add(node,X_AXIS);
          else
            mRight = node;
        }
        break;
    }

  }

  float        getX(void) const { return mX;  }
  float        getY(void) const {  return mY; };
  float        getZ(void) const { return mZ; };
  float        getRadius(void) const { return mRadius; };
  unsigned int getIndex(void) const { return mIndex; };
  void *       getUserData(void) const { return mUserData; };

  void search(Axes axis,const float *pos,float radius,unsigned int &count,unsigned int maxObjects,KdTreeFindNode *found)
  {

    float dx = pos[0] - getX();
    float dy = pos[1] - getY();
    float dz = pos[2] - getZ();

    KdTreeNode *search1 = 0;
    KdTreeNode *search2 = 0;

    switch ( axis )
    {
      case X_AXIS:
       if ( dx <= 0 )     // JWR  if we are to the left
       {
        search1 = mLeft; // JWR  then search to the left
        if ( -dx < radius )  // JWR  if distance to the right is less than our search radius, continue on the right as well.
          search2 = mRight;
       }
       else
       {
         search1 = mRight; // JWR  ok, we go down the left tree
         if ( dx < radius ) // JWR  if the distance from the right is less than our search radius
	  			search2 = mLeft;
        }
        axis = Y_AXIS;
        break;
      case Y_AXIS:
        if ( dy <= 0 )
        {
          search1 = mLeft;
          if ( -dy < radius )
    				search2 = mRight;
        }
        else
        {
          search1 = mRight;
          if ( dy < radius )
    				search2 = mLeft;
        }
        axis = Z_AXIS;
        break;
      case Z_AXIS:
        if ( dz <= 0 )
        {
          search1 = mLeft;
          if ( -dz < radius )
    				search2 = mRight;
        }
        else
        {
          search1 = mRight;
          if ( dz < radius )
    				search2 = mLeft;
        }
        axis = X_AXIS;
        break;
    }

    float r2 = radius*radius;
    float m  = dx*dx+dy*dy+dz*dz;

    if ( m < r2 )
    {
      switch ( count )
      {
        case 0:
          found[count].mNode = this;
          found[count].mDistance = m;
          break;
        case 1:
          if ( m < found[0].mDistance )
          {
            if ( maxObjects == 1 )
            {
              found[0].mNode = this;
              found[0].mDistance = m;
            }
            else
            {
              found[1] = found[0];
              found[0].mNode = this;
              found[0].mDistance = m;
            }
          }
          else if ( maxObjects > 1)
          {
            found[1].mNode = this;
            found[1].mDistance = m;
          }
          break;
        default:
          if ( 1 )
          {
            bool inserted = false;

            for (unsigned int i=0; i<count; i++)
            {
              if ( m < found[i].mDistance ) // if this one is closer than a pre-existing one...
              {
                // insertion sort...
                unsigned int scan = count;
                if ( scan >= maxObjects ) scan=maxObjects-1;
                for (unsigned int j=scan; j>i; j--)
                {
                  found[j] = found[j-1];
                }
                found[i].mNode = this;
                found[i].mDistance = m;
                inserted = true;
                break;
              }
            }

            if ( !inserted && count < maxObjects )
            {
              found[count].mNode = this;
              found[count].mDistance = m;
            }
          }
          break;
      }
      count++;
      if ( count > maxObjects )
      {
        count = maxObjects;
      }
    }


    if ( search1 )
  		search1->search( axis, pos,radius, count, maxObjects, found);

    if ( search2 )
	  	search2->search( axis, pos,radius, count, maxObjects, found);

  }

  float distanceSquared(const float *pos) const
  {
    float dx = pos[0] - mX;
    float dy = pos[1] - mY;
    float dz = pos[2] - mZ;
    return dx*dx+dy*dy+dz*dz;
  }


private:

  void setLeft(KdTreeNode *left) { mLeft = left; };
  void setRight(KdTreeNode *right) { mRight = right; };

	KdTreeNode *getLeft(void)         { return mLeft; }
	KdTreeNode *getRight(void)        { return mRight; }

  unsigned int    mIndex;
  void           *mUserData;
  float           mX;
  float           mY;
  float           mZ;
  float           mRadius;
  KdTreeNode     *mLeft;
  KdTreeNode     *mRight;
};


class KdTree
{
public:
  KdTree(void)
  {
    mRoot = 0;
  }

  ~KdTree(void)
  {
    reset();
  }

  unsigned int search(const float *pos,float radius,unsigned int maxObjects,KdTreeFindNode *found) const
  {
    if ( !mRoot )	return 0;
    unsigned int count = 0;
    mRoot->search(X_AXIS,pos,radius,count,maxObjects,found);
    return count;
  }

  void reset(void)
  {
    mRoot = 0;
    for (unsigned int i=0; i<mObjects.size(); i++)
    {
      KdTreeNode *node = mObjects[i];
      delete node;
    }
    mObjects.clear();
  }

  unsigned int add(float x,float y,float z,float radius,void *userData)
  {
    unsigned int ret = mObjects.size();
    KdTreeNode *node = new KdTreeNode( x, y, z, radius, userData, ret );
    mObjects.push_back(node);
    if ( mRoot )
    {
      mRoot->add(node,X_AXIS);
    }
    else
    {
      mRoot = node;
    }
    return ret;
  }

  unsigned int getNearest(const float *pos,float radius,bool &_found) const // returns the nearest possible neighbor's index.
  {
    unsigned int ret = 0;

    _found = false;
    KdTreeFindNode found[1];
    unsigned int count = search(pos,radius,1,found);
    if ( count )
    {
      KdTreeNode *node = found[0].mNode;
      ret = node->getIndex();
      _found = true;
    }
    return ret;

  }

  KdTreeNode * getNode(unsigned int index)
  {
    KdTreeNode *ret = 0;
    if ( index < mObjects.size() )
    {
      ret = mObjects[index];
    }
    return ret;
  }

private:
  KdTreeNode      *mRoot;
  KdTreeNodeVector mObjects;
};

#endif

class VertexPool
{
public:
  VertexPool(float snap)
  {
    mSnap = snap;
    mSnapSquared = snap*snap;
  }

  ~VertexPool(void)
  {

  }

  bool sameVert(const float *p1,const float *p2) const
  {
    bool ret = false;
    float dx = p1[0]-p2[0];
    float dy = p1[1]-p2[1];
    float dz = p1[2]-p2[2];
    float m  = dx*dx+dy*dy+dz*dz;
    if ( m < mSnapSquared ) ret = true;
    return ret;
  }

  unsigned int getVertex(const float *p)
  {
    unsigned int ret=0xFFFFFFFF;

#if USE_KDTREE
    bool found;
    ret = mTree.getNearest(p,mSnap,found);
    if ( !found )
    {
      ret = mTree.add(p[0],p[1],p[2],0.0f,0);
      assert( ret == mVertices.size()/3 );
      mVertices.push_back(p[0]);
      mVertices.push_back(p[1]);
      mVertices.push_back(p[2]);
    }
#else
    unsigned int vcount = mVertices.size()/3;
    const float *vertices = &mVertices[0];
    for (unsigned int i=0; i<vcount; i++)
    {
      if ( sameVert(p,vertices) )
      {
        ret = i;
        break;
      }
      vertices+=3;
    }

    if ( ret == 0xFFFFFFFF )
    {
      ret = vcount;
      mVertices.push_back( p[0] );
      mVertices.push_back( p[1] );
      mVertices.push_back( p[2] );
    }
#endif

    return ret;
  }

  const float * getPos(unsigned int index) const
  {
    assert( index >= 0 && index < (mVertices.size()/3) );
    return &mVertices[index*3];
  }

  unsigned int getVcount(void) const
  {
    return mVertices.size()/3;
  }

private:
  float                           mSnap;
  float                           mSnapSquared;
  std::vector< float >            mVertices;
#if USE_KDTREE
  KdTree                          mTree;
#endif
};




//using namespace Vlookup;

VertexLookup Vl_createVertexLookup(float snap)
{
  VertexLookup ret = new VertexPool(snap);
  return ret;
}

void          Vl_releaseVertexLookup(VertexLookup vlook)
{
  VertexPool *vp = (VertexPool *) vlook;
  delete vp;
}

unsigned int  Vl_getIndex(VertexLookup vlook,const float *pos)  // get index.
{
  VertexPool *vp = (VertexPool *) vlook;
  return vp->getVertex(pos);
}

const float * Vl_getVertices(VertexLookup vlook)
{
  VertexPool *vp = (VertexPool *) vlook;
  return vp->getPos(0);
}


unsigned int  Vl_getVcount(VertexLookup vlook)
{
  VertexPool *vp = (VertexPool *) vlook;
  return vp->getVcount();
}


bool          Vl_saveAsObj(VertexLookup vlook,const char *fname,unsigned int tcount,unsigned int *indices) // helper function to save out an indexed triangle list as a raw wavefront OBJ mesh.
{
  bool ret = false;

  FILE *fph = fopen(fname,"wb");
  if ( fph )
  {
    ret = true;

    unsigned int vcount = Vl_getVcount(vlook);
    const float *v      = Vl_getVertices(vlook);

    for (unsigned int i=0; i<vcount; i++)
    {
      fprintf(fph,"v %0.9f %0.9f %0.9f\r\n", v[0], v[1], v[2] );
      v+=3;
    }

    for (unsigned int i=0; i<tcount; i++)
    {
      unsigned int i1 = *indices++;
      unsigned int i2 = *indices++;
      unsigned int i3 = *indices++;
      fprintf(fph,"f %d %d %d\r\n", i1+1, i2+1, i3+1 );
    }
    fclose(fph);
  }

  return ret;
}

