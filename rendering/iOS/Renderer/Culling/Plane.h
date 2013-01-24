/*
 Oolong Engine for the iPhone / iPod touch
 Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/
 
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose, 
 including commercial applications, and to alter it and redistribute it freely, 
 subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */
#ifndef PLANE_H_
#define PLANE_H_

#include "Mathematics.h"

// 
// Plane class
//
class CPlane
{
public:
	CPlane(){}
	~CPlane(){}
	
	CPlane(VERTTYPE x, VERTTYPE y, VERTTYPE z, VERTTYPE w)
	{
		Plane.x = x;
		Plane.y = y;
		Plane.z = z;
		Plane.w = w;
	}
/*
	CPlane(const VECTOR3& p, const VECTOR3& n)
	{
		ComputePlaneFromPointNormal(p, n);
	}
	
	void Set(const VECTOR3& p, const VECTOR3& n)
	{
		ComputePlaneFromPointNormal(p, n);
	}
*/	
	void Set(const VERTTYPE x, const VERTTYPE y, const VERTTYPE z, const VERTTYPE w)
	{
		Plane.x = x;
		Plane.y = y;
		Plane.z = z;
		Plane.w = w;
	}
	
	void Set(const VECTOR4& v)
	{
		Plane.x = v.x;
		Plane.y = v.y;
		Plane.z = v.z;
		Plane.w = v.w;
	}
		
	// gets the point on the plane nearest to the origin
	VECTOR3 GetPos() const
	{
		VECTOR3 v;
		v.x = Plane.x * Plane.w;
		v.y = Plane.y * Plane.w;
		v.z = Plane.z * Plane.w;
		
		return v;
	}
	
	VECTOR3 GetNormal() const
	{			
		VECTOR3 v;
		v.x = Plane.x;
		v.y = Plane.y;
		v.z = Plane.z;
		
		return v;
	}
	
	VECTOR4& GetPlane()
	{			
		return Plane;
	}

/* Probably wrong and not ready for fixed-point arithemtic
	// Given three points on a plane, calculate a plane
	// a, b, c are the distance on the plane
	//
	// The normal of the plane is stored in the x, y, z channel and the distance from teh origin is stored in the w channel
	// The points must be in clockwise order, or else the negative of the desired normal plane will be calculated
	inline void ComputePlane(const VECTOR3& a, const VECTOR3& b, const VECTOR3& c);
	
	// Given one point on a plane and the plane normal, calculate a plane
	inline void ComputePlaneFromPointNormal(const VECTOR3& point, const VECTOR3& normal);
*/	
	// calculate the distance between a point and a plane
	inline VERTTYPE DistanceToPlane(const VECTOR3& point);
	
	// normalize plane
	inline void NormalizePlane();
		
private:
	VECTOR4 Plane;
};

#endif // PLANE_H_