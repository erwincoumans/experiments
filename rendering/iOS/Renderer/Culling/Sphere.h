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
#ifndef SPHERE_H_
#define SPHERE_H_

//#define FIXEDPOINTENABLE

#include "Mathematics.h"

#include "AABB.h"

class CAABB;

// 
// simple sphere class
//
class CSphere
{
public:

	//
	// Sphere constructors
	//
	//CSphere() : m_PositionAndRadius.x(f2vt(0.0f)), m_PositionAndRadius.y(f2vt(0.0f)), m_PositionAndRadius.z(f2vt(0.0f)), m_PositionAndRadius.w(f2vt(0.0f)){}
	//CSphere(const VECTOR3 &center, VERTTYPE radius) : m_PositionAndRadius(center.x, center.y, center.z, radius){} 
	CSphere(){m_PositionAndRadius.x = f2vt(0.0f); m_PositionAndRadius.y = f2vt(0.0f);m_PositionAndRadius.z = f2vt(0.0f);m_PositionAndRadius.w = f2vt(0.0f);  };
	
	// Sphere destructor
	~CSphere() {}
		
	// create sphere from AABB
	inline void CreateSphereFromAABB(CAABB& AABB);
	
	// Accessors
	VECTOR3 GetCenter(void) const;
	VERTTYPE GetRadius(void) const;
	VECTOR4 GetVector4() const;
	VERTTYPE GetRadius2(void) const;
	
	void Set(const VECTOR3 &center, VERTTYPE radius);
	void Set(const VECTOR4 &positionandradius);
	void SetCenter(const VECTOR3 &center);
	void SetRadius(VERTTYPE radius);
	
	bool Contains(VECTOR3 point) const;

	// sphere data
	VECTOR4 m_PositionAndRadius;
};


#endif // SPHERE_H_