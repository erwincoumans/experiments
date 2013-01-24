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
#ifndef SPHERE_INLINE_H_
#define SPHERE_INLINE_H_

#include "Sphere.h"

// create a sphere from a AABB
inline void CSphere::CreateSphereFromAABB(CAABB& AABB)
{
	VECTOR3 vRadius;
		
	const VECTOR3& Min = AABB.GetMin();
	const VECTOR3& Max = AABB.GetMax();
		
	VECTOR3 m_vCenter;
	m_vCenter.x = (Max.x + Min.x) / 2;
	m_vCenter.y = (Max.y + Min.y) / 2;
	m_vCenter.z = (Max.z + Min.z) / 2;
		
	vRadius.x = Max.x - m_vCenter.x;
	vRadius.y = Max.y - m_vCenter.y;
	vRadius.z = Max.z - m_vCenter.z;
	m_PositionAndRadius.x = m_vCenter.x;
	m_PositionAndRadius.y = m_vCenter.y;
	m_PositionAndRadius.z = m_vCenter.z;
				
	m_PositionAndRadius.w = MatrixVec3Length(vRadius); 
}


inline VECTOR3 CSphere::GetCenter(void) const
{
	VECTOR3 ret;
	
	// some day I create a vector class with nice accessors :-0
	//m_PositionAndRadius.GetVector3(ret);
	ret.x = m_PositionAndRadius.x;
	ret.y = m_PositionAndRadius.y;
	ret.z = m_PositionAndRadius.z;
	
	return ret;
}

inline VERTTYPE CSphere::GetRadius(void) const
{
	return m_PositionAndRadius.w;
}

inline VERTTYPE CSphere::GetRadius2(void) const
{
	return m_PositionAndRadius.w * m_PositionAndRadius.w;
}

inline VECTOR4 CSphere::GetVector4() const
{
	return m_PositionAndRadius;
}

inline void CSphere::Set(const VECTOR3 &center, VERTTYPE radius)
{
	m_PositionAndRadius.x = center.x;
	m_PositionAndRadius.y = center.y;
	m_PositionAndRadius.z = center.z;
	m_PositionAndRadius.w = radius;
}

inline void CSphere::Set(const VECTOR4 &positionandradius)
{
	m_PositionAndRadius.x = positionandradius.x;
	m_PositionAndRadius.y = positionandradius.y;
	m_PositionAndRadius.z = positionandradius.z;
	m_PositionAndRadius.w = positionandradius.w;
}

inline void CSphere::SetCenter(const VECTOR3 &center)
{
	m_PositionAndRadius.x = center.x;
	m_PositionAndRadius.y = center.y;
	m_PositionAndRadius.z = center.z;
	m_PositionAndRadius.w = m_PositionAndRadius.w;
}

inline void CSphere::SetRadius(VERTTYPE radius)
{
	m_PositionAndRadius.w = radius;
}

inline bool CSphere::Contains(VECTOR3 point) const
{
	VECTOR3 Center = GetCenter();
	
	VECTOR3 temp;

	temp.x = Center.x - point.x;
	temp.y = Center.y - point.y;
	temp.z = Center.z - point.z;
	
	float Dist2 = MatrixVec3DotProduct(temp, temp);
	
//	float Dist2 = rx * rx + ry * ry + rz * rz;

/*
	// vectorized version
	VECTOR3 r = Center - point;
	
	VERTTYPE Dist2 = dot(r, r);
*/	
	return Dist2 <= GetRadius2();
}

#endif 