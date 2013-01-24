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
#ifndef AABB_H_
#define AABB_H_

#include "Mathematics.h"

#include "Sphere.h"


// 
// simple AABB class
//
class CAABB
{
public:

	//
	// AABB constructors
	//
	//AABB() : m_Min(0.0f, 0.0f, 0.0f), m_Max(0.0f, 0.0f, 0.0f), m_BoundingRadius(0.0f){}
	CAABB(){ m_Min.x = f2vt(0.0f);m_Min.y = f2vt(0.0f);m_Min.z = f2vt(0.0f); m_Max.x = f2vt(0.0f);m_Max.y = f2vt(0.0f);m_Max.z = f2vt(0.0f); m_BoundingRadius = f2vt(0.0f);}
	~CAABB(){}
	
	const CAABB &operator=(const CAABB &b)
	{
		m_Min = b.m_Min;
		m_Max = b.m_Max;
		m_BoundingRadius = b.m_BoundingRadius;
		
		return *this;
	}
	
	inline void ComputeAABB(const VECTOR3	* const pV, const int nNumberOfVertices);
	inline void ComputeAABB(const float	* pV, const int stride, const int nNumberOfVertices);
	
	// set min / max and bounding radius
	inline void Set(const VECTOR3 &min, const VECTOR3 &max);
	inline void Set(const VECTOR3 &min, const VECTOR3 &max, float BoundingRadius);
	inline VECTOR3 GetCenter(void) const;
	inline VERTTYPE GetBoundingRadius(void) const;
	inline void Transform(const MATRIX &matrix);

	inline void Set(const VECTOR4 &sphere) const;
	inline bool Contains(VECTOR3 vector) const;
//	bool Contains(CSphere &sphere) const;
//	bool Intersects(const CAABB &box) const;
	
	inline const VECTOR3 &GetMin(void) const;
	inline const VECTOR3 &GetMax(void) const;
	inline VERTTYPE GetWidth(void) const;
	inline VERTTYPE GetHeight(void) const;
	inline VERTTYPE GetDepth(void) const;
	
	// apply a scale to the box
	inline void Scale(float scale);
	
	inline bool IsVisible( const MATRIX	* const pMatrix,
								 bool			* const pNeedsZClipping);
	
	VECTOR3 m_Min, m_Max;
	float m_BoundingRadius;
};

#endif // AABB_H_