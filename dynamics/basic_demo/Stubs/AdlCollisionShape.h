#ifndef COLLISION_SHAPE_H
#define COLLISION_SHAPE_H

#include "Stubs/AdlMath.h"
#include "Stubs/AdlAabb.h"


_MEM_CLASSALIGN16
class CollisionShape
{
	public:
		_MEM_ALIGNED_ALLOCATOR16;

		enum Type
		{
			SHAPE_HEIGHT_FIELD,
			SHAPE_CONVEX_HEIGHT_FIELD,
			SHAPE_PLANE,
			MAX_NUM_SHAPE_TYPES,
		};

		CollisionShape( Type type, float collisionMargin = 0.0025f ) : m_type( type ){ m_collisionMargin = collisionMargin; }
		virtual ~CollisionShape(){}
		virtual float queryDistance(const float4& p) const = 0;
		virtual bool queryDistanceWithNormal(const float4& p, float4& normalOut) const = 0;

	public:
		Type m_type;
		Aabb m_aabb;
		float m_collisionMargin;
};

#endif
