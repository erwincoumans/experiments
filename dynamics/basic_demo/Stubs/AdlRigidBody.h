
#ifndef ADL_RIGID_BODY_H
#define ADL_RIGID_BODY_H

#include "AdlQuaternion.h"

class RigidBodyBase
{
	public:

		_MEM_CLASSALIGN16
		struct Body
		{
			_MEM_ALIGNED_ALLOCATOR16;

			float4 m_pos;
			Quaternion m_quat;
			float4 m_linVel;
			float4 m_angVel;

			u32 m_shapeIdx;
			u32 m_shapeType;

			float m_invMass;
			float m_restituitionCoeff;
			float m_frictionCoeff;
			
		};

		struct Inertia
		{
/*			u16 m_shapeType;
			u16 m_shapeIdx;
			float m_restituitionCoeff;
			float m_frictionCoeff;
			int m_padding;
*/
			Matrix3x3 m_invInertia;
			Matrix3x3 m_initInvInertia;
		};
};

#endif// ADL_RIGID_BODY_H

