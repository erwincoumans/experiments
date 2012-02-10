#ifndef ADL_CONSTRAINT4_H
#define ADL_CONSTRAINT4_H



struct Constraint4
		{
			_MEM_ALIGNED_ALLOCATOR16;

			float4 m_linear;
			float4 m_worldPos[4];
			float4 m_center;	//	friction
			float m_jacCoeffInv[4];
			float m_b[4];
			float m_appliedRambdaDt[4];

			float m_fJacCoeffInv[2];	//	friction
			float m_fAppliedRambdaDt[2];	//	friction

			u32 m_bodyA;
			u32 m_bodyB;

			u32 m_batchIdx;
			u32 m_paddings[1];

			__inline
			void setFrictionCoeff(float value) { m_linear.w = value; }
			__inline
			float getFrictionCoeff() const { return m_linear.w; }
		};

#endif //ADL_CONSTRAINT4_H
		