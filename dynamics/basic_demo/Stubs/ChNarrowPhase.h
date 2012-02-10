#pragma once

#include <Adl/Adl.h>
//#include <Common/Base/SyncObjects.h>

#include "AdlMath.h"
#include "AdlContact4.h"
#include "AdlRigidBody.h"

#include "../ConvexHeightFieldShape.h"

//#include "TypeDefinition.h"
//#include "RigidBody.h"
//#include "ConvexHeightFieldShape.h"

namespace adl
{
class ShapeBase;

class ChNarrowphaseBase
{
	public:
		struct Config
		{
			float m_collisionMargin;
		};
/*
		typedef struct
		{
			//	m_normal.w == height in u8
			float4 m_normal[HEIGHT_RES*HEIGHT_RES*6];
			u32 m_height4[HEIGHT_RES*HEIGHT_RES*6];

			float m_scale;
			float m_padding0;
			float m_padding1;
			float m_padding2;
		} ShapeData;
*/
};

template<DeviceType TYPE>
class ChNarrowphase : public ChNarrowphaseBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct Data
		{
			const Device* m_device;
			Kernel* m_supportCullingKernel;
			Kernel* m_narrowphaseKernel;
			Kernel* m_narrowphaseWithPlaneKernel;

			Buffer<u32>* m_counterBuffer;
		};

		enum
		{
			N_TASKS = 4,
			HEIGHT_RES = ConvexHeightField::HEIGHT_RES,
		};

		struct ShapeData
		{
			float4 m_normal[HEIGHT_RES*HEIGHT_RES*6];
			u32 m_height4[HEIGHT_RES*HEIGHT_RES*6];
			u32 m_supportHeight4[HEIGHT_RES*HEIGHT_RES*6];

			float m_scale;
			float m_padding0;
			float m_padding1;
			float m_padding2;
		};

		struct ConstData
		{
			int m_nPairs;
			float m_collisionMargin;
			int m_capacity;
			int m_paddings[1];
		};
		
		static
		Data* allocate( const Device* device );

		static
		void deallocate( Data* data );
/*
		static
		Buffer<ShapeData>* allocateShapeBuffer( const Device* device, int capacity );

		static
		void deallocateShapeBuffer( Buffer<ShapeData>* shapeBuf );

		static
		void setShape( Buffer<ShapeData>* shapeBuf, ShapeBase* shape, int idx, float collisionMargin );
*/
		static
		ShapeDataType allocateShapeBuffer( const Device* device, int capacity );

		static
		void deallocateShapeBuffer( ShapeDataType shapeBuf );

		static
		void setShape( ShapeDataType shapeBuf, ShapeBase* shape, int idx, float collisionMargin = 0.f );
		
		static
		void setShape( ShapeDataType shapeBuf, ConvexHeightField* cvxShape, int idx, float collisionMargin = 0.f );

		// Run NarrowphaseKernel
		//template<bool USE_OMP>
		static
		void execute( Data* data, const Buffer<int2>* pairs, int nPairs, 
			const Buffer<RigidBodyBase::Body>* bodyBuf, const ShapeDataType shapeBuf,
			Buffer<Contact4>* contactOut, int& nContacts, const Config& cfg );

		// Run NarrowphaseWithPlaneKernel
		//template<bool USE_OMP>
		static
		void execute( Data* data, const Buffer<int2>* pairs, int nPairs, 
			const Buffer<RigidBodyBase::Body>* bodyBuf, const ShapeDataType shapeBuf,
			const Buffer<float4>* vtxBuf, const Buffer<int4>* idxBuf,
			Buffer<Contact4>* contactOut, int& nContacts, const Config& cfg );

		// Run SupportCullingKernel
		//template<bool USE_OMP>
		static
		int culling( Data* data, const Buffer<int2>* pairs, int nPairs, const Buffer<RigidBodyBase::Body>* bodyBuf,
			const ShapeDataType shapeBuf, const Buffer<int2>* pairsOut, const Config& cfg );
};

//#include <AdlPhysics/Narrowphase/ChNarrowphase.inl>
//#include <AdlPhysics/Narrowphase/ChNarrowphaseHost.inl>

#include "ChNarrowphase.inl"

};
