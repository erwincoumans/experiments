
#define USE_LOCAL_MEMORY
typedef struct 
{
	union
	{
		float4	m_min;
		float   m_minElems[4];
		int			m_minIndices[4];
	};
	union
	{
		float4	m_max;
		float   m_maxElems[4];
		int			m_maxIndices[4];
	};
} btAabbCL;


/// conservative test for overlap between two aabbs
bool TestAabbAgainstAabb2(const btAabbCL* aabb1, __local const btAabbCL* aabb2)
{
	bool overlap = true;
	overlap = (aabb1->m_min.x > aabb2->m_max.x || aabb1->m_max.x < aabb2->m_min.x) ? false : overlap;
	overlap = (aabb1->m_min.z > aabb2->m_max.z || aabb1->m_max.z < aabb2->m_min.z) ? false : overlap;
	overlap = (aabb1->m_min.y > aabb2->m_max.y || aabb1->m_max.y < aabb2->m_min.y) ? false : overlap;
	return overlap;
}

bool TestAabbAgainstAabb2Global(const btAabbCL* aabb1, __global const btAabbCL* aabb2)
{
	bool overlap = true;
	overlap = (aabb1->m_min.x > aabb2->m_max.x || aabb1->m_max.x < aabb2->m_min.x) ? false : overlap;
	overlap = (aabb1->m_min.z > aabb2->m_max.z || aabb1->m_max.z < aabb2->m_min.z) ? false : overlap;
	overlap = (aabb1->m_min.y > aabb2->m_max.y || aabb1->m_max.y < aabb2->m_min.y) ? false : overlap;
	return overlap;
}


///aabbs are sorted on the 'axis' coordinate

__kernel void   computePairsKernel( __global const btAabbCL* aabbs, volatile __global int2* pairsOut,volatile  __global int* pairCount, int numObjects, int axis, int maxPairs)
{
				
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	int i = get_global_id(0);
	
	__local btAabbCL localAabbs[128];// = aabbs[i];
#ifdef USE_LOCAL_MEMORY
	__local numActiveWgItems[1];
	__local breakRequest[1];
#endif //USE_LOCAL_MEMORY
	
	int2 myPairs[128];// = aabbs[i];
		btAabbCL myAabb;
	
	if (i>=numObjects)
		return;
	
#ifdef USE_LOCAL_MEMORY
	if (localId==0)
	{
		numActiveWgItems[0] = 0;
		breakRequest[0] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(numActiveWgItems);
	barrier(CLK_LOCAL_MEM_FENCE);

#endif//USE_LOCAL_MEMORY	
	
	int curMyPairs=0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	myAabb = aabbs[i];
	
	const float testVal = myAabb.m_maxElems[axis];
	int localCount=0;
	int block=0;
	int localBreak = 0;
	
#ifdef USE_LOCAL_MEMORY
	localAabbs[localId] = aabbs[i+block];
	localAabbs[localId+64] = aabbs[i+block+64];
	barrier(CLK_LOCAL_MEM_FENCE);
#endif//USE_LOCAL_MEMORY
	int2 prevPair;
 	prevPair.x=-1;
 	prevPair.y=-1;
 	
	for (;(i+1+localCount+block)<numObjects;)
	{
	
		if (!localBreak)
		{
				  
#ifdef USE_LOCAL_MEMORY
			if(testVal < (localAabbs[localCount+localId+1].m_minElems[axis])) 
#else//USE_LOCAL_MEMORY
	  	if(testVal < (aabbs[i+1+localCount+block].m_minElems[axis])) 
#endif//USE_LOCAL_MEMORY
			{
#ifdef USE_LOCAL_MEMORY	
				localBreak = 1;
				atomic_inc(breakRequest);
#else
				break;
#endif
			}

#ifdef USE_LOCAL_MEMORY
			if (TestAabbAgainstAabb2(&myAabb,&localAabbs[localCount+localId+1]))
#else	//USE_LOCAL_MEMORY
			if (TestAabbAgainstAabb2Global(&myAabb,&aabbs[i+1+localCount+block]))
#endif //USE_LOCAL_MEMORY
			{
			
				
				myPairs[curMyPairs].x = myAabb.m_minIndices[3];
			
#ifdef USE_LOCAL_MEMORY
				myPairs[curMyPairs].y = localAabbs[localCount+localId+1].m_minIndices[3];
#else //USE_LOCAL_MEMORY			
				myPairs[curMyPairs].y = aabbs[i+1+localCount+block].m_minIndices[3];
#endif//USE_LOCAL_MEMORY			
				
				if (1)//!((myPairs[curMyPairs].x == prevPair.x) && (myPairs[curMyPairs].y == prevPair.y)))
				{
					prevPair = myPairs[curMyPairs];
					curMyPairs++;
					
				
					//flush to main memory
					if (curMyPairs==64)
					{
						int curPair = atomic_add (pairCount,curMyPairs);
						for (int p=0;p<curMyPairs;p++)
						{
							pairsOut[curPair+p] = myPairs[p];
						}
						curMyPairs=0;
					}
				}
			}
		}
		
		localCount++;
		if (localCount==64)
		{
			localCount = 0;
			block+=64;			
#ifdef USE_LOCAL_MEMORY
			if ((i+block)<numObjects)
				localAabbs[localId] = aabbs[i+block];
			if ((i+64+block)<numObjects)
				localAabbs[localId+64] = aabbs[i+block+64];			
#endif //USE_LOCAL_MEMORY			

		
#ifdef USE_LOCAL_MEMORY
			if (breakRequest[0]==numActiveWgItems[0])
				break;
#endif
		}  
	barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	 //flush remainder to main memory
	if (curMyPairs>0)
	 {
	 	 int curPair = atomic_add (pairCount,curMyPairs);
	 		for (int p=0;p<curMyPairs;p++)
	 		{
				pairsOut[curPair+p] = myPairs[p];
			}
	 }
}

//http://stereopsis.com/radix.html
unsigned int FloatFlip(float fl)
{
	unsigned int f = *(unsigned int*)&fl;
	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
}

float IFloatFlip(unsigned int f)
{
	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	unsigned int fl = f ^ mask;
	return *(float*)&fl;
}

__kernel void   flipFloatKernel( __global const btAabbCL* aabbs, volatile __global int2* sortData, int numObjects, int axis)
{
	int i = get_global_id(0);
	if (i>=numObjects)
		return;
		
		sortData[i].x = FloatFlip(aabbs[i].m_minElems[axis]);
		sortData[i].y = i;
		
}


__kernel void   scatterKernel( __global const btAabbCL* aabbs, volatile __global const int2* sortData, __global btAabbCL* sortedAabbs, int numObjects)
{
	int i = get_global_id(0);
	if (i>=numObjects)
		return;

		sortedAabbs[i] = aabbs[sortData[i].y];
}