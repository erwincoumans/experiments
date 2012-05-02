
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
bool TestAabbAgainstAabb2(__local const btAabbCL* aabb1, __global const btAabbCL* aabb2)
{
	bool overlap = true;
	overlap = (aabb1->m_min.x > aabb2->m_max.x || aabb1->m_max.x < aabb2->m_min.x) ? false : overlap;
	overlap = (aabb1->m_min.z > aabb2->m_max.z || aabb1->m_max.z < aabb2->m_min.z) ? false : overlap;
	overlap = (aabb1->m_min.y > aabb2->m_max.y || aabb1->m_max.y < aabb2->m_min.y) ? false : overlap;
	return overlap;
}


///aabbs are sorted on the 'axis' coordinate

__kernel void   computePairs( __global const btAabbCL* aabbs, volatile __global int2* pairsOut,volatile  __global int* pairCount, int numObjects, int axis, int maxPairs)
{
				
	int groupId = get_group_id(0);
	int localId = get_local_id(0);
	__local btAabbCL localAabbs[128];// = aabbs[i];
	
	
	int2 myPairs[128];// = aabbs[i];
	
	int i = get_global_id(0);
	if (i>=numObjects)
		return;
	
	
	int curMyPairs=0;
	
	
	
	localAabbs[localId] = aabbs[i];
	
	for (int j=i+1;j<numObjects;j++)
	{
	  
		if(localAabbs[localId].m_maxElems[axis] < aabbs[j].m_minElems[axis]) 
		{
			break;
		}

		if (TestAabbAgainstAabb2(&localAabbs[localId],&aabbs[j]))
		{
			myPairs[curMyPairs].x = aabbs[i].m_minIndices[3];
			myPairs[curMyPairs].y = aabbs[j].m_minIndices[3];
			
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