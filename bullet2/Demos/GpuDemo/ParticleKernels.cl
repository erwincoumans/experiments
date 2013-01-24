MSTRINGIFY(


__kernel void  updatePositionsKernel( __global float4* linearVelocities, __global float4* positions,const int numNodes)
{
	int nodeID = get_global_id(0);
	float timeStep = 0.0166666;
	
	float BT_GPU_ANGULAR_MOTION_THRESHOLD = (0.25f * 3.14159254);
	
	if( nodeID < numNodes )
	{
		positions[nodeID] += linearVelocities[nodeID]*timeStep;
	}
}

);