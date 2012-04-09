
#ifndef BTC_FIND_PAIRS_H
#define BTC_FIND_PAIRS_H


#include "btbPlatformDefinitions.h"

#ifdef __cplusplus
extern "C" {
#endif//__cplusplus

BTB_DECLARE_HANDLE(btcAabbSpace);
BTB_DECLARE_HANDLE(btcAabbProxy);

extern btcAabbSpace plCreateBruteforceSpace(int maxNumAabbs, int maxNumPairs);
extern void	plDestroySpace(btcAabbSpace bp);
extern 	btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ);
extern void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle);
extern void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ);
extern int	btcFindPairs(btcAabbSpace bp);
extern btbBuffer btcGetPairBuffer(btcAabbSpace bp);
extern void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes);
extern void btcUnmapBuffer(btcAabbSpace aabbSpace);


#ifdef __cplusplus
}

class btbAabbSpaceInterface
{
public:

	virtual ~btbAabbSpaceInterface(){}
	virtual btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)=0;
	virtual void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle)=0;
	virtual void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)=0;
	virtual int	btcFindPairs(btcAabbSpace bp)=0;
	virtual btbBuffer btcGetPairBuffer(btcAabbSpace bp)=0;
	virtual void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes)=0;
	virtual void btcUnmapBuffer(btcAabbSpace aabbSpace)=0;
};

#endif//__cplusplus


#endif //BTC_FIND_PAIRS_H
