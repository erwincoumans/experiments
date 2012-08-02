//
//  main.cpp
//  bariertest
//
//  Created by Erwin Coumans on 7/20/12.
//  Copyright (c) 2012 Erwin Coumans. All rights reserved.
//


//it is hard to debug code with barriers in OpenCL, so here is some attempt to 'emulate' barriers in plain C
//the kernels can be run in a workgroup of size 'NUM_THREADS'

#include <iostream>
#include <assert.h>

#include "barrier_support.h"


int func(Args* args, int threadId)
{
  
    SKIP_TO_BARRIER;
    
    printf("start of thread %d\n",threadId);
    

    BARRIER(label01);
    
    printf("label1 for thread %d\n", threadId);

    INC_LEVEL;
    printf("thread %d entered level %d\n",threadId,args->m_barrierNestingLevel[threadId]);

    for (I=0;I<10;I++)
    {

        args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId] = args->m_savedBarrier[args->m_barrierNestingLevel[threadId]][threadId];
        
        BARRIER(label11);

        printf("label11 for thread %d at loop iteration %d\n", threadId,I);

        BARRIER(label12);

        printf("label12 for thread %d at loop iteration %d\n", threadId,I);

        
    }
    
    printf("thread %d left level %d\n",threadId,args->m_barrierNestingLevel[threadId]);
    DEC_LEVEL;
    
    

    BARRIER(label02);
    
    printf("label2 for thread %d\n", threadId);

    BARRIER(label03);

    printf("label3 for thread %d\n", threadId);

    
    return 0;
}

int main(int argc, const char * argv[])
{
    Args args;
    
    
    int retval = 1;

    while(retval)
    {
        for (int i=0;i<NUM_THREADS;i++)
        {
            retval = func(&args,i);
        }
    }

    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

