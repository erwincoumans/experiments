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

#define NUM_THREADS 5
#define MAX_NESTED_BARRIER_LEVELS 4

struct Args
{
    int m_i[NUM_THREADS];
    int m_barrierNestingLevel[NUM_THREADS];
    int m_barrier[MAX_NESTED_BARRIER_LEVELS][NUM_THREADS];
    int m_savedBarrier[MAX_NESTED_BARRIER_LEVELS][NUM_THREADS];

    Args()
    {
        for (int i=0;i<NUM_THREADS;i++)
        {
            m_i[i]=0;
            m_barrierNestingLevel[i] = 0;
            for (int n=0;n<MAX_NESTED_BARRIER_LEVELS;n++)
            {
                m_barrier[n][i]=0;
                m_savedBarrier[n][i]=0;

            }
        }
    }
};

#define BARRIER(label)     args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId]++; \
                    return 1; \
label:\

#define SKIP_TO_BARRIER \
switch (args->m_barrierNestingLevel[threadId])                                      \
{                                                                                   \
case 0:                                                                             \
    {                                                                               \
        switch (args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId])   \
        {                                                                           \
            case 0:                                                                 \
                break;                                                              \
            case 1:                                                                 \
                goto label01;                                                       \
            case 2:                                                                 \
                goto label02;                                                       \
            case 3:                                                                 \
                goto label03;                                                       \
            default:                                                                \
                printf("error, unknown barrier\n");                                 \
        };                                                                          \
        break;                                                                      \
    }                                                                               \
case 1:                                                                             \
    {                                                                               \
        switch (args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId])   \
        {                                                                           \
            case 0:                                                                 \
                assert(0);                                                          \
                break;                                                              \
            case 1:                                                                 \
                goto label11;                                                       \
            case 2:                                                                 \
                goto label12;                                                       \
            default:                                                                \
                printf("error, unknown barrier\n");                                 \
        };                                                                          \
        break;                                                                      \
    }                                                                               \
default:                                                                            \
    {                                                                               \
    }                                                                               \
};


#define INC_LEVEL \
args->m_barrierNestingLevel[threadId]++;\
args->m_savedBarrier[args->m_barrierNestingLevel[threadId]][threadId] = args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId];

#define DEC_LEVEL \
args->m_savedBarrier[args->m_barrierNestingLevel[threadId]][threadId] = args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId]; \
args->m_barrierNestingLevel[threadId]--;


#define I (args->m_i[threadId])

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

