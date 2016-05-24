/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <assert.h>
#include "model.h"
#include <math.h>

#define WARP_SIZE 32
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)
#define HISTS_PER_WARP 16
#define NUM_HISTOGRAMS  (NUM_WARPS*HISTS_PER_WARP)
#define THREADS_PER_HIST (WARP_SIZE/HISTS_PER_WARP)

REAL** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

__constant__ REAL dev_binb[NUM_BINS+1];

unsigned int NUM_SETS;
unsigned int NUM_ELEMENTS;

class Context {
public:
    __device__ void start(gloop::DeviceLoop* loop);
    __device__ void finalize(gloop::DeviceLoop* loop);
    __device__ void iterateOverAllDataPoints(gloop::DeviceLoop* loop);
    __device__ void iterateOverAllRandomPoints(gloop::DeviceLoop* loop);


    struct cartesian* m_data;
    unsigned int (*m_warp_hists)[NUM_HISTOGRAMS];

    REAL* m_data_x;
    REAL* m_data_y;
    REAL* m_data_z;
    REAL* m_random_x;
    REAL* m_random_y;
    REAL* m_random_z;
    hist_t* m_histograms;

    unsigned int m_NUM_ELEMENTS;
    int m_i;
    int m_j;

    int m_do_self;
};

// create the bin boundaries
void initBinB( struct pb_TimerSet *timers )
{
    REAL *binb = (REAL*)malloc((NUM_BINS+1)*sizeof(REAL));
    for (int k = 0; k < NUM_BINS+1; k++) {
        binb[k] = cos(pow(10.0, (log10(min_arcmin) + k*1.0/bins_per_dec)) / 60.0*D2R);
    }
    pb_SwitchToTimer( timers, pb_TimerID_COPY );
    cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS+1)*sizeof(REAL));
    pb_SwitchToTimer( timers, pb_TimerID_COMPUTE );
    free(binb);
}

__device__ void Context::iterateOverAllRandomPoints(gloop::DeviceLoop* loop)
{
    // Iterate over all random points
    unsigned int NUM_ELEMENTS = m_NUM_ELEMENTS;
    if (m_j < NUM_ELEMENTS) {
        {
            struct cartesian* data = m_data;
            int do_self = m_do_self;
            unsigned int (*warp_hists)[NUM_HISTOGRAMS] = m_warp_hists;
            int i = m_i;
            int j = m_j;

            // load current random point values
            REAL random_x_s;
            REAL random_y_s;
            REAL random_z_s;

            {
                if(threadIdx.x + j < NUM_ELEMENTS) {
                    random_x_s = m_random_x[threadIdx.x + j];
                    random_y_s = m_random_y[threadIdx.x + j];
                    random_z_s = m_random_z[threadIdx.x + j];
                }
            }

            unsigned int warpnum = threadIdx.x / (WARP_SIZE/HISTS_PER_WARP);

            // Iterate for all elements of current set of data points
            // (BLOCK_SIZE iterations per thread)
            // Each thread calcs against 1 random point within cur set of random
            // (so BLOCK_SIZE threads covers all random points within cur set)
            for(unsigned int k = 0; (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS); k += 1) {
                // do actual calculations on the values:
                REAL distance =
                    data[k].x * random_x_s +
                    data[k].y * random_y_s +
                    data[k].z * random_z_s;

                unsigned int bin_index;

                // run binary search to find bin_index
                unsigned int min = 0;
                unsigned int max = NUM_BINS;
                {
                    unsigned int k2;

                    while (max > min+1) {
                        k2 = (min + max) / 2;
                        if (distance >= dev_binb[k2])
                            max = k2;
                        else
                            min = k2;
                    }
                    bin_index = max - 1;
                }

                if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && (!do_self || (threadIdx.x + j > i + k)) && (threadIdx.x + j < NUM_ELEMENTS)) {
                    atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                }
            }
        }
        gloop::loop::postTask(loop, [this] (gloop::DeviceLoop* loop) {
            BEGIN_SINGLE_THREAD
            {
                m_j += BLOCK_SIZE;
            }
            END_SINGLE_THREAD
            iterateOverAllRandomPoints(loop);
        });
        return;
    }
    gloop::loop::postTask(loop, [this] (gloop::DeviceLoop* loop) {
        BEGIN_SINGLE_THREAD
        {
            m_i += BLOCK_SIZE;
        }
        END_SINGLE_THREAD
        iterateOverAllDataPoints(loop);
    });
}

__device__ void Context::iterateOverAllDataPoints(gloop::DeviceLoop* loop)
{
    // Iterate over all data points
    unsigned int NUM_ELEMENTS = m_NUM_ELEMENTS;
    int i = m_i;
    if (i < NUM_ELEMENTS) {
        // load current set of data into shared memory
        // (total of BLOCK_SIZE points loaded)
        if(threadIdx.x + i < NUM_ELEMENTS) {
            // reading outside of bounds is a-okay
            m_data[threadIdx.x] = (struct cartesian) {m_data_x[threadIdx.x + i], m_data_y[threadIdx.x + i], m_data_z[threadIdx.x + i]};
        }

        __syncthreads();

        BEGIN_SINGLE_THREAD
        {
            m_j = (m_do_self ? i+1 : 0);
        }
        END_SINGLE_THREAD
        gloop::loop::postTask(loop, [this] (gloop::DeviceLoop* loop) {
            iterateOverAllRandomPoints(loop);
        });
        return;
    }
    finalize(loop);
}

__device__ void Context::finalize(gloop::DeviceLoop* loop)
{
    gloop::loop::postTask(loop, [this] (gloop::DeviceLoop* loop) {
        // coalesce the histograms in a block
        unsigned int warp_index = threadIdx.x & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = threadIdx.x / (NUM_HISTOGRAMS>>1);
        unsigned int (*warp_hists)[NUM_HISTOGRAMS] = m_warp_hists;
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; offset >>= 1) {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1)) {
                __syncthreads();
                if(warp_index < offset && bin_base+bin_index < NUM_BINS ) {
                    unsigned long sum =
                        warp_hists[bin_base + bin_index][warp_index] +
                        warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        __syncthreads();

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = m_histograms + NUM_BINS * gloop::logicalBlockIdx.x;
        if(threadIdx.x < NUM_BINS) {
            hist_base[threadIdx.x] = warp_hists[threadIdx.x][0];
        }

        BEGIN_SINGLE_THREAD
        {
            delete [] m_data;
            delete [] m_warp_hists;
            delete this;
        }
        END_SINGLE_THREAD
    });
}

__device__ void Context::start(gloop::DeviceLoop* loop)
{
    gloop::loop::postTask(loop, [this] (gloop::DeviceLoop* loop) {
        iterateOverAllDataPoints(loop);
    });
}

__device__ void gen_hists(gloop::DeviceLoop* loop, hist_t* histograms, REAL* all_x_data, REAL* all_y_data, REAL* all_z_data, int NUM_SETS, int NUM_ELEMENTS)
{
    unsigned int bx = gloop::logicalBlockIdx.x;
    unsigned int tid = threadIdx.x;
    bool do_self = (bx < (NUM_SETS + 1));

    __shared__ Context* ctx;
    __shared__ unsigned int (*warp_hists)[NUM_HISTOGRAMS];
    BEGIN_SINGLE_THREAD
    {
        ctx = new Context();
        ctx->m_data = new struct cartesian[BLOCK_SIZE];
        unsigned int (*wh)[NUM_HISTOGRAMS] = new unsigned int[NUM_BINS][NUM_HISTOGRAMS]; // 640B <1k
        warp_hists = wh;
        ctx->m_warp_hists = wh;

        if(!do_self) {
            ctx->m_data_x = all_x_data;
            ctx->m_data_y = all_y_data;
            ctx->m_data_z = all_z_data;
            ctx->m_random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            ctx->m_random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            ctx->m_random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        } else {
            ctx->m_data_x = ctx->m_random_x = all_x_data + NUM_ELEMENTS * (bx);
            ctx->m_data_y = ctx->m_random_y = all_y_data + NUM_ELEMENTS * (bx);
            ctx->m_data_z = ctx->m_random_z = all_z_data + NUM_ELEMENTS * (bx);
        }
        ctx->m_histograms = histograms;
        ctx->m_do_self = do_self;
        ctx->m_NUM_ELEMENTS = NUM_ELEMENTS;
        ctx->m_i = 0;
        ctx->m_j = 0;
    }
    END_SINGLE_THREAD

    for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE) {
        if(w+tid < NUM_BINS*NUM_HISTOGRAMS) {
            warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
        }
    }

    ctx->start(loop);
}

// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
