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

template<typename Callback>
__device__ void iterateOverAllRandomPoints(gloop::DeviceLoop* loop, struct cartesian* data, bool do_self, REAL* random_x, REAL* random_y, REAL* random_z, unsigned int (*warp_hists)[NUM_HISTOGRAMS], unsigned int NUM_ELEMENTS, unsigned int i, unsigned int j, Callback callback)
{
    // Iterate over all random points
    if (j < NUM_ELEMENTS) {
        // load current random point values
        REAL random_x_s;
        REAL random_y_s;
        REAL random_z_s;

        if(threadIdx.x + j < NUM_ELEMENTS) {
            random_x_s = random_x[threadIdx.x + j];
            random_y_s = random_y[threadIdx.x + j];
            random_z_s = random_z[threadIdx.x + j];
        }

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

            unsigned int warpnum = threadIdx.x / (WARP_SIZE/HISTS_PER_WARP);
            if((distance < dev_binb[min]) && (distance >= dev_binb[max]) &&
                    (!do_self || (threadIdx.x + j > i + k)) && (threadIdx.x + j < NUM_ELEMENTS)) {
                atomicAdd(&warp_hists[bin_index][warpnum], 1U);
            }
        }

        gloop::loop::postTask(loop, [=] (gloop::DeviceLoop* loop) {
            iterateOverAllRandomPoints(loop, data, do_self, random_x, random_y, random_z, warp_hists, NUM_ELEMENTS, i, j + BLOCK_SIZE, callback);
        });
        return;
    }

    callback(loop);
}

template<typename Callback>
__device__ void iterateOverAllDataPoints(gloop::DeviceLoop* loop, struct cartesian* data, REAL* data_x, REAL* data_y, REAL* data_z, bool do_self, REAL* random_x, REAL* random_y, REAL* random_z, unsigned int (*warp_hists)[NUM_HISTOGRAMS], unsigned int NUM_ELEMENTS, unsigned int i, Callback callback)
{
    // Iterate over all data points
    if (i < NUM_ELEMENTS) {
        // load current set of data into shared memory
        // (total of BLOCK_SIZE points loaded)
        if(threadIdx.x + i < NUM_ELEMENTS) {
            // reading outside of bounds is a-okay
            data[threadIdx.x] = (struct cartesian) {data_x[threadIdx.x + i], data_y[threadIdx.x + i], data_z[threadIdx.x + i]};
        }

        __syncthreads();

        iterateOverAllRandomPoints(loop, data, do_self, random_x, random_y, random_z, warp_hists, NUM_ELEMENTS, i, (do_self ? i+1 : 0), [=] (gloop::DeviceLoop* loop) {
            iterateOverAllDataPoints(loop, data, data_x, data_y, data_z, do_self, random_x, random_y, random_z, warp_hists, NUM_ELEMENTS, i + BLOCK_SIZE, callback);
        });
        return;
    }

    callback(loop);
}


__device__ void gen_hists(gloop::DeviceLoop* loop, hist_t* histograms, REAL* all_x_data, REAL* all_y_data, REAL* all_z_data, int NUM_SETS, int NUM_ELEMENTS)
{
    unsigned int bx = gloop::logicalBlockIdx.x;
    unsigned int tid = threadIdx.x;
    bool do_self = (bx < (NUM_SETS + 1));

    REAL* data_x;
    REAL* data_y;
    REAL* data_z;
    REAL* random_x;
    REAL* random_y;
    REAL* random_z;

    __shared__ struct cartesian* data;
    __shared__ unsigned int (*warp_hists)[NUM_HISTOGRAMS];
    BEGIN_SINGLE_THREAD
    {
        data = new struct cartesian[BLOCK_SIZE];
        warp_hists = new unsigned int[NUM_BINS][NUM_HISTOGRAMS]; // 640B <1k
    }
    END_SINGLE_THREAD

    for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE) {
        if(w+tid < NUM_BINS*NUM_HISTOGRAMS) {
            warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
        }
    }

    // Get stuff into shared memory to kick off the loop.
    if(!do_self) {
        data_x = all_x_data;
        data_y = all_y_data;
        data_z = all_z_data;
        random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
        random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
        random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
    } else {
        random_x = all_x_data + NUM_ELEMENTS * (bx);
        random_y = all_y_data + NUM_ELEMENTS * (bx);
        random_z = all_z_data + NUM_ELEMENTS * (bx);

        data_x = random_x;
        data_y = random_y;
        data_z = random_z;
    }

    gloop::loop::postTask(loop, [=] (gloop::DeviceLoop* loop) {
        iterateOverAllDataPoints(loop, data, data_x, data_y, data_z, do_self, random_x, random_y, random_z, warp_hists, NUM_ELEMENTS, 0, [=] (gloop::DeviceLoop* loop) {
            gloop::loop::postTask(loop, [=] (gloop::DeviceLoop* loop) {
                // coalesce the histograms in a block
                unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
                unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
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
                hist_t* hist_base = histograms + NUM_BINS * bx;
                if(tid < NUM_BINS) {
                    hist_base[tid] = warp_hists[tid][0];
                }

                BEGIN_SINGLE_THREAD
                {
                    delete [] data;
                    delete [] warp_hists;
                }
                END_SINGLE_THREAD
            });
        });
    });
}

// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
