/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <gloop/benchmark.h>
#include <gloop/gloop.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "args.h"

#include "model.h"
#include "tpacf_kernel.cu"

#define CUDA_ERRCK { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        return -1; }}

extern unsigned int NUM_SETS;
extern unsigned int NUM_ELEMENTS;

int main( int argc, char** argv)
{
    struct pb_TimerSet timers;
    struct pb_Parameters *params;

    {
        pb_InitializeTimerSet( &timers );
        params = pb_ReadParameters( &argc, argv );

        options args;
        parse_args(argc, argv, &args);

        pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

        NUM_ELEMENTS = args.npoints;
        NUM_SETS = args.random_count;
        int num_elements = NUM_ELEMENTS;

        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(NUM_SETS*2 + 1);
        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, dimGrid, dimGrid);

        printf("Min distance: %f arcmin\n", min_arcmin);
        printf("Max distance: %f arcmin\n", max_arcmin);
        printf("Bins per dec: %i\n", bins_per_dec);
        printf("Total bins  : %i\n", NUM_BINS);

        //read in files
        unsigned mem_size = (1+NUM_SETS)*num_elements*sizeof(struct cartesian);
        unsigned f_mem_size = (1+NUM_SETS)*num_elements*sizeof(REAL);

        // container for all the points read from files
        struct cartesian *h_all_data;
        h_all_data = (struct cartesian*) malloc(mem_size);
        // Until I can get libs fixed

        // iterator for data files
        struct cartesian *working = h_all_data;

        // go through and read all data and random points into h_all_data
        pb_SwitchToTimer( &timers, pb_TimerID_IO );
        readdatafile(params->inpFiles[0], working, num_elements);
        pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

        working += num_elements;
        for(int i = 0; i < (NUM_SETS); i++)
        {
            pb_SwitchToTimer( &timers, pb_TimerID_IO );
            readdatafile(params->inpFiles[i+1], working, num_elements);
            pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

            working += num_elements;
        }

        // split into x, y, and z arrays
        REAL * h_x_data = (REAL*) malloc (3*f_mem_size);
        REAL * h_y_data = h_x_data + NUM_ELEMENTS*(NUM_SETS+1);
        REAL * h_z_data = h_y_data + NUM_ELEMENTS*(NUM_SETS+1);
        for(int i = 0; i < (NUM_SETS+1); ++i)
        {
            for(int j = 0; j < NUM_ELEMENTS; ++j)
            {
                h_x_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].x;
                h_y_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].y;
                h_z_data[i*NUM_ELEMENTS+j] = h_all_data[i*NUM_ELEMENTS+j].z;
            }
        }

        // from on use x, y, and z arrays, free h_all_data
        free(h_all_data);

        // allocate cuda memory to hold all points
        hist_t* new_hists;
        hist_t* d_hists;
        REAL* d_x_data;
        REAL * d_y_data;
        REAL * d_z_data;
        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());

            pb_SwitchToTimer( &timers, pb_TimerID_COPY );
            cudaMalloc((void**) & d_x_data, 3*f_mem_size);
            CUDA_ERRCK
            d_y_data = d_x_data + NUM_ELEMENTS*(NUM_SETS+1);
            d_z_data = d_y_data + NUM_ELEMENTS*(NUM_SETS+1);

            // allocate cuda memory to hold final histograms
            // (1 for dd, and NUM_SETS for dr and rr apiece)
            cudaMalloc((void**) & d_hists, NUM_BINS*(NUM_SETS*2+1)*sizeof(hist_t) );
            CUDA_ERRCK
            pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

            // allocate system memory for final histograms
            new_hists = (hist_t *) malloc(NUM_BINS*(NUM_SETS*2+1)* sizeof(hist_t));

            // Initialize the boundary constants for bin search
            initBinB( &timers );
            CUDA_ERRCK

            // **===------------------ Kick off TPACF on CUDA------------------===**
            pb_SwitchToTimer( &timers, pb_TimerID_COPY );
            cudaMemcpy(d_x_data, h_x_data, 3*f_mem_size, cudaMemcpyHostToDevice);
            CUDA_ERRCK
            pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );
        }

        {
            // FIXME.
            // gloop::Benchmark benchmark;
            // cudaDeviceSynchronize();
            // benchmark.begin();
            hostLoop->launch(*hostContext, dimBlock, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop* loop, hist_t* histograms, REAL* all_x_data, REAL* all_y_data, REAL* all_z_data, unsigned int NUM_SETS, unsigned int NUM_ELEMENTS) {
                gen_hists(loop, histograms, all_x_data, all_y_data, all_z_data, NUM_SETS, NUM_ELEMENTS);
            }, d_hists, d_x_data, d_y_data, d_z_data, NUM_SETS, NUM_ELEMENTS);
            // FIXME.
            // cudaDeviceSynchronize();
            // benchmark.end();
            // benchmark.report();
        }

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            pb_SwitchToTimer( &timers, pb_TimerID_COPY );
            cudaMemcpy(new_hists, d_hists, NUM_BINS*(NUM_SETS*2+1)* sizeof(hist_t), cudaMemcpyDeviceToHost);
            CUDA_ERRCK
            pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
        }
        // **===-----------------------------------------------------------===**

        // references into output histograms
        hist_t *dd_hist = new_hists;
        hist_t *rr_hist = dd_hist + NUM_BINS;
        hist_t *dr_hist = rr_hist + NUM_BINS*NUM_SETS;

        // add up values within dr and rr
        int rr[NUM_BINS];
        for(int i=0; i<NUM_BINS; i++)
        {
            rr[i] = 0;
        }
        for(int i=0; i<NUM_SETS; i++)
        {
            for(int j=0; j<NUM_BINS; j++)
            {
                rr[j] += rr_hist[i*NUM_BINS + j];
            }
        }
        int dr[NUM_BINS];
        for(int i=0; i<NUM_BINS; i++)
        {
            dr[i] = 0;
        }
        for(int i=0; i<NUM_SETS; i++)
        {
            for(int j=0; j<NUM_BINS; j++)
            {
                dr[j] += dr_hist[i*NUM_BINS + j];
            }
        }

        //int dd_t = 0;
        //int dr_t = 0;
        //int rr_t = 0;
        FILE *outfile;
        if ((outfile = fopen(params->outFile, "w")) == NULL)
        {
            fprintf(stderr, "Unable to open output file %s for writing, "
                    "assuming stdout\n", params->outFile);
            outfile = stdout;
        }

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            pb_SwitchToTimer( &timers, pb_TimerID_IO );
            // print out final histograms + omega (while calculating omega)
            for(int i=0; i<NUM_BINS; i++)
            {
                //REAL w = (100.0 * dd_hist[i] - dr[i]) / rr[i] + 1.0;
                //fprintf(outfile, "%f\n", w);
                fprintf(outfile, "%d\n%d\n%d\n", dd_hist[i], dr[i], rr[i]);
                //      dd_t += dd_hist[i];
                //      dr_t += dr[i];
                //      rr_t += rr[i];
            }
            pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
        }

        if(outfile != stdout)
            fclose(outfile);

        // cleanup memory
        free(new_hists);
        free( h_x_data);

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            pb_SwitchToTimer( &timers, pb_TimerID_COPY );
            cudaFree( d_hists );
            cudaFree( d_x_data );
            pb_SwitchToTimer(&timers, pb_TimerID_NONE);
            pb_PrintTimerSet(&timers);
            pb_FreeParameters(params);
        }
    }
}

