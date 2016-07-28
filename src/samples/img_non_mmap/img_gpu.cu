/*
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <gloop/gloop.h>
#include <gloop/utility/util.cu.h>
#include "img.cuh"

__device__ volatile gpunet::INIT_LOCK init_lock;
__device__ volatile gpunet::LAST_SEMAPHORE last_lock;

#define GLOOP_PAGE_SIZE (1 << 20)
// #define GLOOP_PAGE_SIZE GLOOP_SHARED_PAGE_SIZE

struct _pagehelper{
    uchar* page;
    size_t file_offset;
    int isFilled;
};

struct Context {
    float input_img_row[GREP_ROW_WIDTH];
    char* db_files[6];
    int* out_buffer;
    float match_threshold;
    int start_offset;
    int total_rows;
    int src_row_len;
    int num_db_files;
    int zfd_src;
    _pagehelper ph_db;
};

template<typename Callback>
__device__ void get_row(gloop::DeviceLoop<>* loop, uchar** cur_page_ptr, size_t* cur_page_offset, int* isFilled, size_t req_file_offset, int max_file_size, int fd, int type, int id, int db_idx, Callback callback)
{
    if (*isFilled && *cur_page_offset+GLOOP_PAGE_SIZE>req_file_offset) {
        callback(loop, (float*)(*cur_page_ptr+(req_file_offset&(GLOOP_PAGE_SIZE-1))));
        return;
    }

    int mapsize = (max_file_size-req_file_offset)>GLOOP_PAGE_SIZE?GLOOP_PAGE_SIZE:(max_file_size-req_file_offset);
    BEGIN_SINGLE_THREAD
    {

        *cur_page_offset=(req_file_offset& (~(GLOOP_PAGE_SIZE-1)));// round to the beg. of the page
        *isFilled = true;
    }
    END_SINGLE_THREAD
    gloop::fs::read(loop, fd, *cur_page_offset, mapsize, (unsigned char*)*cur_page_ptr, [=](gloop::DeviceLoop<>* loop, int bytesRead) {
//             BEGIN_SINGLE_THREAD
//                 printf("db:(%d),data:(%d),offset:(%u),checksum:(%04x)\n", db_idx, id, (unsigned)(*cur_page_offset), (unsigned)checksum((const uint8_t*)*cur_page_ptr, mapsize));
//             END_SINGLE_THREAD
        callback(loop, (float*)(*cur_page_ptr+(req_file_offset&(GLOOP_PAGE_SIZE-1))));
    });
}

GLOOP_ALWAYS_INLINE __device__ float inner_product( float* a, float* b, int size)
{
    #define ACCUM_N 512
    __shared__ volatile float s_reduction[ACCUM_N];

    float tmp=0;
    //      __syncthreads();
    //      if (threadIdx.x==0) {
    //                      *res=0;
    //      }
    //      __syncthreads();
    int i=0;
    for( i=threadIdx.x;i<size;i+=blockDim.x){

        tmp+=(a[i]-b[i])*(a[i]-b[i]);
    }
    s_reduction[threadIdx.x]=tmp;

    __syncthreads();
    for (int stride = ACCUM_N / 2; stride > 32; stride >>= 1)
    {
        if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
        __syncthreads();
    }
    for (int stride = 32; stride > 0 && threadIdx.x<32 ; stride >>=1 )
    {
        if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
    }

    __syncthreads();

    return s_reduction[0];
}

GLOOP_ALWAYS_INLINE __device__ bool match(float* a, float* b, int size, float match_threshold, int data_idx, int db_idx)
{
//     float result = sqrt(inner_product(a,b,size));
//     BEGIN_SINGLE_THREAD
//         printf("data:(%d),db:(%d),match:(%d),res:(%f)\n", (int)data_idx, (int)(result < match_threshold), (float)result, (int)size);
//     END_SINGLE_THREAD
    return sqrt(inner_product(a,b,size)) < match_threshold;
}

template<typename Callback>
void __device__ process_one_row(gloop::DeviceLoop<>* loop, Context* context, int data_idx, int db_idx, int out_count, int db_size, int zfd_db, int _cursor, int db_rows, float* ptr_row_db, Callback callback)
{
    if (_cursor < db_rows) {
        size_t _req_offset=(_cursor*context->src_row_len)<<2;
        auto continuation = [=] (gloop::DeviceLoop<>* loop, float* ptr_row_db) {

            int found = match(context->input_img_row, ptr_row_db, context->src_row_len, context->match_threshold, data_idx, db_idx);
            BEGIN_SINGLE_THREAD
            {
                // printf("img %d %d\n", db_idx, data_idx);
                if (found) {
                    context->out_buffer[out_count]=data_idx+context->start_offset*context->total_rows;
                    context->out_buffer[out_count+1]=db_idx;
                    context->out_buffer[out_count+2]=_cursor;
                }
            }
            END_SINGLE_THREAD
            if (!found) {
                gloop::loop::postTask(loop, [=](gloop::DeviceLoop<>* loop) {
                    process_one_row(loop, context, data_idx, db_idx, out_count, db_size, zfd_db, _cursor + 1, db_rows, ptr_row_db + context->src_row_len, callback);
                });
                return;
            }
            callback(loop, ptr_row_db, found);
        };
        if (_req_offset - context->ph_db.file_offset >= GLOOP_PAGE_SIZE) {
            get_row(loop, &context->ph_db.page, &context->ph_db.file_offset, &context->ph_db.isFilled, _req_offset, db_size, zfd_db, PROT_READ | PROT_WRITE, data_idx, db_idx, continuation);
            return;
        }
        continuation(loop, ptr_row_db);
        return;
    }
    callback(loop, ptr_row_db, /* not found */ 0);
}

template<typename Callback>
void __device__ process_one_db(gloop::DeviceLoop<>* loop, Context* context, int data_idx, int db_idx, int out_count, Callback callback)
{
    if (db_idx < context->num_db_files) {
        gloop::fs::open(loop, context->db_files[db_idx], /* O_RDONLY */ O_RDWR, [=](gloop::DeviceLoop<>* loop, int zfd_db) {
            if (zfd_db<0) GPU_ERROR("Failed to open DB file");
            gloop::fs::fstat(loop, zfd_db, [=](gloop::DeviceLoop<>* loop, size_t db_size) {
                size_t db_rows=(db_size/context->src_row_len)>>2;

                get_row(loop, &context->ph_db.page, &context->ph_db.file_offset, &context->ph_db.isFilled, 0, db_size, zfd_db, PROT_READ | PROT_WRITE, -1, db_idx, [=](gloop::DeviceLoop<>* loop, float* ptr_row_db) {

                    process_one_row(loop, context, data_idx, db_idx, out_count, db_size, zfd_db, 0, db_rows, ptr_row_db, [=](gloop::DeviceLoop<>* loop, float* ptr_row_db, int found) {
                        context->ph_db.file_offset = 0;
                        context->ph_db.isFilled = false;
                        gloop::fs::close(loop, zfd_db, [=](gloop::DeviceLoop<>* loop, int error) {
                            if (found) {
                                callback(loop, found);
                                return;
                            }
                            process_one_db(loop, context, data_idx, db_idx + 1, out_count, callback);
                        });
                    });
                });
            });
        });
        return;
    }
    context->ph_db.file_offset=0;
    context->ph_db.isFilled=false;
    callback(loop, /* found */ 0);
}

template<typename Callback>
void __device__ process_one_data(gloop::DeviceLoop<>* loop, Context* context, size_t data_idx, int out_count, int limit, Callback callback)
{
    if (data_idx < limit) {
//         BEGIN_SINGLE_THREAD
//             // printf("data %u\n", (unsigned)data_idx);
//         END_SINGLE_THREAD
        gloop::fs::read(loop, context->zfd_src, data_idx*context->src_row_len<<2, GREP_ROW_WIDTH*4, (uchar*)context->input_img_row, [=](gloop::DeviceLoop<>* loop, int bytes_read) {
            if (bytes_read!=GREP_ROW_WIDTH*4) GPU_ERROR("Failed to read src");


            int db_idx = 0;
            context->ph_db.file_offset=0;
            context->ph_db.isFilled=false;
            process_one_db(loop, context, data_idx, db_idx, out_count, [=] (gloop::DeviceLoop<>* loop, int found) {
                BEGIN_SINGLE_THREAD
                {
                    if (!found) {
                        context->out_buffer[out_count]=data_idx+context->start_offset*context->total_rows;
                        context->out_buffer[out_count+1]=-1;
                        context->out_buffer[out_count+2]=-1;
                    }
                }
                END_SINGLE_THREAD
                // Increment
                process_one_data(loop, context, data_idx + 1, out_count + 3, limit, callback);
            });
        });
        return;
    }
    callback(loop);
}

void __device__ img_gpu(
        gloop::DeviceLoop<>* loop,
        char* src, int src_row_len, int num_db_files, float match_threshold, int start_offset,
        char* out, char* out2, char* out3, char* out4, char *out5, char* out6, char* out7)
{
    src_row_len = GREP_ROW_WIDTH;
    gloop::fs::open(loop, out, O_RDWR|O_CREAT, [=](gloop::DeviceLoop<>* loop, int zfd_o) {
        if (zfd_o<0) GPU_ERROR("Failed to open output");

        gloop::fs::open(loop, src, /* O_RDONLY */ O_RDWR, [=](gloop::DeviceLoop<>* loop, int zfd_src) {
            if (zfd_src<0) GPU_ERROR("Failed to open input");

            gloop::fs::fstat(loop, zfd_src, [=](gloop::DeviceLoop<>* loop, int in_size) {
                int total_rows;
                total_rows=(in_size/src_row_len)>>2;

                int rows_per_chunk;
                rows_per_chunk=total_rows/loop->logicalGridDim().x;
                if (rows_per_chunk==0) rows_per_chunk=1;

                int rows_to_process;
                rows_to_process=rows_per_chunk;

                if (loop->logicalBlockIdx().x==loop->logicalGridDim().x-1)
                    rows_to_process=(total_rows - loop->logicalBlockIdx().x*rows_per_chunk);

                __shared__ int toInit;
                __shared__ Context* context;
                BEGIN_SINGLE_THREAD
                {
                    context = (Context*)malloc(sizeof(Context));
                    context->out_buffer=(int*)malloc(rows_to_process*sizeof(int)*3);
                    context->db_files[0] = out2;
                    context->db_files[1] = out3;
                    context->db_files[2] = out4;
                    context->db_files[3] = out5;
                    context->db_files[4] = out6;
                    context->db_files[5] = out7;
                    context->match_threshold = match_threshold;
                    context->start_offset = start_offset;
                    context->total_rows = total_rows;
                    context->src_row_len = src_row_len;
                    context->num_db_files = num_db_files;
                    context->ph_db = {(uchar*)malloc(GLOOP_PAGE_SIZE),0, false};
                    context->zfd_src = zfd_src;

                    toInit=init_lock.try_wait();
                    if (toInit == 1)
                    {
                        // single_thread_ftruncate(zfd_o,0);
                        __threadfence();
                        init_lock.signal();
                    }
                }
                END_SINGLE_THREAD

                /*
                   1. decide how many strings  each block does
                   2. map input line
                   3. map db
                   4. scan through
                   5. write to output
                   */
                int out_count=0;
                int start=loop->logicalBlockIdx().x*rows_per_chunk;

                process_one_data(loop, context, start, out_count, start + rows_to_process, [=] (gloop::DeviceLoop<>* loop) {
                    //we are done.
                    //write the output and finish
                    //if (gmunmap(ptr_row_in,0)) GPU_ERROR("Failed to unmap input");
                    int write_size=rows_to_process*sizeof(int)*3;
                    gloop::fs::write(loop, zfd_o, loop->logicalBlockIdx().x*rows_per_chunk*sizeof(int)*3, write_size, (uchar*)context->out_buffer, [=](gloop::DeviceLoop<>* loop, int written_size) {
                        if (written_size!=write_size) GPU_ERROR("Failed to write output");
                        gloop::fs::close(loop, zfd_src, [=](gloop::DeviceLoop<>* loop, int error) {
                            BEGIN_SINGLE_THREAD
                            {
                                free(context->out_buffer);
                                free(context);
                            }
                            END_SINGLE_THREAD
                            gloop::fs::close(loop, zfd_o, [=](gloop::DeviceLoop<>* loop, int error) { });
                        });
                    });
                });
            });
        });
    });
}

void init_device_app()
{
    // GLOOP_CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1<<25));
    GLOOP_CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, ((512) <<20)));
}

void init_app()
{
    void* inited;

    GLOOP_CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,init_lock));
    GLOOP_CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(gpunet::INIT_LOCK)));

    GLOOP_CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,last_lock));
    GLOOP_CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(gpunet::LAST_SEMAPHORE)));
}

double post_app(double total_time, float trials)
{
    return 0;
}
