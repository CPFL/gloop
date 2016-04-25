/*
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#include <sys/mman.h>
#include <stdio.h>
#include <gloop/gloop.h>
#include <gloop/utility/util.cu.h>
#include "img.cuh"

__device__ volatile gpunet::INIT_LOCK init_lock;
__device__ volatile gpunet::LAST_SEMAPHORE last_lock;

__shared__ float input_img_row[GREP_ROW_WIDTH];
__shared__ float input_db_row[GREP_ROW_WIDTH];

struct _pagehelper{
    volatile uchar* page;
    size_t file_offset;
};

struct Context {
    char* db_files[6];
    int* out_buffer;
    int match_threshold;
    _pagehelper ph_input;
    _pagehelper ph_db;
};

template<typename Callback>
__device__ volatile float* get_row(gloop::DeviceLoop* loop, volatile uchar** cur_page_ptr, size_t* cur_page_offset, size_t req_file_offset, int max_file_size, int fd, int type, Callback callback)
{
    if (*cur_page_ptr!=NULL && *cur_page_offset+GLOOP_SHARED_PAGE_SIZE>req_file_offset) {
        callback(loop, (volatile float*)(*cur_page_ptr+(req_file_offset&(GLOOP_SHARED_PAGE_SIZE-1))));
        return;
    }

    auto continuation = [=] (gloop::DeviceLoop* loop) {
        int mapsize=(max_file_size-req_file_offset)>GLOOP_SHARED_PAGE_SIZE?GLOOP_SHARED_PAGE_SIZE:(max_file_size-req_file_offset);

        *cur_page_offset=(req_file_offset& (~(GLOOP_SHARED_PAGE_SIZE-1)));// round to the beg. of the page
        gloop::fs::mmap(loop, nullptr, mapsize, 0, type, fd, *cur_page_offset, [=](gloop::DeviceLoop* loop, volatile void* memory) {
            *cur_page_ptr=(volatile uchar*) memory;
            if (*cur_page_ptr == MAP_FAILED)
                GPU_ERROR("MMAP failed");
            callback(loop, (volatile float*)(*cur_page_ptr+(req_file_offset&(GLOOP_SHARED_PAGE_SIZE-1))));
        });
    };

    // remap
    if (*cur_page_ptr) {
        gloop::fs::munmap(loop, *cur_page_ptr, 0, [=](gloop::DeviceLoop* loop, int error) {
            if (error)
                GPU_ERROR("Unmap failed");
            continuation(loop);
        });
        return;
    }
    continuation(loop);
}

#define ACCUM_N 512
__shared__ volatile float s_reduction[ACCUM_N];

GLOOP_ALWAYS_INLINE __device__ float inner_product( volatile float* a, volatile float* b, int size)
{
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

GLOOP_ALWAYS_INLINE __device__ bool match(volatile float* a, volatile float* b, int size,float match_threshold)
{
    return sqrt(inner_product(a,b,size)) < match_threshold;
}

template<typename Callback>
void __device__ process_one_row(gloop::DeviceLoop* loop, Context* context, int data_idx, int db_idx, int out_count, int db_size, int zfd_db, int start_offset, int _cursor, int total_rows, int src_row_len, int db_rows, volatile float* ptr_row_db, Callback callback)
{
    if (_cursor<db_rows) {
        size_t _req_offset=(_cursor*src_row_len)<<2;
        auto continuation = [=] (gloop::DeviceLoop* loop, volatile float* ptr_row_db) {
            int found = match(input_img_row, ptr_row_db, src_row_len, context->match_threshold);
            BEGIN_SINGLE_THREAD
            {
                if (found){
                    context->out_buffer[out_count]=data_idx+start_offset*total_rows;
                    context->out_buffer[out_count+1]=db_idx;
                    context->out_buffer[out_count+2]=_cursor;
                }
            }
            END_SINGLE_THREAD
            if (found) {
                callback(loop, ptr_row_db, found);
                return;
            }
            process_one_row(loop, context, data_idx, db_idx, out_count, db_size, zfd_db, start_offset, _cursor + 1, total_rows, src_row_len, db_rows, ptr_row_db + src_row_len, callback);
        };
        if (_req_offset - context->ph_db.file_offset >= GLOOP_SHARED_PAGE_SIZE) {
            get_row(loop, &context->ph_db.page,&context->ph_db.file_offset,_req_offset,db_size,zfd_db,O_RDONLY, continuation);
            return;
        }
        continuation(loop, ptr_row_db);
        return;
    }
    callback(loop, ptr_row_db, /* not found */ 0);
}

template<typename Callback>
void __device__ process_one_db(gloop::DeviceLoop* loop, Context* context, int data_idx, int db_idx, int out_count, int start_offset, int num_db_files, int total_rows, int src_row_len, Callback callback)
{
    if (db_idx<num_db_files) {
        gloop::fs::open(loop, context->db_files[db_idx], O_RDONLY, [=](gloop::DeviceLoop* loop, int zfd_db) {
            if (zfd_db<0) GPU_ERROR("Failed to open DB file");
            gloop::fs::fstat(loop, zfd_db, [=](gloop::DeviceLoop* loop, size_t db_size) {
                size_t db_rows=(db_size/src_row_len)>>2;

                get_row(loop, &context->ph_db.page, &context->ph_db.file_offset, 0, db_size, zfd_db, O_RDONLY, [=](gloop::DeviceLoop* loop, volatile float* ptr_row_db) {
                    int _cursor = 0;
                    process_one_row(loop, context, data_idx, db_idx, out_count, db_size, zfd_db, start_offset, _cursor, total_rows, src_row_len, db_rows, ptr_row_db, [=](gloop::DeviceLoop* loop, volatile float* ptr_row_db, int found) {
                        gloop::fs::munmap(loop, ptr_row_db, 0, [=](gloop::DeviceLoop* loop, int error) {
                            if(error)
                                GPU_ERROR("Failed to unmap db");
                            context->ph_db.page=NULL; context->ph_db.file_offset=0;
                            gloop::fs::close(loop, zfd_db, [=](gloop::DeviceLoop* loop, int error) {
                                if (found) {
                                    callback(loop, found);
                                    return;
                                }
                                process_one_db(loop, context, data_idx, db_idx + 1, out_count, start_offset, num_db_files, total_rows, src_row_len, callback);
                            });
                        });
                    });
                });
            });
        });
        return;
    }
    callback(loop, /* found */ 0);
}

template<typename Callback>
void __device__ process_one_data(gloop::DeviceLoop* loop, Context* context, size_t data_idx, int out_count, int start, int rows_to_process, int zfd_src, int src_row_len, int start_offset, int total_rows, int num_db_files, Callback callback)
{
    if (data_idx < start + rows_to_process) {
        gloop::fs::read(loop, zfd_src, data_idx*src_row_len<<2, GREP_ROW_WIDTH*4, (uchar*)input_img_row, [=](gloop::DeviceLoop* loop, int bytes_read) {
            if (bytes_read!=GREP_ROW_WIDTH*4) GPU_ERROR("Failed to read src");

            int db_idx = 0;
            process_one_db(loop, context, data_idx, db_idx, out_count, start_offset, num_db_files, total_rows, src_row_len, [=] (gloop::DeviceLoop* loop, int found) {
                if (!found) {
                    BEGIN_SINGLE_THREAD
                    {
                        context->out_buffer[out_count]=data_idx+start_offset*total_rows;
                        context->out_buffer[out_count+1]=-1;
                        context->out_buffer[out_count+2]=-1;
                    }
                    END_SINGLE_THREAD
                }
                // Increment
                process_one_data(loop, context, data_idx + 1, out_count + 3, start, rows_to_process, zfd_src, src_row_len, start_offset, total_rows, num_db_files, callback);
            });
        });
        return;
    }
    callback(loop);
}

void __device__ img_gpu(
        gloop::DeviceLoop* loop,
        char* src, int src_row_len, int num_db_files, float match_threshold, int start_offset,
        char* out, char* out2, char* out3, char* out4, char *out5, char* out6, char* out7)
{
    src_row_len = GREP_ROW_WIDTH;
    gloop::fs::open(loop, out, O_RDWR|O_CREAT, [=](gloop::DeviceLoop* loop, int zfd_o) {
        if (zfd_o<0) GPU_ERROR("Failed to open output");

        gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop* loop, int zfd_src) {
            if (zfd_src<0) GPU_ERROR("Failed to open input");

            gloop::fs::fstat(loop, zfd_src, [=](gloop::DeviceLoop* loop, int in_size) {
                __shared__ int total_rows;
                total_rows=in_size/src_row_len>>2;

                __shared__ int rows_per_chunk;
                rows_per_chunk=total_rows/gridDim.x;
                if (rows_per_chunk==0) rows_per_chunk=1;

                __shared__ int rows_to_process;
                rows_to_process=rows_per_chunk;

                if (blockIdx.x==gridDim.x-1) rows_to_process=(total_rows - blockIdx.x*rows_per_chunk);

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
                    context->ph_input = {NULL,0};
                    context->ph_db = {NULL,0};
                }
                toInit=init_lock.try_wait();

                if (toInit == 1)
                {
                    // single_thread_ftruncate(zfd_o,0);
                    __threadfence();
                    init_lock.signal();
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
                volatile float* ptr_row_in=NULL;
                int found=0;
                int start=blockIdx.x*rows_per_chunk;

                process_one_data(loop, context, blockIdx.x*rows_per_chunk, out_count, start, rows_to_process, zfd_src, src_row_len, start_offset, total_rows, num_db_files, [=] (gloop::DeviceLoop* loop) {
                    //we are done.
                    //write the output and finish
                    //if (gmunmap(ptr_row_in,0)) GPU_ERROR("Failed to unmap input");
                    int write_size=rows_to_process*sizeof(int)*3;
                    gloop::fs::write(loop, zfd_o, blockIdx.x*rows_per_chunk*sizeof(int)*3, write_size, (uchar*)context->out_buffer, [=](gloop::DeviceLoop* loop, int written_size) {
                        if (written_size!=write_size) GPU_ERROR("Failed to write output");
                        gloop::fs::close(loop, zfd_src, [=](gloop::DeviceLoop* loop, int error) {
                            BEGIN_SINGLE_THREAD
                            {
                                free(context->out_buffer);
                                free(context);
                            }
                            END_SINGLE_THREAD
                            gloop::fs::close(loop, zfd_o, [=](gloop::DeviceLoop* loop, int error) { });
                        });
                    });
                });
            });
        });
    });
}

void init_device_app()
{
    GLOOP_CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<25));
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
