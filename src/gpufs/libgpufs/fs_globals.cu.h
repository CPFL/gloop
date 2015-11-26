/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#ifndef fs_globals_cu_h
#define fs_globals_cu_h
#include <assert.h>
#include "hash_table.cu.h"
#include "cpu_ipc.cu.h"
#include "fs_structures.cu.h"
#include "mallocfree.cu.h"
#include "async_ipc.cu.h"

struct preclose_table;
/************GLOBALS********/
// CPU Write-shared memory //
extern __device__ volatile CPU_IPC_OPEN_Queue* g_cpu_ipcOpenQueue;
extern __device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue; 
extern __device__  async_close_rb_t* g_async_close_rb;
//
// manager for rw RPC queue

extern __device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;

// Open/Close table
extern __device__ volatile OTable* g_otable;
// Memory pool
extern __device__ volatile PPool* g_ppool;
// File table with block pointers
extern __device__ volatile FTable* g_ftable;

// Radix tree memory pool for rt_nodes
extern __device__ volatile rt_mempool g_rtree_mempool;

// Hash table with all the previously opened files indexed by their inodes
extern __device__ volatile hash_table g_closed_ftable;

// file_id uniq counter
extern __device__ int g_file_id;

//pre close table
//extern __device__ volatile preclose_table* g_preclose_table;


#endif
