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


#ifndef HOST_LOOP_CPP
#define HOST_LOOP_CPP

struct GPUGlobals;

void open_loop(volatile GPUGlobals* globals,int gpuid);
void rw_loop(volatile GPUGlobals* globals);
void async_close_loop(volatile GPUGlobals* globals);
void logGPUfsDone();

#endif
