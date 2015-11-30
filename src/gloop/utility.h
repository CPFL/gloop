/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef GLOOP_UTILITY_H_
#define GLOOP_UTILITY_H_

#define GLOOP_CONCAT1(x, y) x##y
#define GLOOP_CONCAT(x, y) GLOOP_CONCAT1(x, y)

#define GLOOP_SINGLE_THREAD() \
    __syncthreads();\
    for (\
        bool GLOOP_CONCAT(context, __LINE__) { false };\
        threadIdx.x+threadIdx.y+threadIdx.z ==0 && (GLOOP_CONCAT(context, __LINE__) = !GLOOP_CONCAT(context, __LINE__));\
        __syncthreads()\
    )

// see http://www5d.biglobe.ne.jp/~noocyte/Programming/BigAlignmentBlock.html
#define GLOOP_ALIGNED_SIZE(size, alignment) ((size) + (alignment) - 1)
#define GLOOP_ALIGNED_ADDRESS(address, alignment) ((address + (alignment - 1)) & ~(alignment - 1))

// only 2^n and unsigned
#define GLOOP_ROUNDUP(x, y) (((x) + (y - 1)) & ~(y - 1))

// only 2^n and unsinged
#define GLOOP_ROUNDDOWN(x, y) ((x) & (-(y)))

#endif  // GLOOP_UTILITY_H_