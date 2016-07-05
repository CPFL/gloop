/*
  Copyright (C) 2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
#pragma once
#include <gloop/gloop.h>

GLOOP_ALWAYS_INLINE __device__ float4 sortElem(float4 r)
{
    float4 nr;

    nr.x = (r.x > r.y) ? r.y : r.x;
    nr.y = (r.y > r.x) ? r.y : r.x;
    nr.z = (r.z > r.w) ? r.w : r.z;
    nr.w = (r.w > r.z) ? r.w : r.z;

    r.x = (nr.x > nr.z) ? nr.z : nr.x;
    r.y = (nr.y > nr.w) ? nr.w : nr.y;
    r.z = (nr.z > nr.x) ? nr.z : nr.x;
    r.w = (nr.w > nr.y) ? nr.w : nr.y;

    nr.x = r.x;
    nr.y = (r.y > r.z) ? r.z : r.y;
    nr.z = (r.z > r.y) ? r.z : r.y;
    nr.w = r.w;
    return nr;
}

GLOOP_ALWAYS_INLINE __device__ float4 getLowest(float4 a, float4 b)
{
    //float4 na;
    a.x = (a.x < b.w) ? a.x : b.w;
    a.y = (a.y < b.z) ? a.y : b.z;
    a.z = (a.z < b.y) ? a.z : b.y;
    a.w = (a.w < b.x) ? a.w : b.x;
    return a;
}

GLOOP_ALWAYS_INLINE __device__ float4 getHighest(float4 a, float4 b)
{
    b.x = (a.w >= b.x) ? a.w : b.x;
    b.y = (a.z >= b.y) ? a.z : b.y;
    b.z = (a.y >= b.z) ? a.y : b.z;
    b.w = (a.x >= b.w) ? a.x : b.w;
    return b;
}
