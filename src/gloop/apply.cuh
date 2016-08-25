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

#include <thrust/tuple.h>
#include <type_traits>
#include <utility>
namespace gloop {

// http://stackoverflow.com/questions/687490/how-do-i-expand-a-tuple-into-variadic-template-functions-arguments

// ------------- UTILITY---------------
template <int...>
struct index_tuple {
};

template <int I, typename IndexTuple, typename... Types>
struct make_indexes_impl;

template <int I, int... Indexes, typename T, typename... Types>
struct make_indexes_impl<I, index_tuple<Indexes...>, T, Types...> {
    typedef typename make_indexes_impl<I + 1, index_tuple<Indexes..., I>, Types...>::type type;
};

template <int I, int... Indexes>
struct make_indexes_impl<I, index_tuple<Indexes...>> {
    typedef index_tuple<Indexes...> type;
};

template <typename... Types>
struct make_indexes : make_indexes_impl<0, index_tuple<>, Types...> {
};

// Function pointers.
template <class Ret, class... Args, int... Indexes>
__device__ __host__
    Ret
    apply_helper(Ret (*pf)(Args...), index_tuple<Indexes...>, thrust::tuple<Args...>&& tup)
{
    return pf(std::forward<Args>(thrust::get<Indexes>(tup))...);
}

template <class Ret, class... Args>
__device__ __host__
    Ret
    apply(Ret (*pf)(Args...), const thrust::tuple<Args...>& tup)
{
    return apply_helper(pf, typename make_indexes<Args...>::type(), thrust::tuple<Args...>(tup));
}

template <class Ret, class... Args>
__device__ __host__
    Ret
    apply(Ret (*pf)(Args...), thrust::tuple<Args...>&& tup)
{
    return apply_helper(pf, typename make_indexes<Args...>::type(), std::forward<thrust::tuple<Args...>>(tup));
}

// Lambdas.
template <class Lambda, class... Args, int... Indexes>
__device__ __host__ auto apply_helper(Lambda lambda, index_tuple<Indexes...>, thrust::tuple<Args...>&& tup) -> typename std::result_of<decltype(lambda(std::forward<Args>(thrust::get<Indexes>(tup))...))>::type
{
    return lambda(std::forward<Args>(thrust::get<Indexes>(tup))...);
}

template <class Lambda, class... Args>
__device__ __host__ auto apply(Lambda lambda, const thrust::tuple<Args...>& tup) -> typename std::result_of<decltype(apply_helper(lambda, typename make_indexes<Args...>::type(), thrust::tuple<Args...>(tup)))>::type
{
    return apply_helper(lambda, typename make_indexes<Args...>::type(), thrust::tuple<Args...>(tup));
}

template <class Lambda, class... Args>
__device__ __host__ auto apply(Lambda lambda, thrust::tuple<Args...>&& tup) -> typename std::result_of<decltype(apply_helper(lambda, typename make_indexes<Args...>::type(), std::forward<thrust::tuple<Args...>>(tup)))>::type
{
    return apply_helper(lambda, typename make_indexes<Args...>::type(), std::forward<thrust::tuple<Args...>>(tup));
}

} // namespace gloop
