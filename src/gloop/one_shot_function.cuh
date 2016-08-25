/*
  Partially modified.

  OneShotFunction is highly tuned for GLoop's callback.
  This cannot be used in the other purpose. Use gloop::function instead.

  Copyright (C) 2015-2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
/*
 * Copyright 2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#pragma once

#include "config.h"
#include "function.cuh"
#include "utility.h"
#include "utility/util.cu.h"

#if __cplusplus < 201103L
#if defined(_MSC_VER)
#if _MSC_VER < 1800
#error This library requires VS 2013 and above
#endif /* _MSC_VER < 1800 */
#else /* !_MSC_VER */
#error This library requires support for the ISO C++ 2011 standard
#endif /* _MSC_VER */
#endif /* __cplusplus */

#include <cstddef>
#include <new>
#include <type_traits>

// n3290 20.8
// Changed, gloop.
namespace gloop {
// 20.8.11 Polymorphic function wrappers [func.wrap]

// 20.8.11.1 Class bad_function_call [func.wrap.badcall]
// unimplemented because of exception
// class bad_function_call : public std::exception

// 20.8.11.2 Class template function [func.wrap.func]

template <class>
class OneShotFunction; // undefined

// Simplified version of template class function, which
//   * does not support allocator_arg_t;
//   * does not support target and target_type that rely on RTTI
//   * does not throw bad_function_call exception on invoking a NULL target
template <class _RetType, class... _ArgTypes>
class OneShotFunction<_RetType(_ArgTypes...)>
    : public __functional_helpers::__maybe_base_function<_RetType(_ArgTypes...)> {
    // FIXME: Testing...
public:
    __functional_helpers::_Small_functor_data __small_functor_data;
    void* __obj;

private:
    typedef _RetType (*__meta_fn_type)(OneShotFunction<_RetType(_ArgTypes...)>*, _ArgTypes...);
    __meta_fn_type __meta_fn;
    typedef void (*__cloner_type)(OneShotFunction&, const OneShotFunction&);
    __cloner_type __cloner;
    typedef void (*__destructor_type)(OneShotFunction*);
    __destructor_type __destructor;

    template <class _F>
    __device__ __host__ static constexpr bool __use_small_functor_data()
    {
        static_assert(sizeof(_F) <= sizeof(__functional_helpers::_Small_functor_data) && alignof(_F) <= alignof(__functional_helpers::_Small_functor_types), "Currently we restrict the statically allocatable lambda only.");
        return (sizeof(_F) <= sizeof(__functional_helpers::_Small_functor_data) && alignof(_F) <= alignof(__functional_helpers::_Small_functor_types));
    }

    __device__ __host__ void* __get_small_functor_data() const
    {
        return (void*)(&__small_functor_data.__data[0]);
    }

    __device__ __host__ bool __is_small_functor_data() const
    {
        return __obj == __get_small_functor_data();
    }

    template <class _F>
    __device__ __host__ static _F& __get_functor(void* __p)
    {
        return *((_F*)__p);
    }

    template <class _F>
    __device__ __host__ static bool __is_empty_functor(const _F& /*__p*/)
    {
        return false;
    }

    template <class _F>
    __device__ __host__ static bool __is_empty_functor(const _F* __p)
    {
        return !__p;
    }

    template <class _Res, class _C>
    __device__ __host__ static bool __is_empty_functor(const _Res _C::*__p)
    {
        return !__p;
    }

    template <class _Res, class... _Args>
    __device__ __host__ static bool __is_empty_functor(const OneShotFunction<_Res(_Args...)>& __p)
    {
        return !__p;
    }

    template <class _F>
    struct __make_cloner {
        __device__ __host__ static void __clone_data(OneShotFunction& __dest, const OneShotFunction& __src)
        {
            static_assert(__use_small_functor_data<_F>(), "OK");
            __dest.__obj = __dest.__get_small_functor_data();
            new (__dest.__obj) _F(__src.__get_functor<_F>(__src.__obj));
        }
    };

    template <class _F>
    struct __make_destructor {
        __device__ __host__
            GLOOP_ALWAYS_INLINE static void
            __destruct(OneShotFunction* __fn)
        {
            static_assert(__use_small_functor_data<_F>(), "OK");
            (__fn->__get_functor<_F>(__fn->__obj)).~_F();
        }
    };

    template <class _F,
        class Function,
        typename std::enable_if<!std::is_trivially_destructible<_F>::value, std::nullptr_t>::type = nullptr>
    __device__ static void oneShotDestroy(Function* function)
    {
        BEGIN_SINGLE_THREAD
        {
            __make_destructor<_F>::__destruct(function);
        }
        END_SINGLE_THREAD
    }

    template <class _F,
        class Function,
        typename std::enable_if<std::is_trivially_destructible<_F>::value, std::nullptr_t>::type = nullptr>
    __device__ static void oneShotDestroy(Function* function)
    {
    }

    // We cannot simple define __make_functor in the following way:
    // template <class _T, _F>
    // __make_functor;
    // template <class _RetType1, class _F, class... _ArgTypes1>
    // struct __make_functor<_RetType1(_ArgTypes1...), _F>
    //
    // because VS 2013 cannot unpack _RetType1(_ArgTypes1...)
    template <class FunctionType, class _RetType1, class _F, class... _ArgTypes1>
    struct __make_functor {
        typedef _RetType1 type;

        static_assert(std::is_same<_RetType1, void>::value, "Assume void.");

        __device__ static _RetType1 __invoke(FunctionType* function, _ArgTypes1... __args)
        {
            __get_functor<_F>(function->__obj)(internal::forward<_ArgTypes1>(__args)...);
            oneShotDestroy<_F>(function);
        }
    };

    template <class FunctionType, class _RetType1, class _C, class _M, class... _ArgTypes1>
    struct __make_functor<FunctionType, _RetType1, _M _C::*, _ArgTypes1...> {
        typedef _RetType1 type;
        typedef _RetType1 (*_F)(_ArgTypes1...);

        static_assert(std::is_same<_RetType1, void>::value, "Assume void.");

        __device__ static _RetType1 __invoke(FunctionType* function, _ArgTypes1... __args)
        {
            __get_functor<_F>(function->__obj)(internal::forward<_ArgTypes1>(__args)...);
            oneShotDestroy<_F>(function);
        }
    };

// workaround for GCC version below 4.8
#if (__GNUC__ == 4) && (__GNUC_MINOR__ < 8)
    template <class _F>
    struct __check_callability
        : public std::integral_constant<bool,
              !std::is_same<_F, std::nullptr_t>::value> {
    };
#elif defined(_MSC_VER)
    // simulate VC 2013's behavior...
    template <class _F>
    struct __check_callability1
        : public std::integral_constant<bool,
              // std::result_of does not handle member pointers well
              std::is_member_pointer<_F>::value || std::is_convertible<_RetType,
                                                       typename std::result_of<_F(_ArgTypes...)>::type>::value> {
    };

    template <class _F>
    struct __check_callability
        : public std::integral_constant<bool,
              !std::is_same<_F, OneShotFunction>::value && __check_callability1<typename std::remove_cv<_F>::type>::value> {
    };
#else /* !((__GNUC__ == 4) && (__GNUC_MINOR__ < 8)) _MSC_VER */
    template <class _F,
        class _T = typename std::result_of<_F(_ArgTypes...)>::type>
    struct __check_callability
        : public std::integral_constant<bool,
              !std::is_same<_F, OneShotFunction>::value && std::is_convertible<_T, _RetType>::value> {
    };
#endif /* __GNUC__ == 4) && (__GNUC_MINOR__ < 8) */

    __device__ __host__ void __destroy()
    {
        if (__obj) {
            __destructor(this);
            __obj = 0;
        }
    }

    __device__ __host__ void __clear()
    {
        __obj = 0;
        __meta_fn = 0;
        __cloner = 0;
        __destructor = 0;
    }

public:
    typedef _RetType result_type;

    /*
 * These typedef(s) are derived from __maybe_base_function
 * typedef T1 argument_type;        // only if sizeof...(ArgTypes) == 1 and
 *                                  // the type in ArgTypes is T1
 * typedef T1 first_argument_type;  // only if sizeof...(ArgTypes) == 2 and
 *                                  // ArgTypes contains T1 and T2
 * typedef T2 second_argument_type; // only if sizeof...(ArgTypes) == 2 and
 *                                  // ArgTypes contains T1 and T2
 */

    // 20.8.11.2.1 construct/copy/destroy [func.wrap.con]
    __device__ __host__
    OneShotFunction() noexcept
        : __obj(0),
          __meta_fn(0),
          __cloner(0),
          __destructor(0)
    {
    }

    __device__ __host__
        OneShotFunction(std::nullptr_t) noexcept
        : __obj(0),
          __meta_fn(0),
          __cloner(0),
          __destructor(0)
    {
    }

    __device__ __host__
    OneShotFunction(const OneShotFunction& __fn)
    {
        if (__fn.__obj == 0) {
            __clear();
        } else {
            __meta_fn = __fn.__meta_fn;
            __destructor = __fn.__destructor;
            __fn.__cloner(*this, __fn);
            __cloner = __fn.__cloner;
        }
    }

    __device__ __host__
    OneShotFunction(OneShotFunction&& __fn)
    {
        __fn.swap(*this);
    }

    // VS 2013 cannot process __check_callability type trait.
    // So, we check callability using static_assert instead of
    // using SFINAE such as
    // template<class _F,
    //          class = typename std::enable_if<
    //                    __check_callability<_F>::value
    //         >::type>
    template <class _F>
    __device__ __host__
        OneShotFunction(_F);

    // copy and swap
    __device__ __host__
        OneShotFunction&
        operator=(const OneShotFunction& __fn)
    {
        OneShotFunction(__fn).swap(*this);
        return *this;
    }

    __device__ __host__
        OneShotFunction&
        operator=(OneShotFunction&& __fn)
    {
        OneShotFunction(internal::move(__fn)).swap(*this);
        return *this;
    }

    __device__ __host__
        OneShotFunction&
        operator=(std::nullptr_t)
    {
        __destroy();
        return *this;
    }

    template <class _F>
    __device__ __host__
        OneShotFunction&
        operator=(_F&& __fn)
    {
        static_assert(__check_callability<_F>::value,
            "Unable to create functor object!");
        OneShotFunction(internal::forward<_F>(__fn)).swap(*this);
        return *this;
    }

    __device__ __host__ ~OneShotFunction()
    {
        __destroy();
    }

    // 20.8.11.2.2 OneShotFunction modifiers [func.wrap.func.mod]
    __device__ __host__ void swap(OneShotFunction& __fn) noexcept
    {
        internal::swap(__meta_fn, __fn.__meta_fn);
        internal::swap(__cloner, __fn.__cloner);
        internal::swap(__destructor, __fn.__destructor);

        if (__is_small_functor_data() && __fn.__is_small_functor_data()) {
            internal::swap(__small_functor_data, __fn.__small_functor_data);
        } else if (__is_small_functor_data()) {
            internal::swap(__small_functor_data, __fn.__small_functor_data);
            internal::swap(__obj, __fn.__obj);
            __fn.__obj = __fn.__get_small_functor_data();
        } else if (__fn.__is_small_functor_data()) {
            internal::swap(__small_functor_data, __fn.__small_functor_data);
            internal::swap(__obj, __fn.__obj);
            __obj = __get_small_functor_data();
        } else {
            internal::swap(__obj, __fn.__obj);
        }
    }

    // 20.8.11.2.3 OneShotFunction capacity [func.wrap.func.cap]
    __device__ __host__ explicit operator bool() const noexcept
    {
        return __obj;
    }

    // 20.8.11.2.4 OneShotFunction invocation [func.wrap.func.inv]
    // OneShotFunction::operator() can only be called in device code
    // to avoid cross-execution space calls
    __device__
        _RetType
        operator()(_ArgTypes...);
};

// Out-of-line definitions
template <class _RetType, class... _ArgTypes>
template <class _F>
__device__ __host__
OneShotFunction<_RetType(_ArgTypes...)>::OneShotFunction(_F __fn)
    : __obj(0)
    , __meta_fn(0)
    , __cloner(0)
    , __destructor(0)
{
    static_assert(__check_callability<_F>::value,
        "Unable to construct functor object!");
    if (__is_empty_functor(__fn))
        return;
    __meta_fn = &__make_functor<OneShotFunction<_RetType(_ArgTypes...)>, _RetType, _F, _ArgTypes...>::__invoke;
    __cloner = &__make_cloner<_F>::__clone_data;
    __destructor = &__make_destructor<_F>::__destruct;

    static_assert(__use_small_functor_data<_F>(), "OK");
    __obj = __get_small_functor_data();
    new ((void*)__obj) _F(internal::move(__fn));
}

template <class _RetType, class... _ArgTypes>
__device__
    _RetType
        OneShotFunction<_RetType(_ArgTypes...)>::operator()(_ArgTypes... __args)
{
    return __meta_fn(this, internal::device_forward<_ArgTypes>(__args)...);
}

// 20.8.11.2.6, Null pointer comparisons:

template <class _R, class... _ArgTypes>
__device__ __host__ bool operator==(const OneShotFunction<_R(_ArgTypes...)>& __fn, std::nullptr_t) noexcept
{
    return !__fn;
}

template <class _R, class... _ArgTypes>
__device__ __host__ bool operator==(std::nullptr_t, const OneShotFunction<_R(_ArgTypes...)>& __fn) noexcept
{
    return !__fn;
}

template <class _R, class... _ArgTypes>
__device__ __host__ bool operator!=(const OneShotFunction<_R(_ArgTypes...)>& __fn, std::nullptr_t) noexcept
{
    return static_cast<bool>(__fn);
}

template <class _R, class... _ArgTypes>
__device__ __host__ bool operator!=(std::nullptr_t, const OneShotFunction<_R(_ArgTypes...)>& __fn) noexcept
{
    return static_cast<bool>(__fn);
}

// 20.8.11.2.7, specialized algorithms:
template <class _R, class... _ArgTypes>
__device__ __host__ void swap(OneShotFunction<_R(_ArgTypes...)>& __fn1, OneShotFunction<_R(_ArgTypes...)>& __fn2)
{
    __fn1.swap(__fn2);
}

} // namespace gloop
