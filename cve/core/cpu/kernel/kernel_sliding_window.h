/*====================================================================
BSD 2-Clause License

Copyright (c) 2018, Ruler
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_KERNEL_SLIDING_WINDOW_H__
#define __CORE_CPU_KERNEL_SLIDING_WINDOW_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template kernel_sliding_block

	template<class T, cpu_inst_type inst>
	struct kernel_sliding_block
	{
		void operator()(T *data, const T step, const T stride, const T offset) const
		{
			T delta = offset;
			T *pointer = data + stride;

			for (T j = 0; j < step; ++j)
			{
				for (T i = 0; i < stride; ++i)
					pointer[i] = data[i] + delta;
				delta += offset;
				pointer += stride;
			}
		}
	};

	template<>
	struct kernel_sliding_block<unsigned int, cpu_sse2>
	{
		void operator()(unsigned int *data, const unsigned int step, const unsigned int stride, const unsigned int offset) const
		{
			constexpr unsigned int block = 4;
			const unsigned int aligned = stride & ~(block - 1);
			unsigned int delta = offset;
			unsigned int *pointer = data + stride;
			__m128i xmm_a, xmm_b, xmm_c;

			for (unsigned int j = 0; j < step; ++j)
			{
				xmm_b = _mm_set1_epi32(delta);
				for (unsigned int i = 0; i < aligned; i += block)
				{
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
					xmm_c = _mm_add_epi32(xmm_a, xmm_b);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(pointer + i), xmm_c);
				}
				for (unsigned int i = aligned; i < stride; ++i)
					pointer[i] = data[i] + delta;
				delta += offset;
				pointer += stride;
			}
		}
	};

	template<>
	struct kernel_sliding_block<unsigned __int64, cpu_sse2>
	{
		void operator()(unsigned __int64 *data, const unsigned __int64 step, const unsigned __int64 stride, const unsigned __int64 offset) const
		{
			constexpr unsigned __int64 block = 2;
			const unsigned __int64 aligned = stride & ~(block - 1);
			unsigned __int64 delta = offset;
			unsigned __int64 *pointer = data + stride;
			__m128i xmm_a, xmm_b, xmm_c;

			for (unsigned __int64 j = 0; j < step; ++j)
			{
				xmm_b = _mm_set1_epi64x(delta);
				for (unsigned __int64 i = 0; i < aligned; i += block)
				{
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
					xmm_c = _mm_add_epi64(xmm_a, xmm_b);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(pointer + i), xmm_c);
				}
				for (unsigned __int64 i = aligned; i < stride; ++i)
					pointer[i] = data[i] + delta;
				delta += offset;
				pointer += stride;
			}
		}
	};

	template<>
	struct kernel_sliding_block<unsigned int, cpu_avx2>
	{
		void operator()(unsigned int *data, const unsigned int step, const unsigned int stride, const unsigned int offset) const
		{
			constexpr unsigned int block = 8;
			const unsigned int aligned = stride & ~(block - 1);
			unsigned int delta = offset;
			unsigned int *pointer = data + stride;
			__m256i ymm_a, ymm_b, ymm_c;

			for (unsigned int j = 0; j < step; ++j)
			{
				ymm_b = _mm256_set1_epi32(delta);
				for (unsigned int i = 0; i < aligned; i += block)
				{
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
					ymm_c = _mm256_add_epi32(ymm_a, ymm_b);
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(pointer + i), ymm_c);
				}
				for (unsigned int i = aligned; i < stride; ++i)
					pointer[i] = data[i] + delta;
				delta += offset;
				pointer += stride;
			}
		}
	};

	template<>
	struct kernel_sliding_block<unsigned __int64, cpu_avx2>
	{
		void operator()(unsigned __int64 *data, const unsigned __int64 step, const unsigned __int64 stride, const unsigned __int64 offset) const
		{
			constexpr unsigned __int64 block = 4;
			const unsigned __int64 aligned = stride & ~(block - 1);
			unsigned __int64 delta = offset;
			unsigned __int64 *pointer = data + stride;
			__m256i ymm_a, ymm_b, ymm_c;

			for (unsigned __int64 j = 0; j < step; ++j)
			{
				ymm_b = _mm256_set1_epi64x(delta);
				for (unsigned __int64 i = 0; i < aligned; i += block)
				{
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
					ymm_c = _mm256_add_epi64(ymm_a, ymm_b);
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(pointer + i), ymm_c);
				}
				for (unsigned __int64 i = aligned; i < stride; ++i)
					pointer[i] = data[i] + delta;
				delta += offset;
				pointer += stride;
			}
		}
	};

	// Function template kernel_sliding_window
	template<class T, cpu_inst_type inst>
	void kernel_sliding_window(T *data, const T rows, const T columns, const T channels, const T window_h, const T window_w, const T stride_h, const T stride_w)
	{
		T value = 0;
		T stride = columns * channels;
		T window_area = window_h * window_w;
		T window_size = channels * window_area;
		T step_h = (rows - window_h) / stride_h;
		T step_w = (columns - window_w) / stride_w;
		const struct kernel_sliding_block<T, inst> functor;

		// 1 * window_w
		for (T i = 0; i < window_w; ++i)
		{
			data[i] = value;
			value += channels;
		}
		// window_h * window_w
		functor(data, window_h, window_w, stride);
		// channels * window_h * window_w
		functor(data, channels, window_area, 1);
		// horizontal sliding window
		functor(data, step_w, window_size, stride_w * channels);
		// vertical sliding window
		functor(data, step_h, (step_w + 1) * window_size, stride_h * stride);
	};

} // namespace core

#endif
