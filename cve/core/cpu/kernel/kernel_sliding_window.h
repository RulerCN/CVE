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

#include "../cpu.h"

namespace core
{
	// Class template kernel_sliding_block
	template<class T, cpu_inst_type inst>
	struct kernel_sliding_block
	{
		void operator()(T *data, const T step, const T stride, const T offset) const
		{
			//constexpr T block = 8;
			T *a = data;
			T *b = data + stride;

			for (T j = 1; j < step; ++j)
			{
				for (T i = 0; i < stride; ++i)
					b[i] = a[i] + offset;
				a += stride;
				b += stride;

				//i = stride;
				//while (i > block)
				//{
				//	b[0] = a[0] + offset;
				//	b[1] = a[1] + offset;
				//	b[2] = a[2] + offset;
				//	b[3] = a[3] + offset;
				//	b[4] = a[4] + offset;
				//	b[5] = a[5] + offset;
				//	b[6] = a[6] + offset;
				//	b[7] = a[7] + offset;
				//	a += block;
				//	b += block;
				//	i -= block;
				//}
				//while (i > 0)
				//{
				//	*b++ = *a++ + offset;
				//	--i;
				//}
			}
		}
	};

	// Class template kernel_sliding_block
	template<>
	struct kernel_sliding_block<unsigned int, cpu_sse2>
	{
		void operator()(unsigned int *data, const unsigned int step, const unsigned int stride, const unsigned int offset) const
		{
			constexpr unsigned int block = 8;
			unsigned int *a = data;
			unsigned int *b = data + stride;
			__m128i xmm_val = _mm_set1_epi32(offset);
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0, xmm_b1;

			for (unsigned __int64 i, j = 1; j < step; ++j)
			{
				i = stride;
				while (i > block)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 4));
					// numerical migration
					xmm_b0 = _mm_add_epi32(xmm_a0, xmm_val);
					xmm_b1 = _mm_add_epi32(xmm_a1, xmm_val);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + 4), xmm_b1);
					a += block;
					b += block;
					i -= block;
				}
				while (i > 0)
				{
					*b++ = *a++ + offset;
					--i;
				}
			}
		}
	};

	// Class template kernel_sliding_block
	template<>
	struct kernel_sliding_block<unsigned __int64, cpu_sse2>
	{
		void operator()(unsigned __int64 *data, const unsigned __int64 step, const unsigned __int64 stride, const unsigned __int64 offset) const
		{
			constexpr unsigned __int64 block = 8;
			unsigned __int64 *a = data;
			unsigned __int64 *b = data + stride;
			__m128i xmm_val = _mm_set1_epi64x(offset);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			for (unsigned __int64 i, j = 1; j < step; ++j)
			{
				i = stride;
				while (i > block)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 2));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 4));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 6));
					// numerical migration
					xmm_b0 = _mm_add_epi64(xmm_a0, xmm_val);
					xmm_b1 = _mm_add_epi64(xmm_a1, xmm_val);
					xmm_b2 = _mm_add_epi64(xmm_a2, xmm_val);
					xmm_b3 = _mm_add_epi64(xmm_a3, xmm_val);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + 2), xmm_b1);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + 4), xmm_b2);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + 6), xmm_b3);
					a += block;
					b += block;
					i -= block;
				}
				while (i > 0)
				{
					*b++ = *a++ + offset;
					--i;
				}
			}
		}
	};

	// Class template kernel_sliding_block
	template<>
	struct kernel_sliding_block<unsigned int, cpu_avx2>
	{
		void operator()(unsigned int *data, const unsigned int step, const unsigned int stride, const unsigned int offset) const
		{
			constexpr unsigned int block = 16;
			unsigned int *a = data;
			unsigned int *b = data + stride;
			__m256i ymm_val = _mm256_set1_epi32(offset);
			__m256i ymm_a0, ymm_a1;
			__m256i ymm_b0, ymm_b1;

			for (unsigned __int64 i, j = 1; j < step; ++j)
			{
				i = stride;
				while (i > block)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 8));
					// numerical migration
					ymm_b0 = _mm256_add_epi32(ymm_a0, ymm_val);
					ymm_b1 = _mm256_add_epi32(ymm_a1, ymm_val);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + 8), ymm_b1);
					a += block;
					b += block;
					i -= block;
				}
				while (i > 0)
				{
					*b++ = *a++ + offset;
					--i;
				}
			}
		}
	};

	// Class template kernel_sliding_block
	template<>
	struct kernel_sliding_block<unsigned __int64, cpu_avx2>
	{
		void operator()(unsigned __int64 *data, const unsigned __int64 step, const unsigned __int64 stride, const unsigned __int64 offset) const
		{
			constexpr unsigned __int64 block = 2;
			const unsigned __int64 aligned = stride & ~(block - 1);
			unsigned __int64 *a = data;
			unsigned __int64 *b = data + stride;
			__m256i ymm_val = _mm256_set1_epi64x(offset);
			__m256i ymm_a;
			__m256i ymm_b;

			for (unsigned __int64 j = 1; j < step; ++j)
			{
				for (unsigned __int64 i = 0; i < aligned; i += block)
				{
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
					ymm_b = _mm256_add_epi64(ymm_a, ymm_val);
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + i), ymm_b);
				}
				for (unsigned __int64 i = aligned; i < stride; ++i)
				{
					b[i] = a[i] + offset;
				}
				a += stride;
				b += stride;
			}
		}
		//void operator()(unsigned __int64 *data, const unsigned __int64 step, const unsigned __int64 stride, const unsigned __int64 offset) const
		//{
		//	constexpr unsigned __int64 block = 16;
		//	unsigned __int64 *a = data;
		//	unsigned __int64 *b = data + stride;
		//	__m256i ymm_val = _mm256_set1_epi64x(offset);
		//	__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
		//	__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

		//	for (unsigned __int64 i, j = 1; j < step; ++j)
		//	{
		//		i = stride;
		//		while (i > block)
		//		{
		//			// load data from memory
		//			ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
		//			ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 4));
		//			ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 8));
		//			ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 12));
		//			// numerical migration
		//			ymm_b0 = _mm256_add_epi64(ymm_a0, ymm_val);
		//			ymm_b1 = _mm256_add_epi64(ymm_a1, ymm_val);
		//			ymm_b2 = _mm256_add_epi64(ymm_a2, ymm_val);
		//			ymm_b3 = _mm256_add_epi64(ymm_a3, ymm_val);
		//			// store data into memory
		//			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		//			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + 4), ymm_b1);
		//			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + 8), ymm_b2);
		//			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + 12), ymm_b3);
		//			a += block;
		//			b += block;
		//			i -= block;
		//		}
		//		while (i > 0)
		//		{
		//			*b++ = *a++ + offset;
		//			--i;
		//		}
		//	}
		//}
	};

	// Function template kernel_sliding_window
	template<class T, cpu_inst_type inst>
	void kernel_sliding_window(T *data, const T rows, const T columns, const T channels, const T window_h, const T window_w, const T stride_h, const T stride_w)
	{
		//T value = 0;
		//T stride = columns * channels;
		//T window_area = window_h * window_w;
		//T window_size = channels * window_area;
		//T height = 1 + (rows - window_h) / stride_h;
		//T width = 1 + (columns - window_w) / stride_w;
		//const struct kernel_sliding_block<T, inst> functor;

		T value, *src, *dst;
		T step_h = (rows - window_h) / stride_h;
		T step_w = (columns - window_w) / stride_w;
		T stride = columns * channels;
		T window_area = window_h * window_w;
		T window_size = channels * window_area;
		T sliding_size = (step_w + 1) * window_size;

		// 1 * window_w
		value = 0;
		for (T i = 0; i < window_w; ++i)
		{
			data[i] = value;
			value += channels;
		}
		// window_h * window_w
		src = data;
		dst = src + window_w;
		for (T i = 1; i < window_h; ++i)
		{
			for (T j = 0; j < window_w; ++j)
				dst[j] = src[j] + stride;
			src += window_w;
			dst += window_w;
		}
		// channels * window_h * window_w
		src = data;
		dst = src + window_area;
		for (T i = 1; i < channels; ++i)
		{
			for (T j = 0; j < window_area; ++j)
				dst[j] = src[j] + 1;
			src += window_area;
			dst += window_area;
		}
		// horizontal sliding window
		src = data;
		dst = src + window_size;
		value = stride_w * channels;
		for (T i = 0; i < step_w; ++i)
		{
			for (T j = 0; j < window_size; ++j)
				dst[j] = src[j] + value;
			src += window_size;
			dst += window_size;
		}
		// vertical sliding window
		src = data;
		dst = src + sliding_size;
		value = stride_h * stride;
		for (T i = 0; i < step_h; ++i)
		{
			for (T j = 0; j < sliding_size; ++j)
				dst[j] = src[j] + value;
			src += sliding_size;
			dst += sliding_size;
		}
	}

	//// Function template kernel_sliding_window
	//template<class T, cpu_inst_type inst>
	//void kernel_sliding_window(T *data, const T rows, const T columns, const T channels, const T window_h, const T window_w, const T stride_h, const T stride_w)
	//{
	//	T value = 0;
	//	T stride = columns * channels;
	//	T window_area = window_h * window_w;
	//	T window_size = channels * window_area;
	//	T height = 1 + (rows - window_h) / stride_h;
	//	T width = 1 + (columns - window_w) / stride_w;
	//	const struct kernel_sliding_block<T, inst> functor;

	//	// 1 * window_w
	//	for (T i = 0; i < window_w; ++i)
	//	{
	//		data[i] = value;
	//		value += channels;
	//	}
	//	// window_h * window_w
	//	functor(data, window_h, window_w, stride);
	//	// channels * window_h * window_w
	//	functor(data, channels, window_area, 1);
	//	// horizontal sliding window
	//	functor(data, width, window_size, stride_w * channels);
	//	// vertical sliding window
	//	functor(data, height, width * window_size, stride_h * stride);
	//}

	//// Function template kernel_sliding_window
	//template<class T>
	//void kernel_sliding_window(T *data, T rows, T columns, T channels, T window_h, T window_w, T stride_h, T stride_w)
	//{
	//	T value, *src, *dst;
	//	T step_h = (rows - window_h) / stride_h;
	//	T step_w = (columns - window_w) / stride_w;
	//	T stride = columns * channels;
	//	T window_area = window_h * window_w;
	//	T window_size = channels * window_area;
	//	T sliding_size = (step_w + 1) * window_size;

	//	// 1 * window_w
	//	value = 0;
	//	for (T i = 0; i < window_w; ++i)
	//	{
	//		data[i] = value;
	//		value += channels;
	//	}
	//	// window_h * window_w
	//	src = data;
	//	dst = src + window_w;
	//	for (T i = 1; i < window_h; ++i)
	//	{
	//		for (T j = 0; j < window_w; ++j)
	//			dst[j] = src[j] + stride;
	//		src += window_w;
	//		dst += window_w;
	//	}
	//	// channels * window_h * window_w
	//	src = data;
	//	dst = src + window_area;
	//	for (T i = 1; i < channels; ++i)
	//	{
	//		for (T j = 0; j < window_area; ++j)
	//			dst[j] = src[j] + 1;
	//		src += window_area;
	//		dst += window_area;
	//	}
	//	// horizontal sliding window
	//	src = data;
	//	dst = src + window_size;
	//	value = stride_w * channels;
	//	for (T i = 0; i < step_w; ++i)
	//	{
	//		for (T j = 0; j < window_size; ++j)
	//			dst[j] = src[j] + value;
	//		src += window_size;
	//		dst += window_size;
	//	}
	//	// vertical sliding window
	//	src = data;
	//	dst = src + sliding_size;
	//	value = stride_h * stride;
	//	for (T i = 0; i < step_h; ++i)
	//	{
	//		for (T j = 0; j < sliding_size; ++j)
	//			dst[j] = src[j] + value;
	//		src += sliding_size;
	//		dst += sliding_size;
	//	}
	//}

} // namespace core

#endif
