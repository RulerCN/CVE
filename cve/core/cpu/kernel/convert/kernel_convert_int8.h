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

#ifndef __CORE_CPU_KERNEL_CONVERT_INT8_H__
#define __CORE_CPU_KERNEL_CONVERT_INT8_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_convert_int8
	template<class T, cpu_inst_type inst>
	struct kernel_convert_int8
	{
		void operator()(size_t n, const signed char *a, T *b) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				b[0] = static_cast<T>(a[0]);
				b[1] = static_cast<T>(a[1]);
				b[2] = static_cast<T>(a[2]);
				b[3] = static_cast<T>(a[3]);
				b[4] = static_cast<T>(a[4]);
				b[5] = static_cast<T>(a[5]);
				b[6] = static_cast<T>(a[6]);
				b[7] = static_cast<T>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<T>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<signed short, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, signed short *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi8_epi16(xmm_a0);
				xmm_b1 = _mm_cvtepi8_epi16(xmm_a1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<signed short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, unsigned short *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0, xmm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi8_epi16(xmm_a0);
				xmm_b1 = _mm_cvtepi8_epi16(xmm_a1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<unsigned short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<signed int, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, signed int *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, 1);
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, 2);
				xmm_a3 = _mm_shuffle_epi32(xmm_a0, 3);
				xmm_b0 = _mm_cvtepi8_epi32(xmm_a0);
				xmm_b1 = _mm_cvtepi8_epi32(xmm_a1);
				xmm_b2 = _mm_cvtepi8_epi32(xmm_a2);
				xmm_b3 = _mm_cvtepi8_epi32(xmm_a3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<signed int>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<unsigned int, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, unsigned int *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, 1);
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, 2);
				xmm_a3 = _mm_shuffle_epi32(xmm_a0, 3);
				xmm_b0 = _mm_cvtepi8_epi32(xmm_a0);
				xmm_b1 = _mm_cvtepi8_epi32(xmm_a1);
				xmm_b2 = _mm_cvtepi8_epi32(xmm_a2);
				xmm_b3 = _mm_cvtepi8_epi32(xmm_a3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<unsigned int>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<float, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, float *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, 1);
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, 2);
				xmm_a3 = _mm_shuffle_epi32(xmm_a0, 3);
				xmm_a0 = _mm_cvtepi8_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi8_epi32(xmm_a1);
				xmm_a2 = _mm_cvtepi8_epi32(xmm_a2);
				xmm_a3 = _mm_cvtepi8_epi32(xmm_a3);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a1);
				xmm_b2 = _mm_cvtepi32_ps(xmm_a2);
				xmm_b3 = _mm_cvtepi32_ps(xmm_a3);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				_mm_storeu_ps(b + 4, xmm_b1);
				_mm_storeu_ps(b + 8, xmm_b2);
				_mm_storeu_ps(b + 12, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<float>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<double, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, double *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, 1);
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, 2);
				xmm_a6 = _mm_shuffle_epi32(xmm_a0, 3);
				xmm_a0 = _mm_cvtepi8_epi32(xmm_a0);
				xmm_a2 = _mm_cvtepi8_epi32(xmm_a2);
				xmm_a4 = _mm_cvtepi8_epi32(xmm_a4);
				xmm_a6 = _mm_cvtepi8_epi32(xmm_a6);
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a5 = _mm_shuffle_epi32(xmm_a4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a7 = _mm_shuffle_epi32(xmm_a6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi32_pd(xmm_a0);
				xmm_b1 = _mm_cvtepi32_pd(xmm_a1);
				xmm_b2 = _mm_cvtepi32_pd(xmm_a2);
				xmm_b3 = _mm_cvtepi32_pd(xmm_a3);
				xmm_b4 = _mm_cvtepi32_pd(xmm_a4);
				xmm_b5 = _mm_cvtepi32_pd(xmm_a5);
				xmm_b6 = _mm_cvtepi32_pd(xmm_a6);
				xmm_b7 = _mm_cvtepi32_pd(xmm_a7);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
				_mm_storeu_pd(b + 8, xmm_b4);
				_mm_storeu_pd(b + 10, xmm_b5);
				_mm_storeu_pd(b + 12, xmm_b6);
				_mm_storeu_pd(b + 14, xmm_b7);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<double>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, signed short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				ymm_b1 = _mm256_cvtepi8_epi16(xmm_a1);
				ymm_b2 = _mm256_cvtepi8_epi16(xmm_a2);
				ymm_b3 = _mm256_cvtepi8_epi16(xmm_a3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<signed short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, unsigned short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				ymm_b1 = _mm256_cvtepi8_epi16(xmm_a1);
				ymm_b2 = _mm256_cvtepi8_epi16(xmm_a2);
				ymm_b3 = _mm256_cvtepi8_epi16(xmm_a3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<unsigned short>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, signed int *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_b0, ymm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_a1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<signed int>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, unsigned int *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_b0, ymm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_a1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<unsigned int>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<float, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, float *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0, ymm_a1;
			__m256 ymm_b0, ymm_b1;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				ymm_b1 = _mm256_cvtepi32_ps(ymm_a1);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				_mm256_storeu_ps(b + 8, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<float>(a[i]);
		}
	};

	template<>
	struct kernel_convert_int8<double, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, double *b) const
		{
			constexpr size_t block = 16;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_a0, ymm_a1;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n >= block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				xmm_a2 = _mm256_extracti128_si256(ymm_a1, 0);
				xmm_a3 = _mm256_extracti128_si256(ymm_a1, 1);
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				ymm_b1 = _mm256_cvtepi32_pd(xmm_a1);
				ymm_b2 = _mm256_cvtepi32_pd(xmm_a2);
				ymm_b3 = _mm256_cvtepi32_pd(xmm_a3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = static_cast<double>(a[i]);
		}
	};

} // namespace core

#endif
