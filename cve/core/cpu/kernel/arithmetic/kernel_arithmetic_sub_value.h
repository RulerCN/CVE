/*====================================================================
Copyright (C) 2016-2016 Ruler. All rights reserved.
Author:  Ruler
Address: Nan'an District,Chongqing,China
Contact: 26105499@qq.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. The name of the author may not be used to endorse or promote
products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY RULER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_SUB_VALUE_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_SUB_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_sub_value

	template<class T, cpu_inst_type inst>
	struct kernel_sub_value
	{
		void operator()(size_t n, const T a, T *b) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				b[0] = a - b[0];
				b[1] = a - b[1];
				b[2] = a - b[2];
				b[3] = a - b[3];
				b[4] = a - b[4];
				b[5] = a - b[5];
				b[6] = a - b[6];
				b[7] = a - b[7];
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed char, cpu_sse2>
	{
		void operator()(size_t n, const signed char a, signed char *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi8(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi8(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi8(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi8(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi8(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, const unsigned char a, unsigned char *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi8(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi8(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi8(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi8(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi8(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed short, cpu_sse2>
	{
		void operator()(size_t n, const signed short a, signed short *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi16(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi16(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi16(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi16(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi16(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, const unsigned short a, unsigned short *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi16(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi16(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi16(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi16(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi16(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int a, signed int *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi32(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi32(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi32(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi32(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, const unsigned int a, unsigned int *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b) + 3);
				// b = a - b;
				xmm_b0 = _mm_sub_epi32(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_epi32(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_epi32(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_epi32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 2, xmm_b2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 3, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b));
				// b = a - b;
				xmm_b0 = _mm_sub_epi32(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<float, cpu_sse>
	{
		void operator()(size_t n, const float a, float *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// b = a - b;
				xmm_b0 = _mm_sub_ps(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_ps(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_ps(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_ps(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				_mm_storeu_ps(b + 4, xmm_b1);
				_mm_storeu_ps(b + 8, xmm_b2);
				_mm_storeu_ps(b + 12, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// b = a - b;
				xmm_b0 = _mm_sub_ps(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<double, cpu_sse2>
	{
		void operator()(size_t n, const double a, double *b) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				// b = a - b;
				xmm_b0 = _mm_sub_pd(xmm_a, xmm_b0);
				xmm_b1 = _mm_sub_pd(xmm_a, xmm_b1);
				xmm_b2 = _mm_sub_pd(xmm_a, xmm_b2);
				xmm_b3 = _mm_sub_pd(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				// b = a - b;
				xmm_b0 = _mm_sub_pd(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char a, signed char *b) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi8(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi8(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi8(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi8(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi8(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char a, unsigned char *b) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi8(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi8(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi8(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi8(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi8(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short a, signed short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi16(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi16(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi16(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi16(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi16(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short a, unsigned short *b) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi16(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi16(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi16(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi16(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi16(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed int a, signed int *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_min = _mm256_set1_epi32(int32_min);
			const __m256i ymm_max = _mm256_set1_epi32(int32_max);
			const __m256i ymm_sign = _mm256_set1_epi32(int32_sign);
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi32(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi32(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi32(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int a, unsigned int *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_max = _mm256_set1_epi32(uint32_max);
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b) + 3);
				// b = a - b;
				ymm_b0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_epi32(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_epi32(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_epi32(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 2, ymm_b2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 3, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b));
				// b = a - b;
				ymm_b0 = _mm256_sub_epi32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<float, cpu_avx>
	{
		void operator()(size_t n, const float a, float *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				// b = a - b;
				ymm_b0 = _mm256_sub_ps(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_ps(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_ps(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_ps(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				_mm256_storeu_ps(b + 8, ymm_b1);
				_mm256_storeu_ps(b + 16, ymm_b2);
				_mm256_storeu_ps(b + 24, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				// b = a - b;
				ymm_b0 = _mm256_sub_ps(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

	template<>
	struct kernel_sub_value<double, cpu_avx>
	{
		void operator()(size_t n, const double a, double *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				// b = a - b;
				ymm_b0 = _mm256_sub_pd(ymm_a, ymm_b0);
				ymm_b1 = _mm256_sub_pd(ymm_a, ymm_b1);
				ymm_b2 = _mm256_sub_pd(ymm_a, ymm_b2);
				ymm_b3 = _mm256_sub_pd(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				// b = a - b;
				ymm_b0 = _mm256_sub_pd(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = a - b[i];
		}
	};

} // namespace core

#endif
