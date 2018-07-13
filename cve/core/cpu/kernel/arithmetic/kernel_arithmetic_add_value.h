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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_ADD_VALUE_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_ADD_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_add_value

	template<class T, cpu_inst_type inst>
	struct kernel_add_value
	{
		void operator()(size_t n, const T *a, const T b, T *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a[0] + b;
				c[1] = a[1] + b;
				c[2] = a[2] + b;
				c[3] = a[3] + b;
				c[4] = a[4] + b;
				c[5] = a[5] + b;
				c[6] = a[6] + b;
				c[7] = a[7] + b;
				a += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed char, cpu_sse2>
	{
		void operator()(size_t n, const signed char *a, const signed char b, signed char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_b = _mm_set1_epi8(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi8(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi8(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi8(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi8(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi8(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, const unsigned char *a, const unsigned char b, unsigned char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_b = _mm_set1_epi8(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi8(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi8(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi8(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi8(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi8(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed short, cpu_sse2>
	{
		void operator()(size_t n, const signed short *a, const signed short b, signed short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_b = _mm_set1_epi16(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi16(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi16(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi16(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi16(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi16(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, const unsigned short *a, const unsigned short b, unsigned short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_b = _mm_set1_epi16(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi16(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi16(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi16(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi16(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi16(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, const signed int b, signed int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_b = _mm_set1_epi32(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, const unsigned int *a, const unsigned int b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_b = _mm_set1_epi32(b);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_epi32(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_epi32(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_epi32(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// c = a + b;
				xmm_c0 = _mm_add_epi32(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<float, cpu_sse>
	{
		void operator()(size_t n, const float *a, const float b, float *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_b = _mm_set1_ps(b);
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				xmm_a1 = _mm_loadu_ps(a + 4);
				xmm_a2 = _mm_loadu_ps(a + 8);
				xmm_a3 = _mm_loadu_ps(a + 12);
				// c = a + b;
				xmm_c0 = _mm_add_ps(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_ps(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_ps(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_ps(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				_mm_storeu_ps(c + 4, xmm_c1);
				_mm_storeu_ps(c + 8, xmm_c2);
				_mm_storeu_ps(c + 12, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				// c = a + b;
				xmm_c0 = _mm_add_ps(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<double, cpu_sse2>
	{
		void operator()(size_t n, const double *a, const double b, double *c) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_b = _mm_set1_pd(b);
			__m128d xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				xmm_a1 = _mm_loadu_pd(a + 2);
				xmm_a2 = _mm_loadu_pd(a + 4);
				xmm_a3 = _mm_loadu_pd(a + 6);
				// c = a + b;
				xmm_c0 = _mm_add_pd(xmm_a0, xmm_b);
				xmm_c1 = _mm_add_pd(xmm_a1, xmm_b);
				xmm_c2 = _mm_add_pd(xmm_a2, xmm_b);
				xmm_c3 = _mm_add_pd(xmm_a3, xmm_b);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				// c = a + b;
				xmm_c0 = _mm_add_pd(xmm_a0, xmm_b);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, const signed char b, signed char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_b = _mm256_set1_epi8(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi8(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi8(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi8(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi8(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi8(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char *a, const unsigned char b, unsigned char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_b = _mm256_set1_epi8(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi8(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi8(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi8(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi8(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi8(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short *a, const signed short b, signed short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_b = _mm256_set1_epi16(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi16(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi16(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi16(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi16(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi16(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short *a, const unsigned short b, unsigned short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_b = _mm256_set1_epi16(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi16(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi16(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi16(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi16(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi16(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed int *a, const signed int b, signed int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_b = _mm256_set1_epi32(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int *a, const unsigned int b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_b = _mm256_set1_epi32(b);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 1);
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 2);
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a) + 3);
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_epi32(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_epi32(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_epi32(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
				// c = a + b;
				ymm_c0 = _mm256_add_epi32(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<float, cpu_avx>
	{
		void operator()(size_t n, const float *a, const float b, float *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_b = _mm256_set1_ps(b);
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(a);
				ymm_a1 = _mm256_loadu_ps(a + 8);
				ymm_a2 = _mm256_loadu_ps(a + 16);
				ymm_a3 = _mm256_loadu_ps(a + 24);
				// c = a + b;
				ymm_c0 = _mm256_add_ps(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_ps(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_ps(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_ps(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				_mm256_storeu_ps(c + 8, ymm_c1);
				_mm256_storeu_ps(c + 16, ymm_c2);
				_mm256_storeu_ps(c + 24, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(a);
				// c = a + b;
				ymm_c0 = _mm256_add_ps(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

	template<>
	struct kernel_add_value<double, cpu_avx>
	{
		void operator()(size_t n, const double *a, const double b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_b = _mm256_set1_pd(b);
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				ymm_a1 = _mm256_loadu_pd(a + 4);
				ymm_a2 = _mm256_loadu_pd(a + 8);
				ymm_a3 = _mm256_loadu_pd(a + 12);
				// c = a + b;
				ymm_c0 = _mm256_add_pd(ymm_a0, ymm_b);
				ymm_c1 = _mm256_add_pd(ymm_a1, ymm_b);
				ymm_c2 = _mm256_add_pd(ymm_a2, ymm_b);
				ymm_c3 = _mm256_add_pd(ymm_a3, ymm_b);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				a += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				// c = a + b;
				ymm_c0 = _mm256_add_pd(ymm_a0, ymm_b);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				a += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] + b;
		}
	};

} // namespace core

#endif
