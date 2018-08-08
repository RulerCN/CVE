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

#ifndef __CORE_CPU_KERNEL_CVTMUL_DOUBLE_H__
#define __CORE_CPU_KERNEL_CVTMUL_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_cvtmul_double
	template<class T, cpu_inst_type inst>
	struct kernel_cvtmul_double
	{
		void operator()(size_t n, double a, const T *b, double *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a * static_cast<double>(b[0]);
				c[1] = a * static_cast<double>(b[1]);
				c[2] = a * static_cast<double>(b[2]);
				c[3] = a * static_cast<double>(b[3]);
				c[4] = a * static_cast<double>(b[4]);
				c[5] = a * static_cast<double>(b[5]);
				c[6] = a * static_cast<double>(b[6]);
				c[7] = a * static_cast<double>(b[7]);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<signed char, cpu_sse41>
	{
		void operator()(size_t n, double a, const signed char *b, double *c) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3, xmm_c4, xmm_c5, xmm_c6, xmm_c7;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, 1);
				xmm_b4 = _mm_shuffle_epi32(xmm_b0, 2);
				xmm_b6 = _mm_shuffle_epi32(xmm_b0, 3);
				xmm_b0 = _mm_cvtepi8_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepi8_epi32(xmm_b2);
				xmm_b4 = _mm_cvtepi8_epi32(xmm_b4);
				xmm_b6 = _mm_cvtepi8_epi32(xmm_b6);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b5 = _mm_shuffle_epi32(xmm_b4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b7 = _mm_shuffle_epi32(xmm_b6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_c1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_c2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_c3 = _mm_cvtepi32_pd(xmm_b3);
				xmm_c4 = _mm_cvtepi32_pd(xmm_b4);
				xmm_c5 = _mm_cvtepi32_pd(xmm_b5);
				xmm_c6 = _mm_cvtepi32_pd(xmm_b6);
				xmm_c7 = _mm_cvtepi32_pd(xmm_b7);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_pd(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_pd(xmm_a, xmm_c3);
				xmm_c4 = _mm_mul_pd(xmm_a, xmm_c4);
				xmm_c5 = _mm_mul_pd(xmm_a, xmm_c5);
				xmm_c6 = _mm_mul_pd(xmm_a, xmm_c6);
				xmm_c7 = _mm_mul_pd(xmm_a, xmm_c7);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				_mm_storeu_pd(c + 8, xmm_c4);
				_mm_storeu_pd(c + 10, xmm_c5);
				_mm_storeu_pd(c + 12, xmm_c6);
				_mm_storeu_pd(c + 14, xmm_c7);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, double a, const unsigned char *b, double *c) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3, xmm_c4, xmm_c5, xmm_c6, xmm_c7;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, 1);
				xmm_b4 = _mm_shuffle_epi32(xmm_b0, 2);
				xmm_b6 = _mm_shuffle_epi32(xmm_b0, 3);
				xmm_b0 = _mm_cvtepu8_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepu8_epi32(xmm_b2);
				xmm_b4 = _mm_cvtepu8_epi32(xmm_b4);
				xmm_b6 = _mm_cvtepu8_epi32(xmm_b6);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b5 = _mm_shuffle_epi32(xmm_b4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b7 = _mm_shuffle_epi32(xmm_b6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_c1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_c2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_c3 = _mm_cvtepi32_pd(xmm_b3);
				xmm_c4 = _mm_cvtepi32_pd(xmm_b4);
				xmm_c5 = _mm_cvtepi32_pd(xmm_b5);
				xmm_c6 = _mm_cvtepi32_pd(xmm_b6);
				xmm_c7 = _mm_cvtepi32_pd(xmm_b7);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_pd(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_pd(xmm_a, xmm_c3);
				xmm_c4 = _mm_mul_pd(xmm_a, xmm_c4);
				xmm_c5 = _mm_mul_pd(xmm_a, xmm_c5);
				xmm_c6 = _mm_mul_pd(xmm_a, xmm_c6);
				xmm_c7 = _mm_mul_pd(xmm_a, xmm_c7);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				_mm_storeu_pd(c + 8, xmm_c4);
				_mm_storeu_pd(c + 10, xmm_c5);
				_mm_storeu_pd(c + 12, xmm_c6);
				_mm_storeu_pd(c + 14, xmm_c7);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<signed short, cpu_sse41>
	{
		void operator()(size_t n, double a, const signed short *b, double *c) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi16_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepi16_epi32(xmm_b2);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_c1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_c2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_c3 = _mm_cvtepi32_pd(xmm_b3);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_pd(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_pd(xmm_a, xmm_c3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, double a, const unsigned short *b, double *c) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepu16_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepu16_epi32(xmm_b2);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_c1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_c2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_c3 = _mm_cvtepi32_pd(xmm_b3);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_pd(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_pd(xmm_a, xmm_c3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<signed int, cpu_sse2>
	{
		void operator()(size_t n, double a, const signed int *b, double *c) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_c1 = _mm_cvtepi32_pd(xmm_b1);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, double a, const unsigned int *b, double *c) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi64x(0x41e0000000000000LL);
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_s0, xmm_s1;
			__m128d xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_s0 = _mm_srai_epi32(xmm_b0, 31);
				xmm_b0 = _mm_and_si128(xmm_b0, abs);
				xmm_s1 = _mm_shuffle_epi32(xmm_s0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_s0 = _mm_cvtepi32_epi64(xmm_s0);
				xmm_s1 = _mm_cvtepi32_epi64(xmm_s1);
				xmm_c0 = _mm_castsi128_pd(_mm_and_si128(xmm_s0, val));
				xmm_c1 = _mm_castsi128_pd(_mm_and_si128(xmm_s1, val));
				xmm_c0 = _mm_add_pd(xmm_c0, _mm_cvtepi32_pd(xmm_b0));
				xmm_c1 = _mm_add_pd(xmm_c1, _mm_cvtepi32_pd(xmm_b1));
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<float, cpu_sse>
	{
		void operator()(size_t n, double a, const float *b, double *c) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128 xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// data-type conversion
				xmm_b1 = _mm_shuffle_ps(xmm_b0, xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_c0 = _mm_cvtps_pd(xmm_b0);
				xmm_c1 = _mm_cvtps_pd(xmm_b1);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<double, cpu_sse2>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_pd(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_pd(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_pd(xmm_a, xmm_c3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				// c = a * c;
				xmm_c0 = _mm_mul_pd(xmm_a, xmm_c0);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i];
		}
	};

	template<>
	struct kernel_cvtmul_double<signed char, cpu_avx2>
	{
		void operator()(size_t n, double a, const signed char *b, double *c) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256i ymm_b0, ymm_b1;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi8_epi16(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_b1);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				xmm_b2 = _mm256_extracti128_si256(ymm_b1, 0);
				xmm_b3 = _mm256_extracti128_si256(ymm_b1, 1);
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_c1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_c2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_c3 = _mm256_cvtepi32_pd(xmm_b3);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_pd(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_pd(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned char *b, double *c) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256i ymm_b0, ymm_b1;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepu8_epi16(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_b1);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				xmm_b2 = _mm256_extracti128_si256(ymm_b1, 0);
				xmm_b3 = _mm256_extracti128_si256(ymm_b1, 1);
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_c1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_c2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_c3 = _mm256_cvtepi32_pd(xmm_b3);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_pd(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_pd(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<signed short, cpu_avx2>
	{
		void operator()(size_t n, double a, const signed short *b, double *c) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0;
			__m256d ymm_c0, ymm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_c1 = _mm256_cvtepi32_pd(xmm_b1);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				c += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned short *b, double *c) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0;
			__m256d ymm_c0, ymm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_c1 = _mm256_cvtepi32_pd(xmm_b1);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				c += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<signed int, cpu_avx>
	{
		void operator()(size_t n, double a, const signed int *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// data-type conversion
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_c1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_c2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_c3 = _mm256_cvtepi32_pd(xmm_b3);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_pd(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_pd(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_c0 = _mm256_cvtepi32_pd(xmm_b0);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned int *b, double *c) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi64x(0x41e0000000000000LL);
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128i xmm_b, xmm_s;
			__m256i ymm_s;
			__m256d ymm_c;

			while (n > block)
			{
				// load data from memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_s = _mm_srai_epi32(xmm_b, 31);
				xmm_b = _mm_and_si128(xmm_b, abs);
				ymm_s = _mm256_cvtepi32_epi64(xmm_s);
				ymm_c = _mm256_castsi256_pd(_mm256_and_si256(ymm_s, val));
				ymm_c = _mm256_add_pd(ymm_c, _mm256_cvtepi32_pd(xmm_b));
				// c = a * c;
				ymm_c = _mm256_mul_pd(ymm_a, ymm_c);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<float, cpu_avx>
	{
		void operator()(size_t n, double a, const float *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// data-type conversion
				ymm_c0 = _mm256_cvtps_pd(xmm_b0);
				ymm_c1 = _mm256_cvtps_pd(xmm_b1);
				ymm_c2 = _mm256_cvtps_pd(xmm_b2);
				ymm_c3 = _mm256_cvtps_pd(xmm_b3);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_pd(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_pd(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// data-type conversion
				ymm_c0 = _mm256_cvtps_pd(xmm_b0);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<double>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_double<double, cpu_avx>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_pd(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_pd(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_pd(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				// c = a * c;
				ymm_c0 = _mm256_mul_pd(ymm_a, ymm_c0);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i];
		}
	};

} // namespace core

#endif
