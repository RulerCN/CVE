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
	// Class template kernel_cvtmadd_double
	template<class T, cpu_inst_type inst>
	struct kernel_cvtmadd_double
	{
		void operator()(size_t n, double a, const T *b, double c, double *d) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				d[0] = a * static_cast<double>(b[0]) + c;
				d[1] = a * static_cast<double>(b[1]) + c;
				d[2] = a * static_cast<double>(b[2]) + c;
				d[3] = a * static_cast<double>(b[3]) + c;
				d[4] = a * static_cast<double>(b[4]) + c;
				d[5] = a * static_cast<double>(b[5]) + c;
				d[6] = a * static_cast<double>(b[6]) + c;
				d[7] = a * static_cast<double>(b[7]) + c;
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed char, cpu_sse41>
	{
		void operator()(size_t n, double a, const signed char *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;
			__m128d xmm_d0, xmm_d1, xmm_d2, xmm_d3, xmm_d4, xmm_d5, xmm_d6, xmm_d7;

			while (n >= block)
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
				xmm_d0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_d1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_d2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_d3 = _mm_cvtepi32_pd(xmm_b3);
				xmm_d4 = _mm_cvtepi32_pd(xmm_b4);
				xmm_d5 = _mm_cvtepi32_pd(xmm_b5);
				xmm_d6 = _mm_cvtepi32_pd(xmm_b6);
				xmm_d7 = _mm_cvtepi32_pd(xmm_b7);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d3), xmm_c);
				xmm_d4 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d4), xmm_c);
				xmm_d5 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d5), xmm_c);
				xmm_d6 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d6), xmm_c);
				xmm_d7 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d7), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				_mm_storeu_pd(d + 4, xmm_d2);
				_mm_storeu_pd(d + 6, xmm_d3);
				_mm_storeu_pd(d + 8, xmm_d4);
				_mm_storeu_pd(d + 10, xmm_d5);
				_mm_storeu_pd(d + 12, xmm_d6);
				_mm_storeu_pd(d + 14, xmm_d7);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, double a, const unsigned char *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;
			__m128d xmm_d0, xmm_d1, xmm_d2, xmm_d3, xmm_d4, xmm_d5, xmm_d6, xmm_d7;

			while (n >= block)
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
				xmm_d0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_d1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_d2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_d3 = _mm_cvtepi32_pd(xmm_b3);
				xmm_d4 = _mm_cvtepi32_pd(xmm_b4);
				xmm_d5 = _mm_cvtepi32_pd(xmm_b5);
				xmm_d6 = _mm_cvtepi32_pd(xmm_b6);
				xmm_d7 = _mm_cvtepi32_pd(xmm_b7);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d3), xmm_c);
				xmm_d4 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d4), xmm_c);
				xmm_d5 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d5), xmm_c);
				xmm_d6 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d6), xmm_c);
				xmm_d7 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d7), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				_mm_storeu_pd(d + 4, xmm_d2);
				_mm_storeu_pd(d + 6, xmm_d3);
				_mm_storeu_pd(d + 8, xmm_d4);
				_mm_storeu_pd(d + 10, xmm_d5);
				_mm_storeu_pd(d + 12, xmm_d6);
				_mm_storeu_pd(d + 14, xmm_d7);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed short, cpu_sse41>
	{
		void operator()(size_t n, double a, const signed short *b, double c, double *d) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi16_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepi16_epi32(xmm_b2);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_d0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_d1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_d2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_d3 = _mm_cvtepi32_pd(xmm_b3);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d3), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				_mm_storeu_pd(d + 4, xmm_d2);
				_mm_storeu_pd(d + 6, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, double a, const unsigned short *b, double c, double *d) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepu16_epi32(xmm_b0);
				xmm_b2 = _mm_cvtepu16_epi32(xmm_b2);
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b3 = _mm_shuffle_epi32(xmm_b2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_d0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_d1 = _mm_cvtepi32_pd(xmm_b1);
				xmm_d2 = _mm_cvtepi32_pd(xmm_b2);
				xmm_d3 = _mm_cvtepi32_pd(xmm_b3);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d3), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				_mm_storeu_pd(d + 4, xmm_d2);
				_mm_storeu_pd(d + 6, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed int, cpu_sse2>
	{
		void operator()(size_t n, double a, const signed int *b, double c, double *d) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1;
			__m128d xmm_d0, xmm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_d0 = _mm_cvtepi32_pd(xmm_b0);
				xmm_d1 = _mm_cvtepi32_pd(xmm_b1);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, double a, const unsigned int *b, double c, double *d) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi64x(0x41e0000000000000LL);
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_s0, xmm_s1;
			__m128d xmm_d0, xmm_dv0, xmm_ds0, xmm_d1, xmm_dv1, xmm_ds1;

			while (n >= block)
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
				xmm_d0 = _mm_castsi128_pd(_mm_and_si128(xmm_s0, val));
				xmm_d1 = _mm_castsi128_pd(_mm_and_si128(xmm_s1, val));
				xmm_d0 = _mm_add_pd(xmm_d0, _mm_cvtepi32_pd(xmm_b0));
				xmm_d1 = _mm_add_pd(xmm_d1, _mm_cvtepi32_pd(xmm_b1));
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<float, cpu_sse>
	{
		void operator()(size_t n, double a, const float *b, double c, double *d) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128 xmm_b0, xmm_b1;
			__m128d xmm_d0, xmm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// data-type conversion
				xmm_b1 = _mm_shuffle_ps(xmm_b0, xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_d0 = _mm_cvtps_pd(xmm_b0);
				xmm_d1 = _mm_cvtps_pd(xmm_b1);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_d1), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<double, cpu_sse2>
	{
		void operator()(size_t n, double a, const double *b, double c, double *d) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			const __m128d xmm_c = _mm_set1_pd(c);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b0), xmm_c);
				xmm_d1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b1), xmm_c);
				xmm_d2 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b2), xmm_c);
				xmm_d3 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b3), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				_mm_storeu_pd(d + 2, xmm_d1);
				_mm_storeu_pd(d + 4, xmm_d2);
				_mm_storeu_pd(d + 6, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				// d = a * b + c;
				xmm_d0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b0), xmm_c);
				// store data into memory
				_mm_storeu_pd(d, xmm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * b[i] + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed char, cpu_avx2>
	{
		void operator()(size_t n, double a, const signed char *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256i ymm_b0, ymm_b1;
			__m256d ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
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
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_d1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_d2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_d3 = _mm256_cvtepi32_pd(xmm_b3);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				ymm_d2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d2), ymm_c);
				ymm_d3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d3), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				_mm256_storeu_pd(d + 8, ymm_d2);
				_mm256_storeu_pd(d + 12, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned char *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256i ymm_b0, ymm_b1;
			__m256d ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
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
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_d1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_d2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_d3 = _mm256_cvtepi32_pd(xmm_b3);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				ymm_d2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d2), ymm_c);
				ymm_d3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d3), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				_mm256_storeu_pd(d + 8, ymm_d2);
				_mm256_storeu_pd(d + 12, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed short, cpu_avx2>
	{
		void operator()(size_t n, double a, const signed short *b, double c, double *d) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0;
			__m256d ymm_d0, ymm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_d1 = _mm256_cvtepi32_pd(xmm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				d += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned short *b, double c, double *d) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0;
			__m256d ymm_d0, ymm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				xmm_b0 = _mm256_extracti128_si256(ymm_b0, 0);
				xmm_b1 = _mm256_extracti128_si256(ymm_b0, 1);
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_d1 = _mm256_cvtepi32_pd(xmm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				d += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<signed int, cpu_avx>
	{
		void operator()(size_t n, double a, const signed int *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256d ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// data-type conversion
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				ymm_d1 = _mm256_cvtepi32_pd(xmm_b1);
				ymm_d2 = _mm256_cvtepi32_pd(xmm_b2);
				ymm_d3 = _mm256_cvtepi32_pd(xmm_b3);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				ymm_d2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d2), ymm_c);
				ymm_d3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d3), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				_mm256_storeu_pd(d + 8, ymm_d2);
				_mm256_storeu_pd(d + 12, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_d0 = _mm256_cvtepi32_pd(xmm_b0);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, double a, const unsigned int *b, double c, double *d) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi64x(0x41e0000000000000LL);
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128i xmm_b, xmm_s;
			__m256i ymm_s;
			__m256d ymm_d;

			while (n >= block)
			{
				// load data from memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_s = _mm_srai_epi32(xmm_b, 31);
				xmm_b = _mm_and_si128(xmm_b, abs);
				ymm_s = _mm256_cvtepi32_epi64(xmm_s);
				ymm_d = _mm256_castsi256_pd(_mm256_and_si256(ymm_s, val));
				ymm_d = _mm256_add_pd(ymm_d, _mm256_cvtepi32_pd(xmm_b));
				// d = a * b + c;
				ymm_d = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<float, cpu_avx>
	{
		void operator()(size_t n, double a, const float *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m256d ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// data-type conversion
				ymm_d0 = _mm256_cvtps_pd(xmm_b0);
				ymm_d1 = _mm256_cvtps_pd(xmm_b1);
				ymm_d2 = _mm256_cvtps_pd(xmm_b2);
				ymm_d3 = _mm256_cvtps_pd(xmm_b3);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d1), ymm_c);
				ymm_d2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d2), ymm_c);
				ymm_d3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d3), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				_mm256_storeu_pd(d + 8, ymm_d2);
				_mm256_storeu_pd(d + 12, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// data-type conversion
				ymm_d0 = _mm256_cvtps_pd(xmm_b0);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<double>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_double<double, cpu_avx>
	{
		void operator()(size_t n, double a, const double *b, double c, double *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			const __m256d ymm_c = _mm256_set1_pd(c);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b0), ymm_c);
				ymm_d1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b1), ymm_c);
				ymm_d2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b2), ymm_c);
				ymm_d3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b3), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				_mm256_storeu_pd(d + 4, ymm_d1);
				_mm256_storeu_pd(d + 8, ymm_d2);
				_mm256_storeu_pd(d + 12, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				// d = a * b + c;
				ymm_d0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b0), ymm_c);
				// store data into memory
				_mm256_storeu_pd(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * b[i] + c;
		}
	};

} // namespace core

#endif
