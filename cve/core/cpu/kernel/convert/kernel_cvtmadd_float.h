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

#ifndef __CORE_CPU_KERNEL_CVTMUL_FLOAT_H__
#define __CORE_CPU_KERNEL_CVTMUL_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_cvtmadd_float
	template<class T, cpu_inst_type inst>
	struct kernel_cvtmadd_float
	{
		void operator()(size_t n, float a, const T *b, float c, float *d) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				d[0] = a * static_cast<float>(b[0]) + c;
				d[1] = a * static_cast<float>(b[1]) + c;
				d[2] = a * static_cast<float>(b[2]) + c;
				d[3] = a * static_cast<float>(b[3]) + c;
				d[4] = a * static_cast<float>(b[4]) + c;
				d[5] = a * static_cast<float>(b[5]) + c;
				d[6] = a * static_cast<float>(b[6]) + c;
				d[7] = a * static_cast<float>(b[7]) + c;
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed char, cpu_sse41>
	{
		void operator()(size_t n, float a, const signed char *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, 1);
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, 2);
				xmm_b3 = _mm_shuffle_epi32(xmm_b0, 3);
				xmm_b0 = _mm_cvtepi8_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepi8_epi32(xmm_b1);
				xmm_b2 = _mm_cvtepi8_epi32(xmm_b2);
				xmm_b3 = _mm_cvtepi8_epi32(xmm_b3);
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				xmm_d1 = _mm_cvtepi32_ps(xmm_b1);
				xmm_d2 = _mm_cvtepi32_ps(xmm_b2);
				xmm_d3 = _mm_cvtepi32_ps(xmm_b3);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d3), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				_mm_storeu_ps(d + 8, xmm_d2);
				_mm_storeu_ps(d + 12, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, float a, const unsigned char *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, 1);
				xmm_b2 = _mm_shuffle_epi32(xmm_b0, 2);
				xmm_b3 = _mm_shuffle_epi32(xmm_b0, 3);
				xmm_b0 = _mm_cvtepu8_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepu8_epi32(xmm_b1);
				xmm_b2 = _mm_cvtepu8_epi32(xmm_b2);
				xmm_b3 = _mm_cvtepu8_epi32(xmm_b3);
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				xmm_d1 = _mm_cvtepi32_ps(xmm_b1);
				xmm_d2 = _mm_cvtepi32_ps(xmm_b2);
				xmm_d3 = _mm_cvtepi32_ps(xmm_b3);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d3), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				_mm_storeu_ps(d + 8, xmm_d2);
				_mm_storeu_ps(d + 12, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed short, cpu_sse41>
	{
		void operator()(size_t n, float a, const signed short *b, float c, float *d) const
		{
			constexpr size_t block = 8;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m128 xmm_d0, xmm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi16_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepi16_epi32(xmm_b1);
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				xmm_d1 = _mm_cvtepi32_ps(xmm_b1);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d1), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, float a, const unsigned short *b, float c, float *d) const
		{
			constexpr size_t block = 8;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m128 xmm_d0, xmm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepu16_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepu16_epi32(xmm_b1);
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				xmm_d1 = _mm_cvtepi32_ps(xmm_b1);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d1), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed int, cpu_sse2>
	{
		void operator()(size_t n, float a, const signed int *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// data-type conversion
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				xmm_d1 = _mm_cvtepi32_ps(xmm_b1);
				xmm_d2 = _mm_cvtepi32_ps(xmm_b2);
				xmm_d3 = _mm_cvtepi32_ps(xmm_b3);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d1), xmm_c);
				xmm_d2 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d2), xmm_c);
				xmm_d3 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d3), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				_mm_storeu_ps(d + 8, xmm_d2);
				_mm_storeu_ps(d + 12, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_d0 = _mm_cvtepi32_ps(xmm_b0);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, float a, const unsigned int *b, float c, float *d) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi32(0x4f000000);
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128i xmm_b, xmm_s;
			__m128 xmm_d;

			while (n >= block)
			{
				// load data from memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_s = _mm_srai_epi32(xmm_b, 31);
				xmm_b = _mm_and_si128(xmm_b, abs);
				xmm_d = _mm_castsi128_ps(_mm_and_si128(xmm_s, val));
				xmm_d = _mm_add_ps(xmm_d, _mm_cvtepi32_ps(xmm_b));
				// d = a * b + c;
				xmm_d = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<float, cpu_sse>
	{
		void operator()(size_t n, float a, const float *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_d0, xmm_d1, xmm_d2, xmm_d3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b0), xmm_c);
				xmm_d1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b1), xmm_c);
				xmm_d2 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b2), xmm_c);
				xmm_d3 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b3), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				_mm_storeu_ps(d + 4, xmm_d1);
				_mm_storeu_ps(d + 8, xmm_d2);
				_mm_storeu_ps(d + 12, xmm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b0), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * b[i];
		}
	};

	template<>
	struct kernel_cvtmadd_float<double, cpu_sse2>
	{
		void operator()(size_t n, float a, const double *b, float c, float *d) const
		{
			constexpr size_t block = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			const __m128 xmm_c = _mm_set1_ps(c);
			__m128d xmm_b0, xmm_b1;
			__m128 xmm_d0, xmm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				// data-type conversion
				xmm_d0 = _mm_cvtpd_ps(xmm_b0);
				xmm_d1 = _mm_cvtpd_ps(xmm_b1);
				xmm_d0 = _mm_movelh_ps(xmm_d0, xmm_d1);
				// d = a * b + c;
				xmm_d0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_d0), xmm_c);
				// store data into memory
				_mm_storeu_ps(d, xmm_d0);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed char, cpu_avx2>
	{
		void operator()(size_t n, float a, const signed char *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256 ymm_d0, ymm_d1;

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
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				ymm_d1 = _mm256_cvtepi32_ps(ymm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, float a, const unsigned char *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256 ymm_d0, ymm_d1;

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
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				ymm_d1 = _mm256_cvtepi32_ps(ymm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed short, cpu_avx2>
	{
		void operator()(size_t n, float a, const signed short *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256 ymm_d0, ymm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_b1);
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				ymm_d1 = _mm256_cvtepi32_ps(ymm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				d += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, float a, const unsigned short *b, float c, float *d) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256 ymm_d0, ymm_d1;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepu16_epi32(xmm_b1);
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				ymm_d1 = _mm256_cvtepi32_ps(ymm_b1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d1), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				d += block;
				b += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<signed int, cpu_avx>
	{
		void operator()(size_t n, float a, const signed int *b, float c, float *d) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256 ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// data-type conversion
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				ymm_d1 = _mm256_cvtepi32_ps(ymm_b1);
				ymm_d2 = _mm256_cvtepi32_ps(ymm_b2);
				ymm_d3 = _mm256_cvtepi32_ps(ymm_b3);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d1), ymm_c);
				ymm_d2 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d2), ymm_c);
				ymm_d3 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d3), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				_mm256_storeu_ps(d + 16, ymm_d2);
				_mm256_storeu_ps(d + 24, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// data-type conversion
				ymm_d0 = _mm256_cvtepi32_ps(ymm_b0);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, float a, const unsigned int *b, float c, float *d) const
		{
			constexpr size_t block = 8;
			const __m256i abs = _mm256_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi32(0x4f000000);
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m256i ymm_b, ymm_s;
			__m256 ymm_d;

			while (n >= block)
			{
				// load data from memory
				ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// data-type conversion
				ymm_s = _mm256_srai_epi32(ymm_b, 31);
				ymm_b = _mm256_and_si256(ymm_b, abs);
				ymm_d = _mm256_castsi256_ps(_mm256_and_si256(ymm_s, val));
				ymm_d = _mm256_add_ps(ymm_d, _mm256_cvtepi32_ps(ymm_b));
				// d = a * b + c;
				ymm_d = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<float, cpu_avx>
	{
		void operator()(size_t n, float a, const float *b, float c, float *d) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256 ymm_d0, ymm_d1, ymm_d2, ymm_d3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b0), ymm_c);
				ymm_d1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b1), ymm_c);
				ymm_d2 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b2), ymm_c);
				ymm_d3 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b3), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				_mm256_storeu_ps(d + 8, ymm_d1);
				_mm256_storeu_ps(d + 16, ymm_d2);
				_mm256_storeu_ps(d + 24, ymm_d3);
				b += block;
				d += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b0), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				b += bit;
				d += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * b[i] + c;
		}
	};

	template<>
	struct kernel_cvtmadd_float<double, cpu_avx>
	{
		void operator()(size_t n, float a, const double *b, float c, float *d) const
		{
			constexpr size_t block = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			const __m256 ymm_c = _mm256_set1_ps(c);
			__m128 xmm_b0, xmm_b1;
			__m256d ymm_b0, ymm_b1;
			__m256 ymm_d0;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				// data-type conversion
				xmm_b0 = _mm256_cvtpd_ps(ymm_b0);
				xmm_b1 = _mm256_cvtpd_ps(ymm_b1);
				ymm_d0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm_b0), xmm_b1, 1);
				// d = a * b + c;
				ymm_d0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_d0), ymm_c);
				// store data into memory
				_mm256_storeu_ps(d, ymm_d0);
				b += block;
				d += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				d[i] = a * static_cast<float>(b[i]) + c;
		}
	};

} // namespace core

#endif
