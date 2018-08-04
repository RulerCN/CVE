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

#ifndef __CORE_CPU_KERNEL_CVTMUL_UINT32_H__
#define __CORE_CPU_KERNEL_CVTMUL_UINT32_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_cvtmul_uint32
	template<class T, cpu_inst_type inst>
	struct kernel_cvtmul_uint32
	{
		void operator()(size_t n, unsigned int a, const T *b, unsigned int *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a * static_cast<unsigned int>(b[0]);
				c[1] = a * static_cast<unsigned int>(b[1]);
				c[2] = a * static_cast<unsigned int>(b[2]);
				c[3] = a * static_cast<unsigned int>(b[3]);
				c[4] = a * static_cast<unsigned int>(b[4]);
				c[5] = a * static_cast<unsigned int>(b[5]);
				c[6] = a * static_cast<unsigned int>(b[6]);
				c[7] = a * static_cast<unsigned int>(b[7]);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed char, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const signed char *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
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
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				xmm_c2 = _mm_mul_epu32(xmm_a, xmm_b2);
				xmm_c3 = _mm_mul_epu32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const unsigned char *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
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
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				xmm_c2 = _mm_mul_epu32(xmm_a, xmm_b2);
				xmm_c3 = _mm_mul_epu32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed short, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const signed short *b, unsigned int *c) const
		{
			constexpr size_t block = 8;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m128i xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi16_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepi16_epi32(xmm_b1);
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const unsigned short *b, unsigned int *c) const
		{
			constexpr size_t block = 8;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m128i xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				xmm_b1 = _mm_shuffle_epi32(xmm_b0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepu16_epi32(xmm_b0);
				xmm_b1 = _mm_cvtepu16_epi32(xmm_b1);
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed int, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const signed int *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				xmm_c2 = _mm_mul_epu32(xmm_a, xmm_b2);
				xmm_c3 = _mm_mul_epu32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned int, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_b1);
				xmm_c2 = _mm_mul_epu32(xmm_a, xmm_b2);
				xmm_c3 = _mm_mul_epu32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a * b;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i];
		}
	};

	template<>
	struct kernel_cvtmul_uint32<float, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const float *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// data-type conversion
				xmm_c0 = _mm_cvtps_epi32(xmm_b0);
				xmm_c1 = _mm_cvtps_epi32(xmm_b1);
				xmm_c2 = _mm_cvtps_epi32(xmm_b2);
				xmm_c3 = _mm_cvtps_epi32(xmm_b3);
				// c = a * c;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_c0);
				xmm_c1 = _mm_mul_epu32(xmm_a, xmm_c1);
				xmm_c2 = _mm_mul_epu32(xmm_a, xmm_c2);
				xmm_c3 = _mm_mul_epu32(xmm_a, xmm_c3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// data-type conversion
				xmm_c0 = _mm_cvtps_epi32(xmm_b0);
				// c = a * c;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_c0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<double, cpu_sse41>
	{
		void operator()(size_t n, unsigned int a, const double *b, unsigned int *c) const
		{
			constexpr size_t block = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128d xmm_b0, xmm_b1;
			__m128i xmm_c0, xmm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				// data-type conversion
				xmm_c0 = _mm_cvtpd_epi32(xmm_b0);
				xmm_c1 = _mm_cvtpd_epi32(xmm_b1);
				xmm_c0 = _mm_unpacklo_epi64(xmm_c0, xmm_c1);
				// c = a * c;
				xmm_c0 = _mm_mul_epu32(xmm_a, xmm_c0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed char, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const signed char *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256i ymm_c0, ymm_c1;

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
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const unsigned char *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256i ymm_c0, ymm_c1;

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
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed short, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const signed short *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256i ymm_c0, ymm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepi16_epi32(xmm_b1);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				c += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi16_epi32(xmm_b0);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const unsigned short *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m256i ymm_b0, ymm_b1;
			__m256i ymm_c0, ymm_c1;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				ymm_b1 = _mm256_cvtepu16_epi32(xmm_b1);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				c += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// data-type conversion
				ymm_b0 = _mm256_cvtepu16_epi32(xmm_b0);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<signed int, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const signed int *b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				ymm_c2 = _mm256_mul_epu32(ymm_a, ymm_b2);
				ymm_c3 = _mm256_mul_epu32(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const signed int *b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_b1);
				ymm_c2 = _mm256_mul_epu32(ymm_a, ymm_b2);
				ymm_c3 = _mm256_mul_epu32(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a * b;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i];
		}
	};

	template<>
	struct kernel_cvtmul_uint32<float, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const float *b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				// data-type conversion
				ymm_c0 = _mm256_cvtps_epi32(ymm_b0);
				ymm_c1 = _mm256_cvtps_epi32(ymm_b1);
				ymm_c2 = _mm256_cvtps_epi32(ymm_b2);
				ymm_c3 = _mm256_cvtps_epi32(ymm_b3);
				// c = a * c;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_c0);
				ymm_c1 = _mm256_mul_epu32(ymm_a, ymm_c1);
				ymm_c2 = _mm256_mul_epu32(ymm_a, ymm_c2);
				ymm_c3 = _mm256_mul_epu32(ymm_a, ymm_c3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				// data-type conversion
				ymm_c0 = _mm256_cvtps_epi32(ymm_b0);
				// c = a * c;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_c0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

	template<>
	struct kernel_cvtmul_uint32<double, cpu_avx2>
	{
		void operator()(size_t n, unsigned int a, const double *b, unsigned int *c) const
		{
			constexpr size_t block = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m128i xmm_b0, xmm_b1;
			__m256d ymm_b0, ymm_b1;
			__m256i ymm_c0;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				// data-type conversion
				xmm_b0 = _mm256_cvtpd_epi32(ymm_b0);
				xmm_b1 = _mm256_cvtpd_epi32(ymm_b1);
				ymm_c0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_b0), xmm_b1, 1);
				// c = a * c;
				ymm_c0 = _mm256_mul_epu32(ymm_a, ymm_c0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * static_cast<unsigned int>(b[i]);
		}
	};

} // namespace core

#endif
