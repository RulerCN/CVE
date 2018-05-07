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

#ifndef __CORE_CPU_KERNEL_CONVERT_SCALE_DOUBLE_H__
#define __CORE_CPU_KERNEL_CONVERT_SCALE_DOUBLE_H__

#include "../cpu.h"

namespace core
{
	// Class template kernel_convert_scale_double
	template<class T, cpu_inst_type inst>
	struct kernel_convert_scale_double
	{
		void operator()(size_t n, const T *a, double *b, double scale) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				b[0] = scale * static_cast<double>(a[0]);
				b[1] = scale * static_cast<double>(a[1]);
				b[2] = scale * static_cast<double>(a[2]);
				b[3] = scale * static_cast<double>(a[3]);
				b[4] = scale * static_cast<double>(a[4]);
				b[5] = scale * static_cast<double>(a[5]);
				b[6] = scale * static_cast<double>(a[6]);
				b[7] = scale * static_cast<double>(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed char, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;

			while (n > block)
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
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				xmm_b2 = _mm_mul_pd(xmm_scale, xmm_b2);
				xmm_b3 = _mm_mul_pd(xmm_scale, xmm_b3);
				xmm_b4 = _mm_mul_pd(xmm_scale, xmm_b4);
				xmm_b5 = _mm_mul_pd(xmm_scale, xmm_b5);
				xmm_b6 = _mm_mul_pd(xmm_scale, xmm_b6);
				xmm_b7 = _mm_mul_pd(xmm_scale, xmm_b7);
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
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, const unsigned char *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3, xmm_b4, xmm_b5, xmm_b6, xmm_b7;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, 1);
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, 2);
				xmm_a6 = _mm_shuffle_epi32(xmm_a0, 3);
				xmm_a0 = _mm_cvtepu8_epi32(xmm_a0);
				xmm_a2 = _mm_cvtepu8_epi32(xmm_a2);
				xmm_a4 = _mm_cvtepu8_epi32(xmm_a4);
				xmm_a6 = _mm_cvtepu8_epi32(xmm_a6);
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
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				xmm_b2 = _mm_mul_pd(xmm_scale, xmm_b2);
				xmm_b3 = _mm_mul_pd(xmm_scale, xmm_b3);
				xmm_b4 = _mm_mul_pd(xmm_scale, xmm_b4);
				xmm_b5 = _mm_mul_pd(xmm_scale, xmm_b5);
				xmm_b6 = _mm_mul_pd(xmm_scale, xmm_b6);
				xmm_b7 = _mm_mul_pd(xmm_scale, xmm_b7);
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
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed short, cpu_sse41>
	{
		void operator()(size_t n, const signed short *a, double *b, double scale) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi32_pd(xmm_a0);
				xmm_b1 = _mm_cvtepi32_pd(xmm_a1);
				xmm_b2 = _mm_cvtepi32_pd(xmm_a2);
				xmm_b3 = _mm_cvtepi32_pd(xmm_a3);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				xmm_b2 = _mm_mul_pd(xmm_scale, xmm_b2);
				xmm_b3 = _mm_mul_pd(xmm_scale, xmm_b3);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, const unsigned short *a, double *b, double scale) const
		{
			constexpr size_t block = 8;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu16_epi32(xmm_a0);
				xmm_a2 = _mm_cvtepu16_epi32(xmm_a2);
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi32_pd(xmm_a0);
				xmm_b1 = _mm_cvtepi32_pd(xmm_a1);
				xmm_b2 = _mm_cvtepi32_pd(xmm_a2);
				xmm_b3 = _mm_cvtepi32_pd(xmm_a3);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				xmm_b2 = _mm_mul_pd(xmm_scale, xmm_b2);
				xmm_b3 = _mm_mul_pd(xmm_scale, xmm_b3);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int *a, double *b, double scale) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtepi32_pd(xmm_a0);
				xmm_b1 = _mm_cvtepi32_pd(xmm_a1);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned int, cpu_sse41>
	{
		void operator()(size_t n, const unsigned int *a, double *b, double scale) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi64x(0x41e0000000000000LL);
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128i xmm_a, xmm_av0, xmm_av1, xmm_as0, xmm_as1;
			__m128d xmm_b0, xmm_b1, xmm_bv0, xmm_bv1, xmm_bs0, xmm_bs1;

			while (n > block)
			{
				// load data from memory
				xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_av0 = _mm_and_si128(xmm_a, abs);
				xmm_as0 = _mm_srai_epi32(xmm_a, 31);
				xmm_av1 = _mm_shuffle_epi32(xmm_av0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_as1 = _mm_shuffle_epi32(xmm_as0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_av0 = _mm_cvtepi32_epi64(xmm_av0);
				xmm_av1 = _mm_cvtepi32_epi64(xmm_av1);
				xmm_as0 = _mm_cvtepi32_epi64(xmm_as0);
				xmm_as1 = _mm_cvtepi32_epi64(xmm_as1);
				xmm_bv0 = _mm_cvtepi32_pd(xmm_av0);
				xmm_bv1 = _mm_cvtepi32_pd(xmm_av1);
				xmm_bs0 = _mm_castsi128_pd(_mm_and_si128(xmm_av0, val));
				xmm_bs1 = _mm_castsi128_pd(_mm_and_si128(xmm_av1, val));
				xmm_b0 = _mm_add_pd(xmm_bv0, xmm_bs0);
				xmm_b1 = _mm_add_pd(xmm_bv1, xmm_bs1);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<float, cpu_sse2>
	{
		void operator()(size_t n, const float *a, double *b, double scale) const
		{
			constexpr size_t block = 4;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128 xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				// numerical linear transformation
				xmm_a1 = _mm_shuffle_ps(xmm_a0, xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_b0 = _mm_cvtps_pd(xmm_a0);
				xmm_b1 = _mm_cvtps_pd(xmm_a1);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_b0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_b1);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
				*b++ = scale * static_cast<double>(*a++);
		}
	};

	template<>
	struct kernel_convert_scale_double<double, cpu_sse2>
	{
		void operator()(size_t n, const double *a, double *b, double scale) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_scale = _mm_set1_pd(scale);
			__m128d xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				xmm_a1 = _mm_loadu_pd(a + 2);
				xmm_a2 = _mm_loadu_pd(a + 4);
				xmm_a3 = _mm_loadu_pd(a + 6);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_a0);
				xmm_b1 = _mm_mul_pd(xmm_scale, xmm_a1);
				xmm_b2 = _mm_mul_pd(xmm_scale, xmm_a2);
				xmm_b3 = _mm_mul_pd(xmm_scale, xmm_a3);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				// numerical linear transformation
				xmm_b0 = _mm_mul_pd(xmm_scale, xmm_a0);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			while (n > 0)
			{
				*b++ = scale * (*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_a0, ymm_a1;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
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
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				ymm_b2 = _mm256_mul_pd(ymm_scale, ymm_b2);
				ymm_b3 = _mm256_mul_pd(ymm_scale, ymm_b3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256i ymm_a0, ymm_a1;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
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
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				ymm_b2 = _mm256_mul_pd(ymm_scale, ymm_b2);
				ymm_b3 = _mm256_mul_pd(ymm_scale, ymm_b3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short *a, double *b, double scale) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0;
			__m256d ymm_b0, ymm_b1;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				ymm_b1 = _mm256_cvtepi32_pd(xmm_a1);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short *a, double *b, double scale) const
		{
			constexpr size_t block = 8;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0;
			__m256d ymm_b0, ymm_b1;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				ymm_b1 = _mm256_cvtepi32_pd(xmm_a1);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<signed int, cpu_avx>
	{
		void operator()(size_t n, const signed int *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				ymm_b1 = _mm256_cvtepi32_pd(xmm_a1);
				ymm_b2 = _mm256_cvtepi32_pd(xmm_a2);
				ymm_b3 = _mm256_cvtepi32_pd(xmm_a3);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				ymm_b2 = _mm256_mul_pd(ymm_scale, ymm_b2);
				ymm_b3 = _mm256_mul_pd(ymm_scale, ymm_b3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b0);
				_mm256_storeu_pd(b + 12, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				ymm_b0 = _mm256_cvtepi32_pd(xmm_a0);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int *a, double *b, double scale) const
		{
			constexpr size_t block = 4;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi64x(0x41e0000000000000LL);
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128i xmm_a, xmm_av, xmm_as;
			__m256i ymm_s;
			__m256d ymm_b, ymm_bs, ymm_bv;

			while (n > block)
			{
				// load data from memory
				xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// data-type conversion
				xmm_av = _mm_and_si128(xmm_a, abs);
				xmm_as = _mm_srai_epi32(xmm_a, 31);
				ymm_s = _mm256_cvtepi32_epi64(xmm_as);
				ymm_bv = _mm256_cvtepi32_pd(xmm_av);
				ymm_bs = _mm256_castsi256_pd(_mm256_and_si256(ymm_s, val));
				ymm_b = _mm256_add_pd(ymm_bv, ymm_bs);
				// numerical linear transformation
				ymm_b = _mm256_mul_pd(ymm_scale, ymm_b);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b);
				a += block;
				b += block;
				n -= block;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<float, cpu_avx>
	{
		void operator()(size_t n, const float *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				xmm_a1 = _mm_loadu_ps(a + 4);
				xmm_a2 = _mm_loadu_ps(a + 8);
				xmm_a3 = _mm_loadu_ps(a + 12);
				// data-type conversion
				ymm_b0 = _mm256_cvtps_pd(xmm_a0);
				ymm_b1 = _mm256_cvtps_pd(xmm_a1);
				ymm_b2 = _mm256_cvtps_pd(xmm_a2);
				ymm_b3 = _mm256_cvtps_pd(xmm_a3);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_b1);
				ymm_b2 = _mm256_mul_pd(ymm_scale, ymm_b2);
				ymm_b3 = _mm256_mul_pd(ymm_scale, ymm_b3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b0);
				_mm256_storeu_pd(b + 12, ymm_b1);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				// data-type conversion
				ymm_b0 = _mm256_cvtps_pd(xmm_a0);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_b0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			while (n > 0)
			{
				*b++ = scale * static_cast<double>(*a++);
				--n;
			}
		}
	};

	template<>
	struct kernel_convert_scale_double<double, cpu_avx>
	{
		void operator()(size_t n, const double *a, double *b, double scale) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_scale = _mm256_set1_pd(scale);
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				ymm_a1 = _mm256_loadu_pd(a + 8);
				ymm_a2 = _mm256_loadu_pd(a + 16);
				ymm_a3 = _mm256_loadu_pd(a + 24);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_a0);
				ymm_b1 = _mm256_mul_pd(ymm_scale, ymm_a1);
				ymm_b2 = _mm256_mul_pd(ymm_scale, ymm_a2);
				ymm_b3 = _mm256_mul_pd(ymm_scale, ymm_a3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				// numerical linear transformation
				ymm_b0 = _mm256_mul_pd(ymm_scale, ymm_a0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			while (n > 0)
			{
				*b++ = scale * (*a++);
				--n;
			}
		}
	};

} // namespace core

#endif
