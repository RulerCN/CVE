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

#ifndef __CORE_CPU_KERNEL_COMPARE_GT_VALUE_H__
#define __CORE_CPU_KERNEL_COMPARE_GT_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_gt_value

	template<class T, cpu_inst_type inst>
	struct kernel_gt_value
	{
		void operator()(size_t n, const T a, const T *b, T *c) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				c[0] = a > b[0] ? T(-1) : T(0);
				c[1] = a > b[1] ? T(-1) : T(0);
				c[2] = a > b[2] ? T(-1) : T(0);
				c[3] = a > b[3] ? T(-1) : T(0);
				c[4] = a > b[4] ? T(-1) : T(0);
				c[5] = a > b[5] ? T(-1) : T(0);
				c[6] = a > b[6] ? T(-1) : T(0);
				c[7] = a > b[7] ? T(-1) : T(0);
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? T(-1) : T(0);
		}
	};

	template<>
	struct kernel_gt_value<signed char, cpu_sse2>
	{
		void operator()(size_t n, const signed char a, const signed char *b, signed char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi8(xmm_a, xmm_b0);
				xmm_c1 = _mm_cmpgt_epi8(xmm_a, xmm_b1);
				xmm_c2 = _mm_cmpgt_epi8(xmm_a, xmm_b2);
				xmm_c3 = _mm_cmpgt_epi8(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi8(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int8_true : int8_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned char, cpu_sse2>
	{
		void operator()(size_t n, const unsigned char a, const unsigned char *b, unsigned char *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m128i xmm_zero = _mm_setzero_si128();
			const __m128i xmm_a = _mm_set1_epi8(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_s0 = _mm_cmplt_epi8(_mm_xor_si128(xmm_a, xmm_b0), xmm_zero);
				xmm_s1 = _mm_cmplt_epi8(_mm_xor_si128(xmm_a, xmm_b1), xmm_zero);
				xmm_s2 = _mm_cmplt_epi8(_mm_xor_si128(xmm_a, xmm_b2), xmm_zero);
				xmm_s3 = _mm_cmplt_epi8(_mm_xor_si128(xmm_a, xmm_b3), xmm_zero);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi8(xmm_a, xmm_b0), xmm_s0);
				xmm_c1 = _mm_xor_si128(_mm_cmpgt_epi8(xmm_a, xmm_b1), xmm_s1);
				xmm_c2 = _mm_xor_si128(_mm_cmpgt_epi8(xmm_a, xmm_b2), xmm_s2);
				xmm_c3 = _mm_xor_si128(_mm_cmpgt_epi8(xmm_a, xmm_b3), xmm_s3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_s0 = _mm_cmplt_epi8(_mm_xor_si128(xmm_a, xmm_b0), xmm_zero);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi8(xmm_a, xmm_b0), xmm_s0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint8_true : uint8_false;
		}
	};

	template<>
	struct kernel_gt_value<signed short, cpu_sse2>
	{
		void operator()(size_t n, const signed short a, const signed short *b, signed short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi16(xmm_a, xmm_b0);
				xmm_c1 = _mm_cmpgt_epi16(xmm_a, xmm_b1);
				xmm_c2 = _mm_cmpgt_epi16(xmm_a, xmm_b2);
				xmm_c3 = _mm_cmpgt_epi16(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi16(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int16_true : int16_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned short, cpu_sse2>
	{
		void operator()(size_t n, const unsigned short a, const unsigned short *b, unsigned short *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m128i xmm_a = _mm_set1_epi16(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_s0 = _mm_srai_epi16(_mm_xor_si128(xmm_a, xmm_b0), 15);
				xmm_s1 = _mm_srai_epi16(_mm_xor_si128(xmm_a, xmm_b1), 15);
				xmm_s2 = _mm_srai_epi16(_mm_xor_si128(xmm_a, xmm_b2), 15);
				xmm_s3 = _mm_srai_epi16(_mm_xor_si128(xmm_a, xmm_b3), 15);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi16(xmm_a, xmm_b0), xmm_s0);
				xmm_c1 = _mm_xor_si128(_mm_cmpgt_epi16(xmm_a, xmm_b1), xmm_s1);
				xmm_c2 = _mm_xor_si128(_mm_cmpgt_epi16(xmm_a, xmm_b2), xmm_s2);
				xmm_c3 = _mm_xor_si128(_mm_cmpgt_epi16(xmm_a, xmm_b3), xmm_s3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_s0 = _mm_srai_epi16(_mm_xor_si128(xmm_a, xmm_b0), 15);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi16(xmm_a, xmm_b0), xmm_s0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint16_true : uint16_false;
		}
	};

	template<>
	struct kernel_gt_value<signed int, cpu_sse2>
	{
		void operator()(size_t n, const signed int a, const signed int *b, signed int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi32(xmm_a, xmm_b0);
				xmm_c1 = _mm_cmpgt_epi32(xmm_a, xmm_b1);
				xmm_c2 = _mm_cmpgt_epi32(xmm_a, xmm_b2);
				xmm_c3 = _mm_cmpgt_epi32(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_c0 = _mm_cmpgt_epi32(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int32_true : int32_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned int, cpu_sse2>
	{
		void operator()(size_t n, const unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128i xmm_a = _mm_set1_epi32(a);
			__m128i xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;
			__m128i xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);
				xmm_b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 2);
				xmm_b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 3);
				// c = a > b;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_a, xmm_b0), 31);
				xmm_s1 = _mm_srai_epi32(_mm_xor_si128(xmm_a, xmm_b1), 31);
				xmm_s2 = _mm_srai_epi32(_mm_xor_si128(xmm_a, xmm_b2), 31);
				xmm_s3 = _mm_srai_epi32(_mm_xor_si128(xmm_a, xmm_b3), 31);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a, xmm_b0), xmm_s0);
				xmm_c1 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a, xmm_b1), xmm_s1);
				xmm_c2 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a, xmm_b2), xmm_s2);
				xmm_c3 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a, xmm_b3), xmm_s3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 1, xmm_c1);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 2, xmm_c2);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c) + 3, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				// c = a > b;
				xmm_s0 = _mm_srai_epi32(_mm_xor_si128(xmm_a, xmm_b0), 31);
				xmm_c0 = _mm_xor_si128(_mm_cmpgt_epi32(xmm_a, xmm_b0), xmm_s0);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint32_true : uint32_false;
		}
	};

	template<>
	struct kernel_gt_value<float, cpu_sse>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_ps(xmm_a, xmm_b0);
				xmm_c1 = _mm_cmpgt_ps(xmm_a, xmm_b1);
				xmm_c2 = _mm_cmpgt_ps(xmm_a, xmm_b2);
				xmm_c3 = _mm_cmpgt_ps(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				_mm_storeu_ps(c + 4, xmm_c1);
				_mm_storeu_ps(c + 8, xmm_c2);
				_mm_storeu_ps(c + 12, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_ps(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				reinterpret_cast<int*>(c)[i] = a > b[i] ? int32_true : int32_false;
		}
	};

	template<>
	struct kernel_gt_value<double, cpu_sse2>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n >= block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_pd(xmm_a, xmm_b0);
				xmm_c1 = _mm_cmpgt_pd(xmm_a, xmm_b1);
				xmm_c2 = _mm_cmpgt_pd(xmm_a, xmm_b2);
				xmm_c3 = _mm_cmpgt_pd(xmm_a, xmm_b3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				// c = a > b;
				xmm_c0 = _mm_cmpgt_pd(xmm_a, xmm_b0);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				reinterpret_cast<long long*>(c)[i] = a > b[i] ? int64_true : int64_false;
		}
	};

	template<>
	struct kernel_gt_value<signed char, cpu_avx2>
	{
		void operator()(size_t n, const signed char a, const signed char *b, signed char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi8(ymm_a, ymm_b0);
				ymm_c1 = _mm256_cmpgt_epi8(ymm_a, ymm_b1);
				ymm_c2 = _mm256_cmpgt_epi8(ymm_a, ymm_b2);
				ymm_c3 = _mm256_cmpgt_epi8(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi8(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int8_true : int8_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned char, cpu_avx2>
	{
		void operator()(size_t n, const unsigned char a, const unsigned char *b, unsigned char *c) const
		{
			constexpr size_t block = 128;
			constexpr size_t bit = 32;
			const __m256i ymm_zero = _mm256_setzero_si256();
			const __m256i ymm_a = _mm256_set1_epi8(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi8(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_s1 = _mm256_cmpgt_epi8(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b1));
				ymm_s2 = _mm256_cmpgt_epi8(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b2));
				ymm_s3 = _mm256_cmpgt_epi8(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b3));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi8(ymm_a, ymm_b0), ymm_s0);
				ymm_c1 = _mm256_xor_si256(_mm256_cmpgt_epi8(ymm_a, ymm_b1), ymm_s1);
				ymm_c2 = _mm256_xor_si256(_mm256_cmpgt_epi8(ymm_a, ymm_b2), ymm_s2);
				ymm_c3 = _mm256_xor_si256(_mm256_cmpgt_epi8(ymm_a, ymm_b3), ymm_s3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi8(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi8(ymm_a, ymm_b0), ymm_s0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint8_true : uint8_false;
		}
	};

	template<>
	struct kernel_gt_value<signed short, cpu_avx2>
	{
		void operator()(size_t n, const signed short a, const signed short *b, signed short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi16(ymm_a, ymm_b0);
				ymm_c1 = _mm256_cmpgt_epi16(ymm_a, ymm_b1);
				ymm_c2 = _mm256_cmpgt_epi16(ymm_a, ymm_b2);
				ymm_c3 = _mm256_cmpgt_epi16(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi16(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int16_true : int16_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned short, cpu_avx2>
	{
		void operator()(size_t n, const unsigned short a, const unsigned short *b, unsigned short *c) const
		{
			constexpr size_t block = 64;
			constexpr size_t bit = 16;
			const __m256i ymm_zero = _mm256_setzero_si256();
			const __m256i ymm_a = _mm256_set1_epi16(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi16(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_s1 = _mm256_cmpgt_epi16(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b1));
				ymm_s2 = _mm256_cmpgt_epi16(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b2));
				ymm_s3 = _mm256_cmpgt_epi16(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b3));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi16(ymm_a, ymm_b0), ymm_s0);
				ymm_c1 = _mm256_xor_si256(_mm256_cmpgt_epi16(ymm_a, ymm_b1), ymm_s1);
				ymm_c2 = _mm256_xor_si256(_mm256_cmpgt_epi16(ymm_a, ymm_b2), ymm_s2);
				ymm_c3 = _mm256_xor_si256(_mm256_cmpgt_epi16(ymm_a, ymm_b3), ymm_s3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi16(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi16(ymm_a, ymm_b0), ymm_s0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint16_true : uint16_false;
		}
	};

	template<>
	struct kernel_gt_value<signed int, cpu_avx2>
	{
		void operator()(size_t n, const signed int a, const signed int *b, signed int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi32(ymm_a, ymm_b0);
				ymm_c1 = _mm256_cmpgt_epi32(ymm_a, ymm_b1);
				ymm_c2 = _mm256_cmpgt_epi32(ymm_a, ymm_b2);
				ymm_c3 = _mm256_cmpgt_epi32(ymm_a, ymm_b3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_c0 = _mm256_cmpgt_epi32(ymm_a, ymm_b0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? int32_true : int32_false;
		}
	};

	template<>
	struct kernel_gt_value<unsigned int, cpu_avx2>
	{
		void operator()(size_t n, const unsigned int a, const unsigned int *b, unsigned int *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256i ymm_zero = _mm256_setzero_si256();
			const __m256i ymm_a = _mm256_set1_epi32(a);
			__m256i ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3;
			__m256i ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);
				ymm_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 2);
				ymm_b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 3);
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi32(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_s1 = _mm256_cmpgt_epi32(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b1));
				ymm_s2 = _mm256_cmpgt_epi32(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b2));
				ymm_s3 = _mm256_cmpgt_epi32(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b3));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_b0), ymm_s0);
				ymm_c1 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_b1), ymm_s1);
				ymm_c2 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_b2), ymm_s2);
				ymm_c3 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_b3), ymm_s3);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 1, ymm_c1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 2, ymm_c2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c) + 3, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				// c = a > b;
				ymm_s0 = _mm256_cmpgt_epi32(ymm_zero, _mm256_xor_si256(ymm_a, ymm_b0));
				ymm_c0 = _mm256_xor_si256(_mm256_cmpgt_epi32(ymm_a, ymm_b0), ymm_s0);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a > b[i] ? uint32_true : uint32_false;
		}
	};

	template<>
	struct kernel_gt_value<float, cpu_avx>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				// c = a > b;
				ymm_c0 = _mm256_cmp_ps(ymm_a, ymm_b0, _CMP_GT_OS);
				ymm_c1 = _mm256_cmp_ps(ymm_a, ymm_b1, _CMP_GT_OS);
				ymm_c2 = _mm256_cmp_ps(ymm_a, ymm_b2, _CMP_GT_OS);
				ymm_c3 = _mm256_cmp_ps(ymm_a, ymm_b3, _CMP_GT_OS);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				_mm256_storeu_ps(c + 8, ymm_c1);
				_mm256_storeu_ps(c + 16, ymm_c2);
				_mm256_storeu_ps(c + 24, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				// c = a > b;
				ymm_c0 = _mm256_cmp_ps(ymm_a, ymm_b0, _CMP_GT_OS);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				reinterpret_cast<int*>(c)[i] = a > b[i] ? int32_true : int32_false;
		}
	};

	template<>
	struct kernel_gt_value<double, cpu_avx>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n >= block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				// c = a > b;
				ymm_c0 = _mm256_cmp_pd(ymm_a, ymm_b0, _CMP_GT_OS);
				ymm_c1 = _mm256_cmp_pd(ymm_a, ymm_b1, _CMP_GT_OS);
				ymm_c2 = _mm256_cmp_pd(ymm_a, ymm_b2, _CMP_GT_OS);
				ymm_c3 = _mm256_cmp_pd(ymm_a, ymm_b3, _CMP_GT_OS);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n >= bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				// c = a > b;
				ymm_c0 = _mm256_cmp_pd(ymm_a, ymm_b0, _CMP_GT_OS);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				reinterpret_cast<long long*>(c)[i] = a > b[i] ? int64_true : int64_false;
		}
	};

} // namespace core

#endif
