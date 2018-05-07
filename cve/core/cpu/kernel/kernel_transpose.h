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

#ifndef __CORE_CPU_KERNEL_TRANSPOSE_H__
#define __CORE_CPU_KERNEL_TRANSPOSE_H__

#include "../cpu.h"

namespace core
{
	// Class template common_transpose
	template<class T>
	struct common_transpose
	{
		// b[j][i] = a[i][j]
		void operator()(size_t m, size_t n, const T *a, size_t rsa, T *b, size_t rsb) const
		{
			T *ptr_b;
			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b + i;
				for (size_t j = 0; j < n; ++j)
				{
					*ptr_b = a[j];
					ptr_b += rsb;
				}
				a += rsa;
			}
		}
	};

	// Class template block_transpose
	template<class T, inst_type inst>
	struct block_transpose
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const T *a, size_t rsa, T *b, size_t rsb) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = ptr_a0 + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;

			for (size_t j = 0; j < n;)
			{
				// j
				b[0] = ptr_a0[j];
				b[1] = ptr_a1[j];
				b[2] = ptr_a2[j];
				b[3] = ptr_a3[j];
				b += rsb;
				++j;
				// j + 1
				b[0] = ptr_a0[j];
				b[1] = ptr_a1[j];
				b[2] = ptr_a2[j];
				b[3] = ptr_a3[j];
				b += rsb;
				++j;
				// j + 2
				b[0] = ptr_a0[j];
				b[1] = ptr_a1[j];
				b[2] = ptr_a2[j];
				b[3] = ptr_a3[j];
				b += rsb;
				++j;
				// j + 3
				b[0] = ptr_a0[j];
				b[1] = ptr_a1[j];
				b[2] = ptr_a2[j];
				b[3] = ptr_a3[j];
				b += rsb;
				++j;
			}
		}
	};

	template<>
	struct block_transpose<signed char, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, signed char *b, size_t rsb) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = ptr_a0 + rsa;
			const signed char *ptr_a2 = ptr_a1 + rsa;
			const signed char *ptr_a3 = ptr_a2 + rsa;
			const signed char *ptr_a4 = ptr_a3 + rsa;
			const signed char *ptr_a5 = ptr_a4 + rsa;
			const signed char *ptr_a6 = ptr_a5 + rsa;
			const signed char *ptr_a7 = ptr_a6 + rsa;
			const signed char *ptr_a8 = ptr_a7 + rsa;
			const signed char *ptr_a9 = ptr_a8 + rsa;
			const signed char *ptr_aa = ptr_a9 + rsa;
			const signed char *ptr_ab = ptr_aa + rsa;
			const signed char *ptr_ac = ptr_ab + rsa;
			const signed char *ptr_ad = ptr_ac + rsa;
			const signed char *ptr_ae = ptr_ad + rsa;
			const signed char *ptr_af = ptr_ae + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				xmm_a4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a4 + j));
				xmm_a5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a5 + j));
				xmm_a6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a6 + j));
				xmm_a7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a7 + j));
				xmm_t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_t1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_t2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_t3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_t4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_t5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_t6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_t7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_af + j));
				// matrix transposed
				xmm_a0 = _mm_unpacklo_epi8(xmm_a0, xmm_a1);
				xmm_a1 = _mm_unpacklo_epi8(xmm_a2, xmm_a3);
				xmm_a2 = _mm_unpacklo_epi8(xmm_a4, xmm_a5);
				xmm_a3 = _mm_unpacklo_epi8(xmm_a6, xmm_a7);
				xmm_a4 = _mm_unpacklo_epi8(xmm_t0, xmm_t1);
				xmm_a5 = _mm_unpacklo_epi8(xmm_t2, xmm_t3);
				xmm_a6 = _mm_unpacklo_epi8(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpacklo_epi8(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t4 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t5 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a2 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
				xmm_a5 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
				xmm_a6 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
				xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a4);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a4);
				xmm_t2 = _mm_unpacklo_epi64(xmm_a1, xmm_a5);
				xmm_t3 = _mm_unpackhi_epi64(xmm_a1, xmm_a5);
				xmm_t4 = _mm_unpacklo_epi64(xmm_a2, xmm_a6);
				xmm_t5 = _mm_unpackhi_epi64(xmm_a2, xmm_a6);
				xmm_t6 = _mm_unpacklo_epi64(xmm_a3, xmm_a7);
				xmm_t7 = _mm_unpackhi_epi64(xmm_a3, xmm_a7);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t3);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t4);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t5);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t6);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned char, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, unsigned char *b, size_t rsb) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = ptr_a0 + rsa;
			const unsigned char *ptr_a2 = ptr_a1 + rsa;
			const unsigned char *ptr_a3 = ptr_a2 + rsa;
			const unsigned char *ptr_a4 = ptr_a3 + rsa;
			const unsigned char *ptr_a5 = ptr_a4 + rsa;
			const unsigned char *ptr_a6 = ptr_a5 + rsa;
			const unsigned char *ptr_a7 = ptr_a6 + rsa;
			const unsigned char *ptr_a8 = ptr_a7 + rsa;
			const unsigned char *ptr_a9 = ptr_a8 + rsa;
			const unsigned char *ptr_aa = ptr_a9 + rsa;
			const unsigned char *ptr_ab = ptr_aa + rsa;
			const unsigned char *ptr_ac = ptr_ab + rsa;
			const unsigned char *ptr_ad = ptr_ac + rsa;
			const unsigned char *ptr_ae = ptr_ad + rsa;
			const unsigned char *ptr_af = ptr_ae + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				xmm_a4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a4 + j));
				xmm_a5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a5 + j));
				xmm_a6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a6 + j));
				xmm_a7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a7 + j));
				xmm_t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_t1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_t2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_t3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_t4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_t5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_t6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_t7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr_af + j));
				// matrix transposed
				xmm_a0 = _mm_unpacklo_epi8(xmm_a0, xmm_a1);
				xmm_a1 = _mm_unpacklo_epi8(xmm_a2, xmm_a3);
				xmm_a2 = _mm_unpacklo_epi8(xmm_a4, xmm_a5);
				xmm_a3 = _mm_unpacklo_epi8(xmm_a6, xmm_a7);
				xmm_a4 = _mm_unpacklo_epi8(xmm_t0, xmm_t1);
				xmm_a5 = _mm_unpacklo_epi8(xmm_t2, xmm_t3);
				xmm_a6 = _mm_unpacklo_epi8(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpacklo_epi8(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t4 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t5 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a2 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
				xmm_a5 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
				xmm_a6 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
				xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a4);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a4);
				xmm_t2 = _mm_unpacklo_epi64(xmm_a1, xmm_a5);
				xmm_t3 = _mm_unpackhi_epi64(xmm_a1, xmm_a5);
				xmm_t4 = _mm_unpacklo_epi64(xmm_a2, xmm_a6);
				xmm_t5 = _mm_unpackhi_epi64(xmm_a2, xmm_a6);
				xmm_t6 = _mm_unpacklo_epi64(xmm_a3, xmm_a7);
				xmm_t7 = _mm_unpackhi_epi64(xmm_a3, xmm_a7);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t3);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t4);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t5);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t6);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<signed short, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, signed short *b, size_t rsb) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = ptr_a0 + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			const signed short *ptr_a4 = ptr_a3 + rsa;
			const signed short *ptr_a5 = ptr_a4 + rsa;
			const signed short *ptr_a6 = ptr_a5 + rsa;
			const signed short *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j));
				xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j));
				xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j));
				xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j));
				// matrix transposed
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t3 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t4 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t5 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
				xmm_a5 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
				xmm_a6 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t2 = _mm_unpacklo_epi64(xmm_a2, xmm_a3);
				xmm_t3 = _mm_unpackhi_epi64(xmm_a2, xmm_a3);
				xmm_t4 = _mm_unpacklo_epi64(xmm_a4, xmm_a5);
				xmm_t5 = _mm_unpackhi_epi64(xmm_a4, xmm_a5);
				xmm_t6 = _mm_unpacklo_epi64(xmm_a6, xmm_a7);
				xmm_t7 = _mm_unpackhi_epi64(xmm_a6, xmm_a7);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t3);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t4);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t5);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t6);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned short, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, unsigned short *b, size_t rsb) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = ptr_a0 + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			const unsigned short *ptr_a4 = ptr_a3 + rsa;
			const unsigned short *ptr_a5 = ptr_a4 + rsa;
			const unsigned short *ptr_a6 = ptr_a5 + rsa;
			const unsigned short *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j));
				xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j));
				xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j));
				xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j));
				// matrix transposed
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t3 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t4 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t5 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
				xmm_a5 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
				xmm_a6 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t2 = _mm_unpacklo_epi64(xmm_a2, xmm_a3);
				xmm_t3 = _mm_unpackhi_epi64(xmm_a2, xmm_a3);
				xmm_t4 = _mm_unpacklo_epi64(xmm_a4, xmm_a5);
				xmm_t5 = _mm_unpackhi_epi64(xmm_a4, xmm_a5);
				xmm_t6 = _mm_unpacklo_epi64(xmm_a6, xmm_a7);
				xmm_t7 = _mm_unpackhi_epi64(xmm_a6, xmm_a7);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t3);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t4);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t5);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t6);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<signed int, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, signed int *b, size_t rsb) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// matrix transposed
				xmm_t0 = _mm_unpacklo_epi32(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi32(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi32(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi32(xmm_a2, xmm_a3);
				xmm_a0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi64(xmm_t0, xmm_t1);
				xmm_a2 = _mm_unpacklo_epi64(xmm_t2, xmm_t3);
				xmm_a3 = _mm_unpackhi_epi64(xmm_t2, xmm_t3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a3);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned int, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned int *a, size_t rsa, unsigned int *b, size_t rsb) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// matrix transposed
				xmm_t0 = _mm_unpacklo_epi32(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi32(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi32(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi32(xmm_a2, xmm_a3);
				xmm_a0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi64(xmm_t0, xmm_t1);
				xmm_a2 = _mm_unpacklo_epi64(xmm_t2, xmm_t3);
				xmm_a3 = _mm_unpackhi_epi64(xmm_t2, xmm_t3);
				// store data into memory
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a0);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a1);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a2);
				b += rsb;
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_a3);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<float, inst_sse>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const float *a, size_t rsa, float *b, size_t rsb) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
				xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
				xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
				xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
				// matrix transposed
				xmm_t0 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(1, 0, 1, 0));
				xmm_t1 = _mm_shuffle_ps(xmm_a2, xmm_a3, _MM_SHUFFLE(1, 0, 1, 0));
				xmm_t2 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(3, 2, 3, 2));
				xmm_t3 = _mm_shuffle_ps(xmm_a2, xmm_a3, _MM_SHUFFLE(3, 2, 3, 2));
				xmm_a0 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(2, 0, 2, 0));
				xmm_a1 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(3, 1, 3, 1));
				xmm_a2 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(2, 0, 2, 0));
				xmm_a3 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(3, 1, 3, 1));
				// store data into memory
				_mm_storeu_ps(b, xmm_a0);
				b += rsb;
				_mm_storeu_ps(b, xmm_a1);
				b += rsb;
				_mm_storeu_ps(b, xmm_a2);
				b += rsb;
				_mm_storeu_ps(b, xmm_a3);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<double, inst_sse2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const double *a, size_t rsa, double *b, size_t rsb) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_t0, xmm_t1;

			for (size_t j = 0; j < n; j += 2)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(ptr_a0 + j);
				xmm_a1 = _mm_loadu_pd(ptr_a1 + j);
				// matrix transposed
				xmm_t0 = _mm_shuffle_pd(xmm_a0, xmm_a1, _MM_SHUFFLE(0, 0, 0, 0));
				xmm_t1 = _mm_shuffle_pd(xmm_a0, xmm_a1, _MM_SHUFFLE(0, 0, 3, 3));
				// store data into memory
				_mm_storeu_pd(b, xmm_t0);
				b += rsb;
				_mm_storeu_pd(b, xmm_t1);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<signed char, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, signed char *b, size_t rsb) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = ptr_a0 + rsa;
			const signed char *ptr_a2 = ptr_a1 + rsa;
			const signed char *ptr_a3 = ptr_a2 + rsa;
			const signed char *ptr_a4 = ptr_a3 + rsa;
			const signed char *ptr_a5 = ptr_a4 + rsa;
			const signed char *ptr_a6 = ptr_a5 + rsa;
			const signed char *ptr_a7 = ptr_a6 + rsa;
			const signed char *ptr_a8 = ptr_a7 + rsa;
			const signed char *ptr_a9 = ptr_a8 + rsa;
			const signed char *ptr_aa = ptr_a9 + rsa;
			const signed char *ptr_ab = ptr_aa + rsa;
			const signed char *ptr_ac = ptr_ab + rsa;
			const signed char *ptr_ad = ptr_ac + rsa;
			const signed char *ptr_ae = ptr_ad + rsa;
			const signed char *ptr_af = ptr_ae + rsa;
			const signed char *ptr_ag = ptr_af + rsa;
			const signed char *ptr_ah = ptr_ag + rsa;
			const signed char *ptr_ai = ptr_ah + rsa;
			const signed char *ptr_aj = ptr_ai + rsa;
			const signed char *ptr_ak = ptr_aj + rsa;
			const signed char *ptr_al = ptr_ak + rsa;
			const signed char *ptr_am = ptr_al + rsa;
			const signed char *ptr_an = ptr_am + rsa;
			const signed char *ptr_ao = ptr_an + rsa;
			const signed char *ptr_ap = ptr_ao + rsa;
			const signed char *ptr_aq = ptr_ap + rsa;
			const signed char *ptr_ar = ptr_aq + rsa;
			const signed char *ptr_as = ptr_ar + rsa;
			const signed char *ptr_at = ptr_as + rsa;
			const signed char *ptr_au = ptr_at + rsa;
			const signed char *ptr_av = ptr_au + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf;

			for (size_t j = 0; j < n; j += 16)
			{
				// load data from memory
				ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j)));
				ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j)));
				ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j)));
				ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j)));
				ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j)));
				ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j)));
				ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j)));
				ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j)));
				ymm_a8 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j)));
				ymm_a9 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j)));
				ymm_aa = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j)));
				ymm_ab = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j)));
				ymm_ac = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j)));
				ymm_ad = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j)));
				ymm_ae = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j)));
				ymm_af = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j)));
				ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ag + j)), 1);
				ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ah + j)), 1);
				ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ai + j)), 1);
				ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aj + j)), 1);
				ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ak + j)), 1);
				ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_al + j)), 1);
				ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_am + j)), 1);
				ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_an + j)), 1);
				ymm_a8 = _mm256_insertf128_si256(ymm_a8, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ao + j)), 1);
				ymm_a9 = _mm256_insertf128_si256(ymm_a9, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ap + j)), 1);
				ymm_aa = _mm256_insertf128_si256(ymm_aa, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aq + j)), 1);
				ymm_ab = _mm256_insertf128_si256(ymm_ab, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ar + j)), 1);
				ymm_ac = _mm256_insertf128_si256(ymm_ac, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_as + j)), 1);
				ymm_ad = _mm256_insertf128_si256(ymm_ad, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_at + j)), 1);
				ymm_ae = _mm256_insertf128_si256(ymm_ae, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_au + j)), 1);
				ymm_af = _mm256_insertf128_si256(ymm_af, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_av + j)), 1);
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi8(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi8(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi8(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi8(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpacklo_epi8(ymm_a8, ymm_a9);
				ymm_t5 = _mm256_unpacklo_epi8(ymm_aa, ymm_ab);
				ymm_t6 = _mm256_unpacklo_epi8(ymm_ac, ymm_ad);
				ymm_t7 = _mm256_unpacklo_epi8(ymm_ae, ymm_af);
				ymm_t8 = _mm256_unpackhi_epi8(ymm_a0, ymm_a1);
				ymm_t9 = _mm256_unpackhi_epi8(ymm_a2, ymm_a3);
				ymm_ta = _mm256_unpackhi_epi8(ymm_a4, ymm_a5);
				ymm_tb = _mm256_unpackhi_epi8(ymm_a6, ymm_a7);
				ymm_tc = _mm256_unpackhi_epi8(ymm_a8, ymm_a9);
				ymm_td = _mm256_unpackhi_epi8(ymm_aa, ymm_ab);
				ymm_te = _mm256_unpackhi_epi8(ymm_ac, ymm_ad);
				ymm_tf = _mm256_unpackhi_epi8(ymm_ae, ymm_af);
				ymm_a0 = _mm256_unpacklo_epi16(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi16(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpacklo_epi16(ymm_t4, ymm_t5);
				ymm_a3 = _mm256_unpacklo_epi16(ymm_t6, ymm_t7);
				ymm_a4 = _mm256_unpackhi_epi16(ymm_t0, ymm_t1);
				ymm_a5 = _mm256_unpackhi_epi16(ymm_t2, ymm_t3);
				ymm_a6 = _mm256_unpackhi_epi16(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi16(ymm_t6, ymm_t7);
				ymm_a8 = _mm256_unpacklo_epi16(ymm_t8, ymm_t9);
				ymm_a9 = _mm256_unpacklo_epi16(ymm_ta, ymm_tb);
				ymm_aa = _mm256_unpacklo_epi16(ymm_tc, ymm_td);
				ymm_ab = _mm256_unpacklo_epi16(ymm_te, ymm_tf);
				ymm_ac = _mm256_unpackhi_epi16(ymm_t8, ymm_t9);
				ymm_ad = _mm256_unpackhi_epi16(ymm_ta, ymm_tb);
				ymm_ae = _mm256_unpackhi_epi16(ymm_tc, ymm_td);
				ymm_af = _mm256_unpackhi_epi16(ymm_te, ymm_tf);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t4 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
				ymm_t5 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
				ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
				ymm_t8 = _mm256_unpacklo_epi32(ymm_a8, ymm_a9);
				ymm_t9 = _mm256_unpacklo_epi32(ymm_aa, ymm_ab);
				ymm_ta = _mm256_unpackhi_epi32(ymm_a8, ymm_a9);
				ymm_tb = _mm256_unpackhi_epi32(ymm_aa, ymm_ab);
				ymm_tc = _mm256_unpacklo_epi32(ymm_ac, ymm_ad);
				ymm_td = _mm256_unpacklo_epi32(ymm_ae, ymm_af);
				ymm_te = _mm256_unpackhi_epi32(ymm_ac, ymm_ad);
				ymm_tf = _mm256_unpackhi_epi32(ymm_ae, ymm_af);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_a2 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
				ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
				ymm_a6 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
				ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
				ymm_a8 = _mm256_unpacklo_epi64(ymm_t8, ymm_t9);
				ymm_a9 = _mm256_unpackhi_epi64(ymm_t8, ymm_t9);
				ymm_aa = _mm256_unpacklo_epi64(ymm_ta, ymm_tb);
				ymm_ab = _mm256_unpackhi_epi64(ymm_ta, ymm_tb);
				ymm_ac = _mm256_unpacklo_epi64(ymm_tc, ymm_td);
				ymm_ad = _mm256_unpackhi_epi64(ymm_tc, ymm_td);
				ymm_ae = _mm256_unpacklo_epi64(ymm_te, ymm_tf);
				ymm_af = _mm256_unpackhi_epi64(ymm_te, ymm_tf);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a7);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a8);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a9);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_aa);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ab);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ac);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ad);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ae);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_af);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned char, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, unsigned char *b, size_t rsb) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = ptr_a0 + rsa;
			const unsigned char *ptr_a2 = ptr_a1 + rsa;
			const unsigned char *ptr_a3 = ptr_a2 + rsa;
			const unsigned char *ptr_a4 = ptr_a3 + rsa;
			const unsigned char *ptr_a5 = ptr_a4 + rsa;
			const unsigned char *ptr_a6 = ptr_a5 + rsa;
			const unsigned char *ptr_a7 = ptr_a6 + rsa;
			const unsigned char *ptr_a8 = ptr_a7 + rsa;
			const unsigned char *ptr_a9 = ptr_a8 + rsa;
			const unsigned char *ptr_aa = ptr_a9 + rsa;
			const unsigned char *ptr_ab = ptr_aa + rsa;
			const unsigned char *ptr_ac = ptr_ab + rsa;
			const unsigned char *ptr_ad = ptr_ac + rsa;
			const unsigned char *ptr_ae = ptr_ad + rsa;
			const unsigned char *ptr_af = ptr_ae + rsa;
			const unsigned char *ptr_ag = ptr_af + rsa;
			const unsigned char *ptr_ah = ptr_ag + rsa;
			const unsigned char *ptr_ai = ptr_ah + rsa;
			const unsigned char *ptr_aj = ptr_ai + rsa;
			const unsigned char *ptr_ak = ptr_aj + rsa;
			const unsigned char *ptr_al = ptr_ak + rsa;
			const unsigned char *ptr_am = ptr_al + rsa;
			const unsigned char *ptr_an = ptr_am + rsa;
			const unsigned char *ptr_ao = ptr_an + rsa;
			const unsigned char *ptr_ap = ptr_ao + rsa;
			const unsigned char *ptr_aq = ptr_ap + rsa;
			const unsigned char *ptr_ar = ptr_aq + rsa;
			const unsigned char *ptr_as = ptr_ar + rsa;
			const unsigned char *ptr_at = ptr_as + rsa;
			const unsigned char *ptr_au = ptr_at + rsa;
			const unsigned char *ptr_av = ptr_au + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf;

			for (size_t j = 0; j < n; j += 16)
			{
				// load data from memory
				ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j)));
				ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j)));
				ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j)));
				ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j)));
				ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j)));
				ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j)));
				ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j)));
				ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j)));
				ymm_a8 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j)));
				ymm_a9 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j)));
				ymm_aa = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j)));
				ymm_ab = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j)));
				ymm_ac = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j)));
				ymm_ad = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j)));
				ymm_ae = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j)));
				ymm_af = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j)));
				ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ag + j)), 1);
				ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ah + j)), 1);
				ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ai + j)), 1);
				ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aj + j)), 1);
				ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ak + j)), 1);
				ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_al + j)), 1);
				ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_am + j)), 1);
				ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_an + j)), 1);
				ymm_a8 = _mm256_insertf128_si256(ymm_a8, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ao + j)), 1);
				ymm_a9 = _mm256_insertf128_si256(ymm_a9, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ap + j)), 1);
				ymm_aa = _mm256_insertf128_si256(ymm_aa, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aq + j)), 1);
				ymm_ab = _mm256_insertf128_si256(ymm_ab, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ar + j)), 1);
				ymm_ac = _mm256_insertf128_si256(ymm_ac, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_as + j)), 1);
				ymm_ad = _mm256_insertf128_si256(ymm_ad, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_at + j)), 1);
				ymm_ae = _mm256_insertf128_si256(ymm_ae, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_au + j)), 1);
				ymm_af = _mm256_insertf128_si256(ymm_af, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_av + j)), 1);
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi8(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi8(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi8(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi8(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpacklo_epi8(ymm_a8, ymm_a9);
				ymm_t5 = _mm256_unpacklo_epi8(ymm_aa, ymm_ab);
				ymm_t6 = _mm256_unpacklo_epi8(ymm_ac, ymm_ad);
				ymm_t7 = _mm256_unpacklo_epi8(ymm_ae, ymm_af);
				ymm_t8 = _mm256_unpackhi_epi8(ymm_a0, ymm_a1);
				ymm_t9 = _mm256_unpackhi_epi8(ymm_a2, ymm_a3);
				ymm_ta = _mm256_unpackhi_epi8(ymm_a4, ymm_a5);
				ymm_tb = _mm256_unpackhi_epi8(ymm_a6, ymm_a7);
				ymm_tc = _mm256_unpackhi_epi8(ymm_a8, ymm_a9);
				ymm_td = _mm256_unpackhi_epi8(ymm_aa, ymm_ab);
				ymm_te = _mm256_unpackhi_epi8(ymm_ac, ymm_ad);
				ymm_tf = _mm256_unpackhi_epi8(ymm_ae, ymm_af);
				ymm_a0 = _mm256_unpacklo_epi16(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi16(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpacklo_epi16(ymm_t4, ymm_t5);
				ymm_a3 = _mm256_unpacklo_epi16(ymm_t6, ymm_t7);
				ymm_a4 = _mm256_unpackhi_epi16(ymm_t0, ymm_t1);
				ymm_a5 = _mm256_unpackhi_epi16(ymm_t2, ymm_t3);
				ymm_a6 = _mm256_unpackhi_epi16(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi16(ymm_t6, ymm_t7);
				ymm_a8 = _mm256_unpacklo_epi16(ymm_t8, ymm_t9);
				ymm_a9 = _mm256_unpacklo_epi16(ymm_ta, ymm_tb);
				ymm_aa = _mm256_unpacklo_epi16(ymm_tc, ymm_td);
				ymm_ab = _mm256_unpacklo_epi16(ymm_te, ymm_tf);
				ymm_ac = _mm256_unpackhi_epi16(ymm_t8, ymm_t9);
				ymm_ad = _mm256_unpackhi_epi16(ymm_ta, ymm_tb);
				ymm_ae = _mm256_unpackhi_epi16(ymm_tc, ymm_td);
				ymm_af = _mm256_unpackhi_epi16(ymm_te, ymm_tf);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t4 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
				ymm_t5 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
				ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
				ymm_t8 = _mm256_unpacklo_epi32(ymm_a8, ymm_a9);
				ymm_t9 = _mm256_unpacklo_epi32(ymm_aa, ymm_ab);
				ymm_ta = _mm256_unpackhi_epi32(ymm_a8, ymm_a9);
				ymm_tb = _mm256_unpackhi_epi32(ymm_aa, ymm_ab);
				ymm_tc = _mm256_unpacklo_epi32(ymm_ac, ymm_ad);
				ymm_td = _mm256_unpacklo_epi32(ymm_ae, ymm_af);
				ymm_te = _mm256_unpackhi_epi32(ymm_ac, ymm_ad);
				ymm_tf = _mm256_unpackhi_epi32(ymm_ae, ymm_af);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_a2 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
				ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
				ymm_a6 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
				ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
				ymm_a8 = _mm256_unpacklo_epi64(ymm_t8, ymm_t9);
				ymm_a9 = _mm256_unpackhi_epi64(ymm_t8, ymm_t9);
				ymm_aa = _mm256_unpacklo_epi64(ymm_ta, ymm_tb);
				ymm_ab = _mm256_unpackhi_epi64(ymm_ta, ymm_tb);
				ymm_ac = _mm256_unpacklo_epi64(ymm_tc, ymm_td);
				ymm_ad = _mm256_unpackhi_epi64(ymm_tc, ymm_td);
				ymm_ae = _mm256_unpacklo_epi64(ymm_te, ymm_tf);
				ymm_af = _mm256_unpackhi_epi64(ymm_te, ymm_tf);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a7);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a8);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_a9);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_aa);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ab);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ac);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ad);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_ae);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_af);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<signed short, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, signed short *b, size_t rsb) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = ptr_a0 + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			const signed short *ptr_a4 = ptr_a3 + rsa;
			const signed short *ptr_a5 = ptr_a4 + rsa;
			const signed short *ptr_a6 = ptr_a5 + rsa;
			const signed short *ptr_a7 = ptr_a6 + rsa;
			const signed short *ptr_a8 = ptr_a7 + rsa;
			const signed short *ptr_a9 = ptr_a8 + rsa;
			const signed short *ptr_aa = ptr_a9 + rsa;
			const signed short *ptr_ab = ptr_aa + rsa;
			const signed short *ptr_ac = ptr_ab + rsa;
			const signed short *ptr_ad = ptr_ac + rsa;
			const signed short *ptr_ae = ptr_ad + rsa;
			const signed short *ptr_af = ptr_ae + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j)));
				ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j)));
				ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j)));
				ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j)));
				ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j)));
				ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j)));
				ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j)));
				ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j)));
				ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j)), 1);
				ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j)), 1);
				ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j)), 1);
				ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j)), 1);
				ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j)), 1);
				ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j)), 1);
				ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j)), 1);
				ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j)), 1);
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi16(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi16(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi16(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi16(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi16(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi16(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi16(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_unpacklo_epi32(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi32(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi32(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi32(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi32(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpacklo_epi32(ymm_t6, ymm_t7);
				ymm_a6 = _mm256_unpackhi_epi32(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi32(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_unpacklo_epi64(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpackhi_epi64(ymm_a0, ymm_a1);
				ymm_t2 = _mm256_unpacklo_epi64(ymm_a2, ymm_a3);
				ymm_t3 = _mm256_unpackhi_epi64(ymm_a2, ymm_a3);
				ymm_t4 = _mm256_unpacklo_epi64(ymm_a4, ymm_a5);
				ymm_t5 = _mm256_unpackhi_epi64(ymm_a4, ymm_a5);
				ymm_t6 = _mm256_unpacklo_epi64(ymm_a6, ymm_a7);
				ymm_t7 = _mm256_unpackhi_epi64(ymm_a6, ymm_a7);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned short, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, unsigned short *b, size_t rsb) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = ptr_a0 + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			const unsigned short *ptr_a4 = ptr_a3 + rsa;
			const unsigned short *ptr_a5 = ptr_a4 + rsa;
			const unsigned short *ptr_a6 = ptr_a5 + rsa;
			const unsigned short *ptr_a7 = ptr_a6 + rsa;
			const unsigned short *ptr_a8 = ptr_a7 + rsa;
			const unsigned short *ptr_a9 = ptr_a8 + rsa;
			const unsigned short *ptr_aa = ptr_a9 + rsa;
			const unsigned short *ptr_ab = ptr_aa + rsa;
			const unsigned short *ptr_ac = ptr_ab + rsa;
			const unsigned short *ptr_ad = ptr_ac + rsa;
			const unsigned short *ptr_ae = ptr_ad + rsa;
			const unsigned short *ptr_af = ptr_ae + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j)));
				ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j)));
				ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j)));
				ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j)));
				ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4 + j)));
				ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5 + j)));
				ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6 + j)));
				ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7 + j)));
				ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j)), 1);
				ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j)), 1);
				ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j)), 1);
				ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j)), 1);
				ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j)), 1);
				ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j)), 1);
				ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j)), 1);
				ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j)), 1);
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi16(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi16(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi16(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi16(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi16(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi16(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi16(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_unpacklo_epi32(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi32(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi32(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi32(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi32(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpacklo_epi32(ymm_t6, ymm_t7);
				ymm_a6 = _mm256_unpackhi_epi32(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi32(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_unpacklo_epi64(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpackhi_epi64(ymm_a0, ymm_a1);
				ymm_t2 = _mm256_unpacklo_epi64(ymm_a2, ymm_a3);
				ymm_t3 = _mm256_unpackhi_epi64(ymm_a2, ymm_a3);
				ymm_t4 = _mm256_unpacklo_epi64(ymm_a4, ymm_a5);
				ymm_t5 = _mm256_unpackhi_epi64(ymm_a4, ymm_a5);
				ymm_t6 = _mm256_unpacklo_epi64(ymm_a6, ymm_a7);
				ymm_t7 = _mm256_unpackhi_epi64(ymm_a6, ymm_a7);
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<signed int, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, signed int *b, size_t rsb) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			const signed int *ptr_a4 = ptr_a3 + rsa;
			const signed int *ptr_a5 = ptr_a4 + rsa;
			const signed int *ptr_a6 = ptr_a5 + rsa;
			const signed int *ptr_a7 = ptr_a6 + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0 + j));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1 + j));
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2 + j));
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3 + j));
				ymm_a4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4 + j));
				ymm_a5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5 + j));
				ymm_a6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6 + j));
				ymm_a7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7 + j));
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
				ymm_a6 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t1 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t2 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t3 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t4 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t5 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t6 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t7 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 3, 0, 1));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<unsigned int, inst_avx2>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const unsigned int *a, size_t rsa, unsigned int *b, size_t rsb) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const unsigned int *ptr_a4 = ptr_a3 + rsa;
			const unsigned int *ptr_a5 = ptr_a4 + rsa;
			const unsigned int *ptr_a6 = ptr_a5 + rsa;
			const unsigned int *ptr_a7 = ptr_a6 + rsa;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0 + j));
				ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1 + j));
				ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2 + j));
				ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3 + j));
				ymm_a4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4 + j));
				ymm_a5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5 + j));
				ymm_a6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6 + j));
				ymm_a7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7 + j));
				// matrix transposed
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
				ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
				ymm_a5 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
				ymm_a6 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t1 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t2 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t3 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t4 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t5 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t6 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t7 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 3, 0, 1));
				// store data into memory
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t0);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t1);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t2);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t3);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t4);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t5);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t6);
				b += rsb;
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<float, inst_avx>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const float *a, size_t rsa, float *b, size_t rsb) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(ptr_a0 + j);
				ymm_a1 = _mm256_loadu_ps(ptr_a1 + j);
				ymm_a2 = _mm256_loadu_ps(ptr_a2 + j);
				ymm_a3 = _mm256_loadu_ps(ptr_a3 + j);
				ymm_a4 = _mm256_loadu_ps(ptr_a4 + j);
				ymm_a5 = _mm256_loadu_ps(ptr_a5 + j);
				ymm_a6 = _mm256_loadu_ps(ptr_a6 + j);
				ymm_a7 = _mm256_loadu_ps(ptr_a7 + j);
				// matrix transposed
				ymm_t0 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t1 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t2 = _mm256_shuffle_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t3 = _mm256_shuffle_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t4 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_t5 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_t6 = _mm256_shuffle_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_t7 = _mm256_shuffle_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_a0 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(2, 0, 2, 0));
				ymm_a1 = _mm256_shuffle_ps(ymm_t2, ymm_t3, _MM_SHUFFLE(2, 0, 2, 0));
				ymm_a2 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(3, 1, 3, 1));
				ymm_a3 = _mm256_shuffle_ps(ymm_t2, ymm_t3, _MM_SHUFFLE(3, 1, 3, 1));
				ymm_a4 = _mm256_shuffle_ps(ymm_t4, ymm_t5, _MM_SHUFFLE(2, 0, 2, 0));
				ymm_a5 = _mm256_shuffle_ps(ymm_t6, ymm_t7, _MM_SHUFFLE(2, 0, 2, 0));
				ymm_a6 = _mm256_shuffle_ps(ymm_t4, ymm_t5, _MM_SHUFFLE(3, 1, 3, 1));
				ymm_a7 = _mm256_shuffle_ps(ymm_t6, ymm_t7, _MM_SHUFFLE(3, 1, 3, 1));
				ymm_t0 = _mm256_permute2f128_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t1 = _mm256_permute2f128_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t2 = _mm256_permute2f128_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t3 = _mm256_permute2f128_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t4 = _mm256_permute2f128_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t5 = _mm256_permute2f128_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t6 = _mm256_permute2f128_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t7 = _mm256_permute2f128_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 3, 0, 1));
				// store data into memory
				_mm256_storeu_ps(b, ymm_t0);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t1);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t2);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t3);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t4);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t5);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t6);
				b += rsb;
				_mm256_storeu_ps(b, ymm_t7);
				b += rsb;
			}
		}
	};

	template<>
	struct block_transpose<double, inst_avx>
	{
		// b[j][i] = a[i][j]
		void operator()(size_t n, const double *a, size_t rsa, double *b, size_t rsb) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(ptr_a0 + j);
				ymm_a1 = _mm256_loadu_pd(ptr_a1 + j);
				ymm_a2 = _mm256_loadu_pd(ptr_a2 + j);
				ymm_a3 = _mm256_loadu_pd(ptr_a3 + j);
				// matrix transposed
				ymm_t0 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 0, 0));
				ymm_t1 = _mm256_shuffle_pd(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 0, 0, 0));
				ymm_t2 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 3, 3));
				ymm_t3 = _mm256_shuffle_pd(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 0, 3, 3));
				ymm_a0 = _mm256_permute2f128_pd(ymm_t0, ymm_t1, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_pd(ymm_t2, ymm_t3, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_pd(ymm_t0, ymm_t1, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a3 = _mm256_permute2f128_pd(ymm_t2, ymm_t3, _MM_SHUFFLE(0, 3, 0, 1));
				// store data into memory
				_mm256_storeu_pd(b, ymm_a0);
				b += rsb;
				_mm256_storeu_pd(b, ymm_a1);
				b += rsb;
				_mm256_storeu_pd(b, ymm_a2);
				b += rsb;
				_mm256_storeu_pd(b, ymm_a3);
				b += rsb;
			}
		}
	};

	// Class template reduce_transpose
	template<class T, size_t block_m, size_t block_n, inst_type inst>
	struct kernel_transpose
	{
		// b[j][i] = a[i][j]
		void operator()(size_t m, size_t n, const T *a, size_t rsa, T *b, size_t rsb) const
		{
			const size_t block_rsa = block_m * rsa;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t aligned_rsb = aligned_n * rsb;
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_n = n - aligned_n;
			const struct common_transpose<T> functor;
			const struct block_transpose<T, inst> special_functor;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_n > 0)
					special_functor(aligned_n, a, rsa, b, rsb);
				if (surplus_n > 0)
					functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_rsb, rsb);
				a += block_rsa;
				b += block_m;
			}
			if (surplus_m > 0)
				functor(surplus_m, n, a, rsa, b, rsb);
		}
	};

} // namespace core

#endif
