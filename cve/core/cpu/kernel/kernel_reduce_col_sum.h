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

#ifndef __CORE_CPU_KERNEL_REDUCE_COL_SUM_H__
#define __CORE_CPU_KERNEL_REDUCE_COL_SUM_H__

#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Template class common_reduce_col_sum
	template<class T1, class T2>
	struct common_reduce_col_sum
	{
		// b[j] += a[i][j]
		void operator()(size_t m, size_t n, const T1 *a, size_t rsa, T2 *b) const
		{
			for (size_t i = 0; i < m; ++i)
			{
				for (size_t j = 0; j < n; ++j)
					b[j] += static_cast<T2>(a[j]);
				a += rsa;
			}
		}
	};

	// Template class block_reduce_col_sum
	template<class T1, class T2, inst_type inst>
	struct block_reduce_col_sum
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const T1 *a, size_t rsa, T2 *b) const
		{
			const T1 *ptr_a;
			T2 val_a0, val_a1, val_a2, val_a3;

			for (size_t j = 0; j < n; j += 4)
			{
				ptr_a = a;
				// row1
				val_a0 = static_cast<T2>(ptr_a[0]);
				val_a1 = static_cast<T2>(ptr_a[1]);
				val_a2 = static_cast<T2>(ptr_a[2]);
				val_a3 = static_cast<T2>(ptr_a[3]);
				ptr_a += rsa;
				// row2
				val_a0 += static_cast<T2>(ptr_a[0]);
				val_a1 += static_cast<T2>(ptr_a[1]);
				val_a2 += static_cast<T2>(ptr_a[2]);
				val_a3 += static_cast<T2>(ptr_a[3]);
				ptr_a += rsa;
				// row3
				val_a0 += static_cast<T2>(ptr_a[0]);
				val_a1 += static_cast<T2>(ptr_a[1]);
				val_a2 += static_cast<T2>(ptr_a[2]);
				val_a3 += static_cast<T2>(ptr_a[3]);
				ptr_a += rsa;
				// row4
				val_a0 += static_cast<T2>(ptr_a[0]);
				val_a1 += static_cast<T2>(ptr_a[1]);
				val_a2 += static_cast<T2>(ptr_a[2]);
				val_a3 += static_cast<T2>(ptr_a[3]);
				a += 4;

				b[0] += val_a0;
				b[1] += val_a1;
				b[2] += val_a2;
				b[3] += val_a3;
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed char, signed int, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, signed int *b) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = ptr_a0 + rsa;
			const signed char *ptr_a2 = ptr_a1 + rsa;
			const signed char *ptr_a3 = ptr_a2 + rsa;
			const signed char *ptr_a4 = ptr_a3 + rsa;
			const signed char *ptr_a5 = ptr_a4 + rsa;
			const signed char *ptr_a6 = ptr_a5 + rsa;
			const signed char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b;

			for (size_t j = 0; j < n; j += 16)
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
				// data-type conversion
				xmm_a8 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a9 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_aa = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ab = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ad = _mm_shuffle_epi32(xmm_a5, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ae = _mm_shuffle_epi32(xmm_a6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_af = _mm_shuffle_epi32(xmm_a7, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepi8_epi16(xmm_a1);
				xmm_a2 = _mm_cvtepi8_epi16(xmm_a2);
				xmm_a3 = _mm_cvtepi8_epi16(xmm_a3);
				xmm_a4 = _mm_cvtepi8_epi16(xmm_a4);
				xmm_a5 = _mm_cvtepi8_epi16(xmm_a5);
				xmm_a6 = _mm_cvtepi8_epi16(xmm_a6);
				xmm_a7 = _mm_cvtepi8_epi16(xmm_a7);
				xmm_a8 = _mm_cvtepi8_epi16(xmm_a8);
				xmm_a9 = _mm_cvtepi8_epi16(xmm_a9);
				xmm_aa = _mm_cvtepi8_epi16(xmm_aa);
				xmm_ab = _mm_cvtepi8_epi16(xmm_ab);
				xmm_ac = _mm_cvtepi8_epi16(xmm_ac);
				xmm_ad = _mm_cvtepi8_epi16(xmm_ad);
				xmm_ae = _mm_cvtepi8_epi16(xmm_ae);
				xmm_af = _mm_cvtepi8_epi16(xmm_af);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi16(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi16(xmm_a6, xmm_a7);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_a9);
				xmm_aa = _mm_add_epi16(xmm_aa, xmm_ab);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ad);
				xmm_ae = _mm_add_epi16(xmm_ae, xmm_af);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a6);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_aa);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ae);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a4);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_ac);
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a8, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a8 = _mm_cvtepi16_epi32(xmm_a8);
				xmm_ac = _mm_cvtepi16_epi32(xmm_ac);
				// store data into memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a0));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a4));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a8));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_ac));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed char, float, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, float *b) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = ptr_a0 + rsa;
			const signed char *ptr_a2 = ptr_a1 + rsa;
			const signed char *ptr_a3 = ptr_a2 + rsa;
			const signed char *ptr_a4 = ptr_a3 + rsa;
			const signed char *ptr_a5 = ptr_a4 + rsa;
			const signed char *ptr_a6 = ptr_a5 + rsa;
			const signed char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			for (size_t j = 0; j < n; j += 16)
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
				// data-type conversion
				xmm_a8 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a9 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_aa = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ab = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ad = _mm_shuffle_epi32(xmm_a5, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ae = _mm_shuffle_epi32(xmm_a6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_af = _mm_shuffle_epi32(xmm_a7, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepi8_epi16(xmm_a1);
				xmm_a2 = _mm_cvtepi8_epi16(xmm_a2);
				xmm_a3 = _mm_cvtepi8_epi16(xmm_a3);
				xmm_a4 = _mm_cvtepi8_epi16(xmm_a4);
				xmm_a5 = _mm_cvtepi8_epi16(xmm_a5);
				xmm_a6 = _mm_cvtepi8_epi16(xmm_a6);
				xmm_a7 = _mm_cvtepi8_epi16(xmm_a7);
				xmm_a8 = _mm_cvtepi8_epi16(xmm_a8);
				xmm_a9 = _mm_cvtepi8_epi16(xmm_a9);
				xmm_aa = _mm_cvtepi8_epi16(xmm_aa);
				xmm_ab = _mm_cvtepi8_epi16(xmm_ab);
				xmm_ac = _mm_cvtepi8_epi16(xmm_ac);
				xmm_ad = _mm_cvtepi8_epi16(xmm_ad);
				xmm_ae = _mm_cvtepi8_epi16(xmm_ae);
				xmm_af = _mm_cvtepi8_epi16(xmm_af);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi16(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi16(xmm_a6, xmm_a7);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_a9);
				xmm_aa = _mm_add_epi16(xmm_aa, xmm_ab);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ad);
				xmm_ae = _mm_add_epi16(xmm_ae, xmm_af);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a6);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_aa);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ae);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a4);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_ac);
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a8, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a8 = _mm_cvtepi16_epi32(xmm_a8);
				xmm_ac = _mm_cvtepi16_epi32(xmm_ac);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a4);
				xmm_b2 = _mm_cvtepi32_ps(xmm_a8);
				xmm_b3 = _mm_cvtepi32_ps(xmm_ac);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b0));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b1));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b2));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b3));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned char, signed int, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, signed int *b) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = ptr_a0 + rsa;
			const unsigned char *ptr_a2 = ptr_a1 + rsa;
			const unsigned char *ptr_a3 = ptr_a2 + rsa;
			const unsigned char *ptr_a4 = ptr_a3 + rsa;
			const unsigned char *ptr_a5 = ptr_a4 + rsa;
			const unsigned char *ptr_a6 = ptr_a5 + rsa;
			const unsigned char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b;

			for (size_t j = 0; j < n; j += 16)
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
				// data-type conversion
				xmm_a8 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a9 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_aa = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ab = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ad = _mm_shuffle_epi32(xmm_a5, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ae = _mm_shuffle_epi32(xmm_a6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_af = _mm_shuffle_epi32(xmm_a7, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepu8_epi16(xmm_a1);
				xmm_a2 = _mm_cvtepu8_epi16(xmm_a2);
				xmm_a3 = _mm_cvtepu8_epi16(xmm_a3);
				xmm_a4 = _mm_cvtepu8_epi16(xmm_a4);
				xmm_a5 = _mm_cvtepu8_epi16(xmm_a5);
				xmm_a6 = _mm_cvtepu8_epi16(xmm_a6);
				xmm_a7 = _mm_cvtepu8_epi16(xmm_a7);
				xmm_a8 = _mm_cvtepu8_epi16(xmm_a8);
				xmm_a9 = _mm_cvtepu8_epi16(xmm_a9);
				xmm_aa = _mm_cvtepu8_epi16(xmm_aa);
				xmm_ab = _mm_cvtepu8_epi16(xmm_ab);
				xmm_ac = _mm_cvtepu8_epi16(xmm_ac);
				xmm_ad = _mm_cvtepu8_epi16(xmm_ad);
				xmm_ae = _mm_cvtepu8_epi16(xmm_ae);
				xmm_af = _mm_cvtepu8_epi16(xmm_af);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi16(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi16(xmm_a6, xmm_a7);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_a9);
				xmm_aa = _mm_add_epi16(xmm_aa, xmm_ab);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ad);
				xmm_ae = _mm_add_epi16(xmm_ae, xmm_af);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a6);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_aa);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ae);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a4);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_ac);
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a8, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a8 = _mm_cvtepi16_epi32(xmm_a8);
				xmm_ac = _mm_cvtepi16_epi32(xmm_ac);
				// store data into memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a0));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a4));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a8));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_ac));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned char, float, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, float *b) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = ptr_a0 + rsa;
			const unsigned char *ptr_a2 = ptr_a1 + rsa;
			const unsigned char *ptr_a3 = ptr_a2 + rsa;
			const unsigned char *ptr_a4 = ptr_a3 + rsa;
			const unsigned char *ptr_a5 = ptr_a4 + rsa;
			const unsigned char *ptr_a6 = ptr_a5 + rsa;
			const unsigned char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			for (size_t j = 0; j < n; j += 16)
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
				// data-type conversion
				xmm_a8 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a9 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_aa = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ab = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a4, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ad = _mm_shuffle_epi32(xmm_a5, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ae = _mm_shuffle_epi32(xmm_a6, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_af = _mm_shuffle_epi32(xmm_a7, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepu8_epi16(xmm_a1);
				xmm_a2 = _mm_cvtepu8_epi16(xmm_a2);
				xmm_a3 = _mm_cvtepu8_epi16(xmm_a3);
				xmm_a4 = _mm_cvtepu8_epi16(xmm_a4);
				xmm_a5 = _mm_cvtepu8_epi16(xmm_a5);
				xmm_a6 = _mm_cvtepu8_epi16(xmm_a6);
				xmm_a7 = _mm_cvtepu8_epi16(xmm_a7);
				xmm_a8 = _mm_cvtepu8_epi16(xmm_a8);
				xmm_a9 = _mm_cvtepu8_epi16(xmm_a9);
				xmm_aa = _mm_cvtepu8_epi16(xmm_aa);
				xmm_ab = _mm_cvtepu8_epi16(xmm_ab);
				xmm_ac = _mm_cvtepu8_epi16(xmm_ac);
				xmm_ad = _mm_cvtepu8_epi16(xmm_ad);
				xmm_ae = _mm_cvtepu8_epi16(xmm_ae);
				xmm_af = _mm_cvtepu8_epi16(xmm_af);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi16(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi16(xmm_a6, xmm_a7);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_a9);
				xmm_aa = _mm_add_epi16(xmm_aa, xmm_ab);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ad);
				xmm_ae = _mm_add_epi16(xmm_ae, xmm_af);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi16(xmm_a4, xmm_a6);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_aa);
				xmm_ac = _mm_add_epi16(xmm_ac, xmm_ae);
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a4);
				xmm_a8 = _mm_add_epi16(xmm_a8, xmm_ac);
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_ac = _mm_shuffle_epi32(xmm_a8, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a8 = _mm_cvtepi16_epi32(xmm_a8);
				xmm_ac = _mm_cvtepi16_epi32(xmm_ac);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a4);
				xmm_b2 = _mm_cvtepi32_ps(xmm_a8);
				xmm_b3 = _mm_cvtepi32_ps(xmm_ac);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b0));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b1));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b2));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b3));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed short, signed int, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, signed int *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = ptr_a0 + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a5 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a6 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a7 = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
				xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
				xmm_a3 = _mm_cvtepi16_epi32(xmm_a3);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a5 = _mm_cvtepi16_epi32(xmm_a5);
				xmm_a6 = _mm_cvtepi16_epi32(xmm_a6);
				xmm_a7 = _mm_cvtepi16_epi32(xmm_a7);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi32(xmm_a6, xmm_a7);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a6);
				// store data into memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a0));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a4));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed short, float, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, float *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = ptr_a0 + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b;
			__m128 xmm_b0, xmm_b1;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a5 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a6 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a7 = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
				xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
				xmm_a3 = _mm_cvtepi16_epi32(xmm_a3);
				xmm_a4 = _mm_cvtepi16_epi32(xmm_a4);
				xmm_a5 = _mm_cvtepi16_epi32(xmm_a5);
				xmm_a6 = _mm_cvtepi16_epi32(xmm_a6);
				xmm_a7 = _mm_cvtepi16_epi32(xmm_a7);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi32(xmm_a6, xmm_a7);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a6);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a4);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b0));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b1));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned short, signed int, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, signed int *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = ptr_a0 + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a5 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a6 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a7 = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepu16_epi32(xmm_a1);
				xmm_a2 = _mm_cvtepu16_epi32(xmm_a2);
				xmm_a3 = _mm_cvtepu16_epi32(xmm_a3);
				xmm_a4 = _mm_cvtepu16_epi32(xmm_a4);
				xmm_a5 = _mm_cvtepu16_epi32(xmm_a5);
				xmm_a6 = _mm_cvtepu16_epi32(xmm_a6);
				xmm_a7 = _mm_cvtepu16_epi32(xmm_a7);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi32(xmm_a6, xmm_a7);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a6);
				// store data into memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a0));
				b += 4;
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a4));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned short, float, inst_sse41>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, float *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = ptr_a0 + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b;
			__m128 xmm_b0, xmm_b1;

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// data-type conversion
				xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a5 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a6 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a7 = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepu16_epi32(xmm_a1);
				xmm_a2 = _mm_cvtepu16_epi32(xmm_a2);
				xmm_a3 = _mm_cvtepu16_epi32(xmm_a3);
				xmm_a4 = _mm_cvtepu16_epi32(xmm_a4);
				xmm_a5 = _mm_cvtepu16_epi32(xmm_a5);
				xmm_a6 = _mm_cvtepu16_epi32(xmm_a6);
				xmm_a7 = _mm_cvtepu16_epi32(xmm_a7);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a5);
				xmm_a6 = _mm_add_epi32(xmm_a6, xmm_a7);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_a4 = _mm_add_epi32(xmm_a4, xmm_a6);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				xmm_b1 = _mm_cvtepi32_ps(xmm_a4);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b0));
				b += 4;
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b1));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed int, signed int, inst_sse2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				// store data into memory
				xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
				_mm_storeu_si128(reinterpret_cast<__m128i*>(b), _mm_add_epi32(xmm_b, xmm_a0));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed int, float, inst_sse2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, float *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b;
			__m128 xmm_b0;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_b0 = _mm_cvtepi32_ps(xmm_a0);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_b0));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned int, float, inst_sse2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned int *a, size_t rsa, float *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi32(0x4f000000);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
				// data-type conversion
				xmm_s0 = _mm_srai_epi32(xmm_a0, 31);
				xmm_s1 = _mm_srai_epi32(xmm_a1, 31);
				xmm_s2 = _mm_srai_epi32(xmm_a2, 31);
				xmm_s3 = _mm_srai_epi32(xmm_a3, 31);
				xmm_a0 = _mm_and_si128(xmm_a0, abs);
				xmm_a1 = _mm_and_si128(xmm_a1, abs);
				xmm_a2 = _mm_and_si128(xmm_a2, abs);
				xmm_a3 = _mm_and_si128(xmm_a3, abs);
				xmm_t0 = _mm_castsi128_ps(_mm_and_si128(xmm_s0, val));
				xmm_t1 = _mm_castsi128_ps(_mm_and_si128(xmm_s1, val));
				xmm_t2 = _mm_castsi128_ps(_mm_and_si128(xmm_s2, val));
				xmm_t3 = _mm_castsi128_ps(_mm_and_si128(xmm_s3, val));
				xmm_t0 = _mm_add_ps(xmm_t0, _mm_cvtepi32_ps(xmm_a0));
				xmm_t1 = _mm_add_ps(xmm_t1, _mm_cvtepi32_ps(xmm_a1));
				xmm_t2 = _mm_add_ps(xmm_t2, _mm_cvtepi32_ps(xmm_a2));
				xmm_t3 = _mm_add_ps(xmm_t3, _mm_cvtepi32_ps(xmm_a3));
				// return the summation
				xmm_t0 = _mm_add_ps(xmm_t0, xmm_t1);
				xmm_t2 = _mm_add_ps(xmm_t2, xmm_t3);
				xmm_t0 = _mm_add_ps(xmm_t0, xmm_t2);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_t0));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<float, float, inst_sse>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const float *a, size_t rsa, float *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
				xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
				xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
				xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
				// return the summation
				xmm_a0 = _mm_add_ps(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_ps(xmm_a2, xmm_a3);
				xmm_a0 = _mm_add_ps(xmm_a0, xmm_a2);
				// store data into memory
				_mm_storeu_ps(b, _mm_add_ps(_mm_loadu_ps(b), xmm_a0));
				b += 4;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<double, double, inst_sse2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			__m128d xmm_a0, xmm_a1;

			for (size_t j = 0; j < n; j += 2)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(ptr_a0 + j);
				xmm_a1 = _mm_loadu_pd(ptr_a1 + j);
				// return the summation
				xmm_a0 = _mm_add_pd(xmm_a0, xmm_a1);
				// store data into memory
				_mm_storeu_pd(b, _mm_add_pd(_mm_loadu_pd(b), xmm_a0));
				b += 2;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed char, signed int, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, signed int *b) const
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;

			for (size_t j = 0; j < n; j += 16)
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
				xmm_a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_a9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_aa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_ac = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_ad = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_ae = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_af = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j));
				// data-type conversion
				ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
				ymm_a1 = _mm256_cvtepi8_epi16(xmm_a1);
				ymm_a2 = _mm256_cvtepi8_epi16(xmm_a2);
				ymm_a3 = _mm256_cvtepi8_epi16(xmm_a3);
				ymm_a4 = _mm256_cvtepi8_epi16(xmm_a4);
				ymm_a5 = _mm256_cvtepi8_epi16(xmm_a5);
				ymm_a6 = _mm256_cvtepi8_epi16(xmm_a6);
				ymm_a7 = _mm256_cvtepi8_epi16(xmm_a7);
				ymm_a8 = _mm256_cvtepi8_epi16(xmm_a8);
				ymm_a9 = _mm256_cvtepi8_epi16(xmm_a9);
				ymm_aa = _mm256_cvtepi8_epi16(xmm_aa);
				ymm_ab = _mm256_cvtepi8_epi16(xmm_ab);
				ymm_ac = _mm256_cvtepi8_epi16(xmm_ac);
				ymm_ad = _mm256_cvtepi8_epi16(xmm_ad);
				ymm_ae = _mm256_cvtepi8_epi16(xmm_ae);
				ymm_af = _mm256_cvtepi8_epi16(xmm_af);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_a9);
				ymm_aa = _mm256_add_epi16(ymm_aa, ymm_ab);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ad);
				ymm_ae = _mm256_add_epi16(ymm_ae, ymm_af);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_aa);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ae);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_ac);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a8);
				// data-type conversion
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				// store data into memory
				__m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a0));
				b += 8;
				ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a1));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed char, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed char *a, size_t rsa, float *b) const
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256 ymm_b0, ymm_b1;

			for (size_t j = 0; j < n; j += 16)
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
				xmm_a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_a9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_aa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_ac = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_ad = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_ae = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_af = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j));
				// data-type conversion
				ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
				ymm_a1 = _mm256_cvtepi8_epi16(xmm_a1);
				ymm_a2 = _mm256_cvtepi8_epi16(xmm_a2);
				ymm_a3 = _mm256_cvtepi8_epi16(xmm_a3);
				ymm_a4 = _mm256_cvtepi8_epi16(xmm_a4);
				ymm_a5 = _mm256_cvtepi8_epi16(xmm_a5);
				ymm_a6 = _mm256_cvtepi8_epi16(xmm_a6);
				ymm_a7 = _mm256_cvtepi8_epi16(xmm_a7);
				ymm_a8 = _mm256_cvtepi8_epi16(xmm_a8);
				ymm_a9 = _mm256_cvtepi8_epi16(xmm_a9);
				ymm_aa = _mm256_cvtepi8_epi16(xmm_aa);
				ymm_ab = _mm256_cvtepi8_epi16(xmm_ab);
				ymm_ac = _mm256_cvtepi8_epi16(xmm_ac);
				ymm_ad = _mm256_cvtepi8_epi16(xmm_ad);
				ymm_ae = _mm256_cvtepi8_epi16(xmm_ae);
				ymm_af = _mm256_cvtepi8_epi16(xmm_af);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_a9);
				ymm_aa = _mm256_add_epi16(ymm_aa, ymm_ab);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ad);
				ymm_ae = _mm256_add_epi16(ymm_ae, ymm_af);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_aa);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ae);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_ac);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a8);
				// data-type conversion
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				ymm_b1 = _mm256_cvtepi32_ps(ymm_a1);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b0));
				b += 8;
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b1));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned char, signed int, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, signed int *b) const
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;

			for (size_t j = 0; j < n; j += 16)
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
				xmm_a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_a9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_aa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_ac = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_ad = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_ae = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_af = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j));
				// data-type conversion
				ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
				ymm_a1 = _mm256_cvtepu8_epi16(xmm_a1);
				ymm_a2 = _mm256_cvtepu8_epi16(xmm_a2);
				ymm_a3 = _mm256_cvtepu8_epi16(xmm_a3);
				ymm_a4 = _mm256_cvtepu8_epi16(xmm_a4);
				ymm_a5 = _mm256_cvtepu8_epi16(xmm_a5);
				ymm_a6 = _mm256_cvtepu8_epi16(xmm_a6);
				ymm_a7 = _mm256_cvtepu8_epi16(xmm_a7);
				ymm_a8 = _mm256_cvtepu8_epi16(xmm_a8);
				ymm_a9 = _mm256_cvtepu8_epi16(xmm_a9);
				ymm_aa = _mm256_cvtepu8_epi16(xmm_aa);
				ymm_ab = _mm256_cvtepu8_epi16(xmm_ab);
				ymm_ac = _mm256_cvtepu8_epi16(xmm_ac);
				ymm_ad = _mm256_cvtepu8_epi16(xmm_ad);
				ymm_ae = _mm256_cvtepu8_epi16(xmm_ae);
				ymm_af = _mm256_cvtepu8_epi16(xmm_af);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_a9);
				ymm_aa = _mm256_add_epi16(ymm_aa, ymm_ab);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ad);
				ymm_ae = _mm256_add_epi16(ymm_ae, ymm_af);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_aa);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ae);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_ac);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a8);
				// data-type conversion
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				// store data into memory
				__m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a0));
				b += 8;
				ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a1));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned char, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned char *a, size_t rsa, float *b) const
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256 ymm_b0, ymm_b1;

			for (size_t j = 0; j < n; j += 16)
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
				xmm_a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8 + j));
				xmm_a9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9 + j));
				xmm_aa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa + j));
				xmm_ab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab + j));
				xmm_ac = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac + j));
				xmm_ad = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad + j));
				xmm_ae = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae + j));
				xmm_af = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af + j));
				// data-type conversion
				ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
				ymm_a1 = _mm256_cvtepu8_epi16(xmm_a1);
				ymm_a2 = _mm256_cvtepu8_epi16(xmm_a2);
				ymm_a3 = _mm256_cvtepu8_epi16(xmm_a3);
				ymm_a4 = _mm256_cvtepu8_epi16(xmm_a4);
				ymm_a5 = _mm256_cvtepu8_epi16(xmm_a5);
				ymm_a6 = _mm256_cvtepu8_epi16(xmm_a6);
				ymm_a7 = _mm256_cvtepu8_epi16(xmm_a7);
				ymm_a8 = _mm256_cvtepu8_epi16(xmm_a8);
				ymm_a9 = _mm256_cvtepu8_epi16(xmm_a9);
				ymm_aa = _mm256_cvtepu8_epi16(xmm_aa);
				ymm_ab = _mm256_cvtepu8_epi16(xmm_ab);
				ymm_ac = _mm256_cvtepu8_epi16(xmm_ac);
				ymm_ad = _mm256_cvtepu8_epi16(xmm_ad);
				ymm_ae = _mm256_cvtepu8_epi16(xmm_ae);
				ymm_af = _mm256_cvtepu8_epi16(xmm_af);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_a9);
				ymm_aa = _mm256_add_epi16(ymm_aa, ymm_ab);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ad);
				ymm_ae = _mm256_add_epi16(ymm_ae, ymm_af);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_aa);
				ymm_ac = _mm256_add_epi16(ymm_ac, ymm_ae);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_a8 = _mm256_add_epi16(ymm_a8, ymm_ac);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a8);
				// data-type conversion
				xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
				xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				ymm_b1 = _mm256_cvtepi32_ps(ymm_a1);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b0));
				b += 8;
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b1));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed short, signed int, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;

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
				// data-type conversion
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				ymm_a2 = _mm256_cvtepi16_epi32(xmm_a2);
				ymm_a3 = _mm256_cvtepi16_epi32(xmm_a3);
				ymm_a4 = _mm256_cvtepi16_epi32(xmm_a4);
				ymm_a5 = _mm256_cvtepi16_epi32(xmm_a5);
				ymm_a6 = _mm256_cvtepi16_epi32(xmm_a6);
				ymm_a7 = _mm256_cvtepi16_epi32(xmm_a7);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				// store data into memory
				__m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed short, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed short *a, size_t rsa, float *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;

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
				// data-type conversion
				ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
				ymm_a2 = _mm256_cvtepi16_epi32(xmm_a2);
				ymm_a3 = _mm256_cvtepi16_epi32(xmm_a3);
				ymm_a4 = _mm256_cvtepi16_epi32(xmm_a4);
				ymm_a5 = _mm256_cvtepi16_epi32(xmm_a5);
				ymm_a6 = _mm256_cvtepi16_epi32(xmm_a6);
				ymm_a7 = _mm256_cvtepi16_epi32(xmm_a7);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned short, signed int, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;

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
				// data-type conversion
				ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepu16_epi32(xmm_a1);
				ymm_a2 = _mm256_cvtepu16_epi32(xmm_a2);
				ymm_a3 = _mm256_cvtepu16_epi32(xmm_a3);
				ymm_a4 = _mm256_cvtepu16_epi32(xmm_a4);
				ymm_a5 = _mm256_cvtepu16_epi32(xmm_a5);
				ymm_a6 = _mm256_cvtepu16_epi32(xmm_a6);
				ymm_a7 = _mm256_cvtepu16_epi32(xmm_a7);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				// store data into memory
				__m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned short, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned short *a, size_t rsa, float *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;

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
				// data-type conversion
				ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
				ymm_a1 = _mm256_cvtepu16_epi32(xmm_a1);
				ymm_a2 = _mm256_cvtepu16_epi32(xmm_a2);
				ymm_a3 = _mm256_cvtepu16_epi32(xmm_a3);
				ymm_a4 = _mm256_cvtepu16_epi32(xmm_a4);
				ymm_a5 = _mm256_cvtepu16_epi32(xmm_a5);
				ymm_a6 = _mm256_cvtepu16_epi32(xmm_a6);
				ymm_a7 = _mm256_cvtepu16_epi32(xmm_a7);
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed int, signed int, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, signed int *b) const
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
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				// store data into memory
				__m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), _mm256_add_epi32(ymm_b, ymm_a0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<signed int, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const signed int *a, size_t rsa, float *b) const
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
			__m256 ymm_b0;

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
				// return the summation
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_epi16(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_epi16(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_epi16(ymm_a0, ymm_a4);
				ymm_b0 = _mm256_cvtepi32_ps(ymm_a0);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_b0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<unsigned int, float, inst_avx2>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const unsigned int *a, size_t rsa, float *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const unsigned int *ptr_a4 = ptr_a3 + rsa;
			const unsigned int *ptr_a5 = ptr_a4 + rsa;
			const unsigned int *ptr_a6 = ptr_a5 + rsa;
			const unsigned int *ptr_a7 = ptr_a6 + rsa;
			const __m256i abs = _mm256_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi32(0x4f000000);
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3, ymm_s4, ymm_s5, ymm_s6, ymm_s7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

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
				// data-type conversion
				ymm_s0 = _mm256_srai_epi32(ymm_a0, 31);
				ymm_s1 = _mm256_srai_epi32(ymm_a1, 31);
				ymm_s2 = _mm256_srai_epi32(ymm_a2, 31);
				ymm_s3 = _mm256_srai_epi32(ymm_a3, 31);
				ymm_s4 = _mm256_srai_epi32(ymm_a4, 31);
				ymm_s5 = _mm256_srai_epi32(ymm_a5, 31);
				ymm_s6 = _mm256_srai_epi32(ymm_a6, 31);
				ymm_s7 = _mm256_srai_epi32(ymm_a7, 31);
				ymm_a0 = _mm256_and_si256(ymm_a0, abs);
				ymm_a1 = _mm256_and_si256(ymm_a1, abs);
				ymm_a2 = _mm256_and_si256(ymm_a2, abs);
				ymm_a3 = _mm256_and_si256(ymm_a3, abs);
				ymm_a4 = _mm256_and_si256(ymm_a4, abs);
				ymm_a5 = _mm256_and_si256(ymm_a5, abs);
				ymm_a6 = _mm256_and_si256(ymm_a6, abs);
				ymm_a7 = _mm256_and_si256(ymm_a7, abs);
				ymm_t0 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s0, val));
				ymm_t1 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s1, val));
				ymm_t2 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s2, val));
				ymm_t3 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s3, val));
				ymm_t4 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s4, val));
				ymm_t5 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s5, val));
				ymm_t6 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s6, val));
				ymm_t7 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s7, val));
				ymm_t0 = _mm256_add_ps(ymm_t0, _mm256_cvtepi32_ps(ymm_a0));
				ymm_t1 = _mm256_add_ps(ymm_t1, _mm256_cvtepi32_ps(ymm_a1));
				ymm_t2 = _mm256_add_ps(ymm_t2, _mm256_cvtepi32_ps(ymm_a2));
				ymm_t3 = _mm256_add_ps(ymm_t3, _mm256_cvtepi32_ps(ymm_a3));
				ymm_t4 = _mm256_add_ps(ymm_t4, _mm256_cvtepi32_ps(ymm_a4));
				ymm_t5 = _mm256_add_ps(ymm_t5, _mm256_cvtepi32_ps(ymm_a5));
				ymm_t6 = _mm256_add_ps(ymm_t6, _mm256_cvtepi32_ps(ymm_a6));
				ymm_t7 = _mm256_add_ps(ymm_t7, _mm256_cvtepi32_ps(ymm_a7));
				// return the summation
				ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t1);
				ymm_t2 = _mm256_add_ps(ymm_t2, ymm_t3);
				ymm_t4 = _mm256_add_ps(ymm_t4, ymm_t5);
				ymm_t6 = _mm256_add_ps(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t2);
				ymm_t4 = _mm256_add_ps(ymm_t4, ymm_t6);
				ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t4);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_t0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<float, float, inst_avx>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const float *a, size_t rsa, float *b) const
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
				// return the summation
				ymm_a0 = _mm256_add_ps(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_ps(ymm_a2, ymm_a3);
				ymm_a4 = _mm256_add_ps(ymm_a4, ymm_a5);
				ymm_a6 = _mm256_add_ps(ymm_a6, ymm_a7);
				ymm_a0 = _mm256_add_ps(ymm_a0, ymm_a2);
				ymm_a4 = _mm256_add_ps(ymm_a4, ymm_a6);
				ymm_a0 = _mm256_add_ps(ymm_a0, ymm_a4);
				// store data into memory
				_mm256_storeu_ps(b, _mm256_add_ps(_mm256_loadu_ps(b), ymm_a0));
				b += 8;
			}
		}
	};

	template<>
	struct block_reduce_col_sum<double, double, inst_avx>
	{
		// b[j] += a[i][j]
		void operator()(size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(ptr_a0 + j);
				ymm_a1 = _mm256_loadu_pd(ptr_a1 + j);
				ymm_a2 = _mm256_loadu_pd(ptr_a2 + j);
				ymm_a3 = _mm256_loadu_pd(ptr_a3 + j);
				// return the summation
				ymm_a0 = _mm256_add_pd(ymm_a0, ymm_a1);
				ymm_a2 = _mm256_add_pd(ymm_a2, ymm_a3);
				ymm_a0 = _mm256_add_pd(ymm_a0, ymm_a2);
				// store data into memory
				_mm256_storeu_pd(b, _mm256_add_pd(_mm256_loadu_pd(b), ymm_a0));
				b += 4;
			}
		}
	};

	// Template class kernel_reduce_col_sum
	template<class T1, class T2, size_t block_m, size_t block_n, inst_type inst>
	struct kernel_reduce_col_sum
	{
		// b[j] += a[i][j]
		void operator()(size_t m, size_t n, const T1 *a, size_t rsa, T2 *b) const
		{
			const size_t block_rsa = block_m * rsa;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_n = n - aligned_n;
			const struct common_reduce_col_sum<T1, T2> functor;
			const struct block_reduce_col_sum<T1, T2, inst> special_functor;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_n > 0)
					special_functor(aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_n);
				a += block_rsa;
			}
			if (surplus_m > 0)
				functor(surplus_m, n, a, rsa, b);
		}
	};

} // namespace core

#endif
