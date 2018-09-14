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

#ifndef __CORE_CPU_BLOCK_MINT_H__
#define __CORE_CPU_BLOCK_MINT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_mint

	template<class T, cpu_inst_type inst>
	struct block_mint
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t, size_t n, const T *a, size_t rsa, T *b) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = a + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			T val_a0, val_a1, val_a2, val_a3;
			T val_b0;

			for (size_t j = 0; j < n; ++j)
			{
				val_a0 = ptr_a0[j];
				val_a1 = ptr_a1[j];
				val_a2 = ptr_a2[j];
				val_a3 = ptr_a3[j];
				val_b0 = b[j];
				// return minimum values
				val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
				val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
				val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
				b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
			}
		}
	};

	template<>
	struct block_mint<signed char, cpu_sse41>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = a + rsa;
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
			signed char val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af;
			signed char val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 16)
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
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epi8(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epi8(xmm_a2, xmm_a3);
					xmm_a4 = _mm_min_epi8(xmm_a4, xmm_a5);
					xmm_a6 = _mm_min_epi8(xmm_a6, xmm_a7);
					xmm_a8 = _mm_min_epi8(xmm_a8, xmm_a9);
					xmm_aa = _mm_min_epi8(xmm_aa, xmm_ab);
					xmm_ac = _mm_min_epi8(xmm_ac, xmm_ad);
					xmm_ae = _mm_min_epi8(xmm_ae, xmm_af);
					xmm_a0 = _mm_min_epi8(xmm_a0, xmm_a2);
					xmm_a4 = _mm_min_epi8(xmm_a4, xmm_a6);
					xmm_a8 = _mm_min_epi8(xmm_a8, xmm_aa);
					xmm_ac = _mm_min_epi8(xmm_ac, xmm_ae);
					xmm_a0 = _mm_min_epi8(xmm_a0, xmm_a4);
					xmm_a8 = _mm_min_epi8(xmm_a8, xmm_ac);
					xmm_a0 = _mm_min_epi8(xmm_a0, xmm_a8);
					xmm_b0 = _mm_min_epi8(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned char, cpu_sse2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = a + rsa;
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
			signed char val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af;
			unsigned char val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 16)
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
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epu8(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epu8(xmm_a2, xmm_a3);
					xmm_a4 = _mm_min_epu8(xmm_a4, xmm_a5);
					xmm_a6 = _mm_min_epu8(xmm_a6, xmm_a7);
					xmm_a8 = _mm_min_epu8(xmm_a8, xmm_a9);
					xmm_aa = _mm_min_epu8(xmm_aa, xmm_ab);
					xmm_ac = _mm_min_epu8(xmm_ac, xmm_ad);
					xmm_ae = _mm_min_epu8(xmm_ae, xmm_af);
					xmm_a0 = _mm_min_epu8(xmm_a0, xmm_a2);
					xmm_a4 = _mm_min_epu8(xmm_a4, xmm_a6);
					xmm_a8 = _mm_min_epu8(xmm_a8, xmm_aa);
					xmm_ac = _mm_min_epu8(xmm_ac, xmm_ae);
					xmm_a0 = _mm_min_epu8(xmm_a0, xmm_a4);
					xmm_a8 = _mm_min_epu8(xmm_a8, xmm_ac);
					xmm_a0 = _mm_min_epu8(xmm_a0, xmm_a8);
					xmm_b0 = _mm_min_epu8(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<signed short, cpu_sse2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = a + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			const signed short *ptr_a4 = ptr_a3 + rsa;
			const signed short *ptr_a5 = ptr_a4 + rsa;
			const signed short *ptr_a6 = ptr_a5 + rsa;
			const signed short *ptr_a7 = ptr_a6 + rsa;
			signed short val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7;
			signed short val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epi16(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epi16(xmm_a2, xmm_a3);
					xmm_a4 = _mm_min_epi16(xmm_a4, xmm_a5);
					xmm_a6 = _mm_min_epi16(xmm_a6, xmm_a7);
					xmm_a0 = _mm_min_epi16(xmm_a0, xmm_a2);
					xmm_a4 = _mm_min_epi16(xmm_a4, xmm_a6);
					xmm_a0 = _mm_min_epi16(xmm_a0, xmm_a4);
					xmm_b0 = _mm_min_epi16(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned short, cpu_sse41>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = a + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			const unsigned short *ptr_a4 = ptr_a3 + rsa;
			const unsigned short *ptr_a5 = ptr_a4 + rsa;
			const unsigned short *ptr_a6 = ptr_a5 + rsa;
			const unsigned short *ptr_a7 = ptr_a6 + rsa;
			unsigned short val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7;
			unsigned short val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epu16(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epu16(xmm_a2, xmm_a3);
					xmm_a4 = _mm_min_epu16(xmm_a4, xmm_a5);
					xmm_a6 = _mm_min_epu16(xmm_a6, xmm_a7);
					xmm_a0 = _mm_min_epu16(xmm_a0, xmm_a2);
					xmm_a4 = _mm_min_epu16(xmm_a4, xmm_a6);
					xmm_a0 = _mm_min_epu16(xmm_a0, xmm_a4);
					xmm_b0 = _mm_min_epu16(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<signed int, cpu_sse41>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = a + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			signed int val_a0, val_a1, val_a2, val_a3;
			signed int val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epi32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epi32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_min_epi32(xmm_a0, xmm_a2);
					xmm_b0 = _mm_min_epi32(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned int, cpu_sse41>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = a + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			unsigned int val_a0, val_a1, val_a2, val_a3;
			unsigned int val_b0;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
					// return minimum values
					xmm_a0 = _mm_min_epu32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_epu32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_min_epu32(xmm_a0, xmm_a2);
					xmm_b0 = _mm_min_epu32(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<float, cpu_sse>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			float val_a0, val_a1, val_a2, val_a3;
			float val_b0;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
					xmm_b0 = _mm_loadu_ps(b + j);
					// return minimum values
					xmm_a0 = _mm_min_ps(xmm_a0, xmm_a1);
					xmm_a2 = _mm_min_ps(xmm_a2, xmm_a3);
					xmm_a0 = _mm_min_ps(xmm_a0, xmm_a2);
					xmm_b0 = _mm_min_ps(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_ps(b + j, xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<double, cpu_sse2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			double val_a0, val_a1;
			double val_b0;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(ptr_a0 + j);
					xmm_a1 = _mm_loadu_pd(ptr_a1 + j);
					xmm_b0 = _mm_loadu_pd(b + j);
					// return minimum values
					xmm_a0 = _mm_min_pd(xmm_a0, xmm_a1);
					xmm_b0 = _mm_min_pd(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_pd(b + j, xmm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<signed char, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = a + rsa;
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
			signed char val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af,
				val_ag, val_ah, val_ai, val_aj, val_ak, val_al, val_am, val_an, val_ao, val_ap, val_aq, val_ar, val_as, val_at, val_au, val_av;
			signed char val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af,
				ymm_ag, ymm_ah, ymm_ai, ymm_aj, ymm_ak, ymm_al, ymm_am, ymm_an, ymm_ao, ymm_ap, ymm_aq, ymm_ar, ymm_as, ymm_at, ymm_au, ymm_av;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 32)
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
					ymm_a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8 + j));
					ymm_a9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9 + j));
					ymm_aa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa + j));
					ymm_ab = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab + j));
					ymm_ac = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac + j));
					ymm_ad = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad + j));
					ymm_ae = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae + j));
					ymm_af = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af + j));
					ymm_ag = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ag + j));
					ymm_ah = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ah + j));
					ymm_ai = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ai + j));
					ymm_aj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aj + j));
					ymm_ak = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ak + j));
					ymm_al = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_al + j));
					ymm_am = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_am + j));
					ymm_an = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_an + j));
					ymm_ao = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ao + j));
					ymm_ap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ap + j));
					ymm_aq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aq + j));
					ymm_ar = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ar + j));
					ymm_as = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_as + j));
					ymm_at = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_at + j));
					ymm_au = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_au + j));
					ymm_av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_av + j));
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epi8(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epi8(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epi8(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epi8(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_min_epi8(ymm_a8, ymm_a9);
					ymm_aa = _mm256_min_epi8(ymm_aa, ymm_ab);
					ymm_ac = _mm256_min_epi8(ymm_ac, ymm_ad);
					ymm_ae = _mm256_min_epi8(ymm_ae, ymm_af);
					ymm_ag = _mm256_min_epi8(ymm_ag, ymm_ah);
					ymm_ai = _mm256_min_epi8(ymm_ai, ymm_aj);
					ymm_ak = _mm256_min_epi8(ymm_ak, ymm_al);
					ymm_am = _mm256_min_epi8(ymm_am, ymm_an);
					ymm_ao = _mm256_min_epi8(ymm_ao, ymm_ap);
					ymm_aq = _mm256_min_epi8(ymm_aq, ymm_ar);
					ymm_as = _mm256_min_epi8(ymm_as, ymm_at);
					ymm_au = _mm256_min_epi8(ymm_au, ymm_av);
					ymm_a0 = _mm256_min_epi8(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epi8(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_min_epi8(ymm_a8, ymm_aa);
					ymm_ac = _mm256_min_epi8(ymm_ac, ymm_ae);
					ymm_ag = _mm256_min_epi8(ymm_ag, ymm_ai);
					ymm_ak = _mm256_min_epi8(ymm_ak, ymm_am);
					ymm_ao = _mm256_min_epi8(ymm_ao, ymm_aq);
					ymm_as = _mm256_min_epi8(ymm_as, ymm_au);
					ymm_a0 = _mm256_min_epi8(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_min_epi8(ymm_a8, ymm_ac);
					ymm_ag = _mm256_min_epi8(ymm_ag, ymm_ak);
					ymm_ao = _mm256_min_epi8(ymm_ao, ymm_as);
					ymm_a0 = _mm256_min_epi8(ymm_a0, ymm_a8);
					ymm_ag = _mm256_min_epi8(ymm_ag, ymm_ao);
					ymm_a0 = _mm256_min_epi8(ymm_a0, ymm_ag);
					ymm_b0 = _mm256_min_epi8(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_ag = ptr_ag[j];
					val_ah = ptr_ah[j];
					val_ai = ptr_ai[j];
					val_aj = ptr_aj[j];
					val_ak = ptr_ak[j];
					val_al = ptr_al[j];
					val_am = ptr_am[j];
					val_an = ptr_an[j];
					val_ao = ptr_ao[j];
					val_ap = ptr_ap[j];
					val_aq = ptr_aq[j];
					val_ar = ptr_ar[j];
					val_as = ptr_as[j];
					val_at = ptr_at[j];
					val_au = ptr_au[j];
					val_av = ptr_av[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_ag = val_ag < val_ah ? val_ag : val_ah;
					val_ai = val_ai < val_aj ? val_ai : val_aj;
					val_ak = val_ak < val_al ? val_ak : val_al;
					val_am = val_am < val_an ? val_am : val_an;
					val_ao = val_ao < val_ap ? val_ao : val_ap;
					val_aq = val_aq < val_ar ? val_aq : val_ar;
					val_as = val_as < val_at ? val_as : val_at;
					val_au = val_au < val_av ? val_au : val_av;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_ag = val_ag < val_ai ? val_ag : val_ai;
					val_ak = val_ak < val_am ? val_ak : val_am;
					val_ao = val_ao < val_aq ? val_ao : val_aq;
					val_as = val_as < val_au ? val_as : val_au;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_ag = val_ag < val_ak ? val_ag : val_ak;
					val_ao = val_ao < val_as ? val_ao : val_as;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					val_ag = val_ag < val_ao ? val_ag : val_ao;
					val_a0 = val_a0 < val_ag ? val_a0 : val_ag;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned char, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = a + rsa;
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
			unsigned char val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af,
				val_ag, val_ah, val_ai, val_aj, val_ak, val_al, val_am, val_an, val_ao, val_ap, val_aq, val_ar, val_as, val_at, val_au, val_av;
			unsigned char val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af,
				ymm_ag, ymm_ah, ymm_ai, ymm_aj, ymm_ak, ymm_al, ymm_am, ymm_an, ymm_ao, ymm_ap, ymm_aq, ymm_ar, ymm_as, ymm_at, ymm_au, ymm_av;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 32)
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
					ymm_a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8 + j));
					ymm_a9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9 + j));
					ymm_aa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa + j));
					ymm_ab = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab + j));
					ymm_ac = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac + j));
					ymm_ad = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad + j));
					ymm_ae = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae + j));
					ymm_af = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af + j));
					ymm_ag = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ag + j));
					ymm_ah = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ah + j));
					ymm_ai = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ai + j));
					ymm_aj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aj + j));
					ymm_ak = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ak + j));
					ymm_al = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_al + j));
					ymm_am = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_am + j));
					ymm_an = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_an + j));
					ymm_ao = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ao + j));
					ymm_ap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ap + j));
					ymm_aq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aq + j));
					ymm_ar = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ar + j));
					ymm_as = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_as + j));
					ymm_at = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_at + j));
					ymm_au = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_au + j));
					ymm_av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_av + j));
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epu8(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epu8(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epu8(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epu8(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_min_epu8(ymm_a8, ymm_a9);
					ymm_aa = _mm256_min_epu8(ymm_aa, ymm_ab);
					ymm_ac = _mm256_min_epu8(ymm_ac, ymm_ad);
					ymm_ae = _mm256_min_epu8(ymm_ae, ymm_af);
					ymm_ag = _mm256_min_epu8(ymm_ag, ymm_ah);
					ymm_ai = _mm256_min_epu8(ymm_ai, ymm_aj);
					ymm_ak = _mm256_min_epu8(ymm_ak, ymm_al);
					ymm_am = _mm256_min_epu8(ymm_am, ymm_an);
					ymm_ao = _mm256_min_epu8(ymm_ao, ymm_ap);
					ymm_aq = _mm256_min_epu8(ymm_aq, ymm_ar);
					ymm_as = _mm256_min_epu8(ymm_as, ymm_at);
					ymm_au = _mm256_min_epu8(ymm_au, ymm_av);
					ymm_a0 = _mm256_min_epu8(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epu8(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_min_epu8(ymm_a8, ymm_aa);
					ymm_ac = _mm256_min_epu8(ymm_ac, ymm_ae);
					ymm_ag = _mm256_min_epu8(ymm_ag, ymm_ai);
					ymm_ak = _mm256_min_epu8(ymm_ak, ymm_am);
					ymm_ao = _mm256_min_epu8(ymm_ao, ymm_aq);
					ymm_as = _mm256_min_epu8(ymm_as, ymm_au);
					ymm_a0 = _mm256_min_epu8(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_min_epu8(ymm_a8, ymm_ac);
					ymm_ag = _mm256_min_epu8(ymm_ag, ymm_ak);
					ymm_ao = _mm256_min_epu8(ymm_ao, ymm_as);
					ymm_a0 = _mm256_min_epu8(ymm_a0, ymm_a8);
					ymm_ag = _mm256_min_epu8(ymm_ag, ymm_ao);
					ymm_a0 = _mm256_min_epu8(ymm_a0, ymm_ag);
					ymm_b0 = _mm256_min_epu8(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_ag = ptr_ag[j];
					val_ah = ptr_ah[j];
					val_ai = ptr_ai[j];
					val_aj = ptr_aj[j];
					val_ak = ptr_ak[j];
					val_al = ptr_al[j];
					val_am = ptr_am[j];
					val_an = ptr_an[j];
					val_ao = ptr_ao[j];
					val_ap = ptr_ap[j];
					val_aq = ptr_aq[j];
					val_ar = ptr_ar[j];
					val_as = ptr_as[j];
					val_at = ptr_at[j];
					val_au = ptr_au[j];
					val_av = ptr_av[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_ag = val_ag < val_ah ? val_ag : val_ah;
					val_ai = val_ai < val_aj ? val_ai : val_aj;
					val_ak = val_ak < val_al ? val_ak : val_al;
					val_am = val_am < val_an ? val_am : val_an;
					val_ao = val_ao < val_ap ? val_ao : val_ap;
					val_aq = val_aq < val_ar ? val_aq : val_ar;
					val_as = val_as < val_at ? val_as : val_at;
					val_au = val_au < val_av ? val_au : val_av;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_ag = val_ag < val_ai ? val_ag : val_ai;
					val_ak = val_ak < val_am ? val_ak : val_am;
					val_ao = val_ao < val_aq ? val_ao : val_aq;
					val_as = val_as < val_au ? val_as : val_au;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_ag = val_ag < val_ak ? val_ag : val_ak;
					val_ao = val_ao < val_as ? val_ao : val_as;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					val_ag = val_ag < val_ao ? val_ag : val_ao;
					val_a0 = val_a0 < val_ag ? val_a0 : val_ag;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<signed short, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = a + rsa;
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
			signed short val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af;
			signed short val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 16)
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
					ymm_a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8 + j));
					ymm_a9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9 + j));
					ymm_aa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa + j));
					ymm_ab = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab + j));
					ymm_ac = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac + j));
					ymm_ad = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad + j));
					ymm_ae = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae + j));
					ymm_af = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af + j));
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epi16(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epi16(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epi16(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epi16(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_min_epi16(ymm_a8, ymm_a9);
					ymm_aa = _mm256_min_epi16(ymm_aa, ymm_ab);
					ymm_ac = _mm256_min_epi16(ymm_ac, ymm_ad);
					ymm_ae = _mm256_min_epi16(ymm_ae, ymm_af);
					ymm_a0 = _mm256_min_epi16(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epi16(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_min_epi16(ymm_a8, ymm_aa);
					ymm_ac = _mm256_min_epi16(ymm_ac, ymm_ae);
					ymm_a0 = _mm256_min_epi16(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_min_epi16(ymm_a8, ymm_ac);
					ymm_a0 = _mm256_min_epi16(ymm_a0, ymm_a8);
					ymm_b0 = _mm256_min_epi16(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned short, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = a + rsa;
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
			unsigned short val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_aa, val_ab, val_ac, val_ad, val_ae, val_af;
			unsigned short val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 16)
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
					ymm_a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8 + j));
					ymm_a9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9 + j));
					ymm_aa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa + j));
					ymm_ab = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab + j));
					ymm_ac = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac + j));
					ymm_ad = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad + j));
					ymm_ae = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae + j));
					ymm_af = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af + j));
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epu16(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epu16(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epu16(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epu16(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_min_epu16(ymm_a8, ymm_a9);
					ymm_aa = _mm256_min_epu16(ymm_aa, ymm_ab);
					ymm_ac = _mm256_min_epu16(ymm_ac, ymm_ad);
					ymm_ae = _mm256_min_epu16(ymm_ae, ymm_af);
					ymm_a0 = _mm256_min_epu16(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epu16(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_min_epu16(ymm_a8, ymm_aa);
					ymm_ac = _mm256_min_epu16(ymm_ac, ymm_ae);
					ymm_a0 = _mm256_min_epu16(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_min_epu16(ymm_a8, ymm_ac);
					ymm_a0 = _mm256_min_epu16(ymm_a0, ymm_a8);
					ymm_b0 = _mm256_min_epu16(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_a8 = ptr_a8[j];
					val_a9 = ptr_a9[j];
					val_aa = ptr_aa[j];
					val_ab = ptr_ab[j];
					val_ac = ptr_ac[j];
					val_ad = ptr_ad[j];
					val_ae = ptr_ae[j];
					val_af = ptr_af[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a8 = val_a8 < val_a9 ? val_a8 : val_a9;
					val_aa = val_aa < val_ab ? val_aa : val_ab;
					val_ac = val_ac < val_ad ? val_ac : val_ad;
					val_ae = val_ae < val_af ? val_ae : val_af;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a8 = val_a8 < val_aa ? val_a8 : val_aa;
					val_ac = val_ac < val_ae ? val_ac : val_ae;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					val_a8 = val_a8 < val_ac ? val_a8 : val_ac;
					val_a0 = val_a0 < val_a8 ? val_a0 : val_a8;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<signed int, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = a + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			const signed int *ptr_a4 = ptr_a3 + rsa;
			const signed int *ptr_a5 = ptr_a4 + rsa;
			const signed int *ptr_a6 = ptr_a5 + rsa;
			const signed int *ptr_a7 = ptr_a6 + rsa;
			signed int val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7;
			signed int val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epi32(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epi32(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epi32(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epi32(ymm_a6, ymm_a7);
					ymm_a0 = _mm256_min_epi32(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epi32(ymm_a4, ymm_a6);
					ymm_a0 = _mm256_min_epi32(ymm_a0, ymm_a4);
					ymm_b0 = _mm256_min_epi32(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<unsigned int, cpu_avx2>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = a + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const unsigned int *ptr_a4 = ptr_a3 + rsa;
			const unsigned int *ptr_a5 = ptr_a4 + rsa;
			const unsigned int *ptr_a6 = ptr_a5 + rsa;
			const unsigned int *ptr_a7 = ptr_a6 + rsa;
			unsigned int val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7;
			unsigned int val_b0;
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
					// return minimum values
					ymm_a0 = _mm256_min_epu32(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_epu32(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_epu32(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_epu32(ymm_a6, ymm_a7);
					ymm_a0 = _mm256_min_epu32(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_epu32(ymm_a4, ymm_a6);
					ymm_a0 = _mm256_min_epu32(ymm_a0, ymm_a4);
					ymm_b0 = _mm256_min_epu32(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<float, cpu_avx>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			float val_a0, val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7;
			float val_b0;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					ymm_b0 = _mm256_loadu_ps(b + j);
					// return minimum values
					ymm_a0 = _mm256_min_ps(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_ps(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_min_ps(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_min_ps(ymm_a6, ymm_a7);
					ymm_a0 = _mm256_min_ps(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_min_ps(ymm_a4, ymm_a6);
					ymm_a0 = _mm256_min_ps(ymm_a0, ymm_a4);
					ymm_b0 = _mm256_min_ps(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_ps(b + j, ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_a4 = ptr_a4[j];
					val_a5 = ptr_a5[j];
					val_a6 = ptr_a6[j];
					val_a7 = ptr_a7[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a4 = val_a4 < val_a5 ? val_a4 : val_a5;
					val_a6 = val_a6 < val_a7 ? val_a6 : val_a7;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					val_a4 = val_a4 < val_a6 ? val_a4 : val_a6;
					val_a0 = val_a0 < val_a4 ? val_a0 : val_a4;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

	template<>
	struct block_mint<double, cpu_avx>
	{
		// b[j] = min(b[j], a[i][j])
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			double val_a0, val_a1, val_a2, val_a3;
			double val_b0;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0;

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(ptr_a0 + j);
					ymm_a1 = _mm256_loadu_pd(ptr_a1 + j);
					ymm_a2 = _mm256_loadu_pd(ptr_a2 + j);
					ymm_a3 = _mm256_loadu_pd(ptr_a3 + j);
					ymm_b0 = _mm256_loadu_pd(b + j);
					// return minimum values
					ymm_a0 = _mm256_min_pd(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_min_pd(ymm_a2, ymm_a3);
					ymm_a0 = _mm256_min_pd(ymm_a0, ymm_a2);
					ymm_b0 = _mm256_min_pd(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_pd(b + j, ymm_b0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					val_a0 = ptr_a0[j];
					val_a1 = ptr_a1[j];
					val_a2 = ptr_a2[j];
					val_a3 = ptr_a3[j];
					val_b0 = b[j];
					// return minimum values
					val_a0 = val_a0 < val_a1 ? val_a0 : val_a1;
					val_a2 = val_a2 < val_a3 ? val_a2 : val_a3;
					val_a0 = val_a0 < val_a2 ? val_a0 : val_a2;
					b[j] = val_b0 < val_a0 ? val_b0 : val_a0;
				}
			}
		}
	};

} // namespace core

#endif
