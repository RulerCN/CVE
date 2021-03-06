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

#ifndef __CORE_CPU_BLOCK_MAX_H__
#define __CORE_CPU_BLOCK_MAX_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_max

	template<class T, cpu_inst_type inst>
	struct block_max
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t /*aligned_n*/, size_t n, const T *a, size_t rsa, T *b) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = a + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			T val_b0 = b[0];
			T val_b1 = b[1];
			T val_b2 = b[2];
			T val_b3 = b[3];

			for (size_t j = 0; j < n; ++j)
			{
				val_b0 = val_b0 > ptr_a0[j] ? val_b0 : ptr_a0[j];
				val_b1 = val_b1 > ptr_a1[j] ? val_b1 : ptr_a1[j];
				val_b2 = val_b2 > ptr_a2[j] ? val_b2 : ptr_a2[j];
				val_b3 = val_b3 > ptr_a3[j] ? val_b3 : ptr_a3[j];
			}
			b[0] = val_b0;
			b[1] = val_b1;
			b[2] = val_b2;
			b[3] = val_b3;
		}
	};

	template<>
	struct block_max<signed char, cpu_sse41>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7, xmm_t8, xmm_t9, xmm_ta, xmm_tb, xmm_tc, xmm_td, xmm_te, xmm_tf;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				xmm_t4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4));
				xmm_t5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5));
				xmm_t6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6));
				xmm_t7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7));
				xmm_t8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8));
				xmm_t9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9));
				xmm_ta = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa));
				xmm_tb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab));
				xmm_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac));
				xmm_td = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad));
				xmm_te = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae));
				xmm_tf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af));
				for (size_t j = 16; j < aligned_n; j += 16)
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
					// return maximum values
					xmm_t0 = _mm_max_epi8(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epi8(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epi8(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epi8(xmm_t3, xmm_a3);
					xmm_t4 = _mm_max_epi8(xmm_t4, xmm_a4);
					xmm_t5 = _mm_max_epi8(xmm_t5, xmm_a5);
					xmm_t6 = _mm_max_epi8(xmm_t6, xmm_a6);
					xmm_t7 = _mm_max_epi8(xmm_t7, xmm_a7);
					xmm_t8 = _mm_max_epi8(xmm_t8, xmm_a8);
					xmm_t9 = _mm_max_epi8(xmm_t9, xmm_a9);
					xmm_ta = _mm_max_epi8(xmm_ta, xmm_aa);
					xmm_tb = _mm_max_epi8(xmm_tb, xmm_ab);
					xmm_tc = _mm_max_epi8(xmm_tc, xmm_ac);
					xmm_td = _mm_max_epi8(xmm_td, xmm_ad);
					xmm_te = _mm_max_epi8(xmm_te, xmm_ae);
					xmm_tf = _mm_max_epi8(xmm_tf, xmm_af);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi8(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi8(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpacklo_epi8(xmm_t4, xmm_t5);
				xmm_a3 = _mm_unpacklo_epi8(xmm_t6, xmm_t7);
				xmm_a4 = _mm_unpacklo_epi8(xmm_t8, xmm_t9);
				xmm_a5 = _mm_unpacklo_epi8(xmm_ta, xmm_tb);
				xmm_a6 = _mm_unpacklo_epi8(xmm_tc, xmm_td);
				xmm_a7 = _mm_unpacklo_epi8(xmm_te, xmm_tf);
				xmm_a8 = _mm_unpackhi_epi8(xmm_t0, xmm_t1);
				xmm_a9 = _mm_unpackhi_epi8(xmm_t2, xmm_t3);
				xmm_aa = _mm_unpackhi_epi8(xmm_t4, xmm_t5);
				xmm_ab = _mm_unpackhi_epi8(xmm_t6, xmm_t7);
				xmm_ac = _mm_unpackhi_epi8(xmm_t8, xmm_t9);
				xmm_ad = _mm_unpackhi_epi8(xmm_ta, xmm_tb);
				xmm_ae = _mm_unpackhi_epi8(xmm_tc, xmm_td);
				xmm_af = _mm_unpackhi_epi8(xmm_te, xmm_tf);
				xmm_a0 = _mm_max_epi8(xmm_a0, xmm_a8);
				xmm_a1 = _mm_max_epi8(xmm_a1, xmm_a9);
				xmm_a2 = _mm_max_epi8(xmm_a2, xmm_aa);
				xmm_a3 = _mm_max_epi8(xmm_a3, xmm_ab);
				xmm_a4 = _mm_max_epi8(xmm_a4, xmm_ac);
				xmm_a5 = _mm_max_epi8(xmm_a5, xmm_ad);
				xmm_a6 = _mm_max_epi8(xmm_a6, xmm_ae);
				xmm_a7 = _mm_max_epi8(xmm_a7, xmm_af);
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t3 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t4 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t5 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_t0 = _mm_max_epi8(xmm_t0, xmm_t4);
				xmm_t1 = _mm_max_epi8(xmm_t1, xmm_t5);
				xmm_t2 = _mm_max_epi8(xmm_t2, xmm_t6);
				xmm_t3 = _mm_max_epi8(xmm_t3, xmm_t7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a0 = _mm_max_epi8(xmm_a0, xmm_a2);
				xmm_a1 = _mm_max_epi8(xmm_a1, xmm_a3);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t0 = _mm_max_epi8(xmm_t0, xmm_t1);
				xmm_b0 = _mm_max_epi8(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi8(ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epi8(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<unsigned char, cpu_sse2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7, xmm_t8, xmm_t9, xmm_ta, xmm_tb, xmm_tc, xmm_td, xmm_te, xmm_tf;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				xmm_t4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4));
				xmm_t5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5));
				xmm_t6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6));
				xmm_t7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7));
				xmm_t8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a8));
				xmm_t9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a9));
				xmm_ta = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_aa));
				xmm_tb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ab));
				xmm_tc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ac));
				xmm_td = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ad));
				xmm_te = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_ae));
				xmm_tf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_af));
				for (size_t j = 16; j < aligned_n; j += 16)
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
					// return maximum values
					xmm_t0 = _mm_max_epu8(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epu8(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epu8(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epu8(xmm_t3, xmm_a3);
					xmm_t4 = _mm_max_epu8(xmm_t4, xmm_a4);
					xmm_t5 = _mm_max_epu8(xmm_t5, xmm_a5);
					xmm_t6 = _mm_max_epu8(xmm_t6, xmm_a6);
					xmm_t7 = _mm_max_epu8(xmm_t7, xmm_a7);
					xmm_t8 = _mm_max_epu8(xmm_t8, xmm_a8);
					xmm_t9 = _mm_max_epu8(xmm_t9, xmm_a9);
					xmm_ta = _mm_max_epu8(xmm_ta, xmm_aa);
					xmm_tb = _mm_max_epu8(xmm_tb, xmm_ab);
					xmm_tc = _mm_max_epu8(xmm_tc, xmm_ac);
					xmm_td = _mm_max_epu8(xmm_td, xmm_ad);
					xmm_te = _mm_max_epu8(xmm_te, xmm_ae);
					xmm_tf = _mm_max_epu8(xmm_tf, xmm_af);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi8(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi8(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpacklo_epi8(xmm_t4, xmm_t5);
				xmm_a3 = _mm_unpacklo_epi8(xmm_t6, xmm_t7);
				xmm_a4 = _mm_unpacklo_epi8(xmm_t8, xmm_t9);
				xmm_a5 = _mm_unpacklo_epi8(xmm_ta, xmm_tb);
				xmm_a6 = _mm_unpacklo_epi8(xmm_tc, xmm_td);
				xmm_a7 = _mm_unpacklo_epi8(xmm_te, xmm_tf);
				xmm_a8 = _mm_unpackhi_epi8(xmm_t0, xmm_t1);
				xmm_a9 = _mm_unpackhi_epi8(xmm_t2, xmm_t3);
				xmm_aa = _mm_unpackhi_epi8(xmm_t4, xmm_t5);
				xmm_ab = _mm_unpackhi_epi8(xmm_t6, xmm_t7);
				xmm_ac = _mm_unpackhi_epi8(xmm_t8, xmm_t9);
				xmm_ad = _mm_unpackhi_epi8(xmm_ta, xmm_tb);
				xmm_ae = _mm_unpackhi_epi8(xmm_tc, xmm_td);
				xmm_af = _mm_unpackhi_epi8(xmm_te, xmm_tf);
				xmm_a0 = _mm_max_epu8(xmm_a0, xmm_a8);
				xmm_a1 = _mm_max_epu8(xmm_a1, xmm_a9);
				xmm_a2 = _mm_max_epu8(xmm_a2, xmm_aa);
				xmm_a3 = _mm_max_epu8(xmm_a3, xmm_ab);
				xmm_a4 = _mm_max_epu8(xmm_a4, xmm_ac);
				xmm_a5 = _mm_max_epu8(xmm_a5, xmm_ad);
				xmm_a6 = _mm_max_epu8(xmm_a6, xmm_ae);
				xmm_a7 = _mm_max_epu8(xmm_a7, xmm_af);
				xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
				xmm_t3 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
				xmm_t4 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
				xmm_t5 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
				xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
				xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
				xmm_t0 = _mm_max_epu8(xmm_t0, xmm_t4);
				xmm_t1 = _mm_max_epu8(xmm_t1, xmm_t5);
				xmm_t2 = _mm_max_epu8(xmm_t2, xmm_t6);
				xmm_t3 = _mm_max_epu8(xmm_t3, xmm_t7);
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a0 = _mm_max_epu8(xmm_a0, xmm_a2);
				xmm_a1 = _mm_max_epu8(xmm_a1, xmm_a3);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t0 = _mm_max_epu8(xmm_t0, xmm_t1);
				xmm_b0 = _mm_max_epu8(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi8(ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epu8(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<signed short, cpu_sse2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				xmm_t4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4));
				xmm_t5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5));
				xmm_t6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6));
				xmm_t7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7));
				for (size_t j = 8; j < aligned_n; j += 8)
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
					// return maximum values
					xmm_t0 = _mm_max_epi16(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epi16(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epi16(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epi16(xmm_t3, xmm_a3);
					xmm_t4 = _mm_max_epi16(xmm_t4, xmm_a4);
					xmm_t5 = _mm_max_epi16(xmm_t5, xmm_a5);
					xmm_t6 = _mm_max_epi16(xmm_t6, xmm_a6);
					xmm_t7 = _mm_max_epi16(xmm_t7, xmm_a7);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi16(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi16(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpacklo_epi16(xmm_t4, xmm_t5);
				xmm_a3 = _mm_unpacklo_epi16(xmm_t6, xmm_t7);
				xmm_a4 = _mm_unpackhi_epi16(xmm_t0, xmm_t1);
				xmm_a5 = _mm_unpackhi_epi16(xmm_t2, xmm_t3);
				xmm_a6 = _mm_unpackhi_epi16(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpackhi_epi16(xmm_t6, xmm_t7);
				xmm_a0 = _mm_max_epi16(xmm_a0, xmm_a4);
				xmm_a1 = _mm_max_epi16(xmm_a1, xmm_a5);
				xmm_a2 = _mm_max_epi16(xmm_a2, xmm_a6);
				xmm_a3 = _mm_max_epi16(xmm_a3, xmm_a7);
				xmm_t0 = _mm_unpacklo_epi32(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi32(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi32(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi32(xmm_a2, xmm_a3);
				xmm_t0 = _mm_max_epi16(xmm_t0, xmm_t2);
				xmm_t1 = _mm_max_epi16(xmm_t1, xmm_t3);
				xmm_a0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi64(xmm_t0, xmm_t1);
				xmm_t0 = _mm_max_epi16(xmm_a0, xmm_a1);
				xmm_b0 = _mm_max_epi16(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi16(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epi16(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<unsigned short, cpu_sse41>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				xmm_t4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a4));
				xmm_t5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a5));
				xmm_t6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a6));
				xmm_t7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a7));
				for (size_t j = 8; j < aligned_n; j += 8)
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
					// return maximum values
					xmm_t0 = _mm_max_epu16(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epu16(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epu16(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epu16(xmm_t3, xmm_a3);
					xmm_t4 = _mm_max_epu16(xmm_t4, xmm_a4);
					xmm_t5 = _mm_max_epu16(xmm_t5, xmm_a5);
					xmm_t6 = _mm_max_epu16(xmm_t6, xmm_a6);
					xmm_t7 = _mm_max_epu16(xmm_t7, xmm_a7);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi16(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi16(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpacklo_epi16(xmm_t4, xmm_t5);
				xmm_a3 = _mm_unpacklo_epi16(xmm_t6, xmm_t7);
				xmm_a4 = _mm_unpackhi_epi16(xmm_t0, xmm_t1);
				xmm_a5 = _mm_unpackhi_epi16(xmm_t2, xmm_t3);
				xmm_a6 = _mm_unpackhi_epi16(xmm_t4, xmm_t5);
				xmm_a7 = _mm_unpackhi_epi16(xmm_t6, xmm_t7);
				xmm_a0 = _mm_max_epu16(xmm_a0, xmm_a4);
				xmm_a1 = _mm_max_epu16(xmm_a1, xmm_a5);
				xmm_a2 = _mm_max_epu16(xmm_a2, xmm_a6);
				xmm_a3 = _mm_max_epu16(xmm_a3, xmm_a7);
				xmm_t0 = _mm_unpacklo_epi32(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpacklo_epi32(xmm_a2, xmm_a3);
				xmm_t2 = _mm_unpackhi_epi32(xmm_a0, xmm_a1);
				xmm_t3 = _mm_unpackhi_epi32(xmm_a2, xmm_a3);
				xmm_t0 = _mm_max_epu16(xmm_t0, xmm_t2);
				xmm_t1 = _mm_max_epu16(xmm_t1, xmm_t3);
				xmm_a0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpackhi_epi64(xmm_t0, xmm_t1);
				xmm_t0 = _mm_max_epu16(xmm_a0, xmm_a1);
				xmm_b0 = _mm_max_epu16(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi16(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epu16(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<signed int, cpu_sse41>
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = a + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				for (size_t j = 4; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					// return maximum values
					xmm_t0 = _mm_max_epi32(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epi32(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epi32(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epi32(xmm_t3, xmm_a3);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a0 = _mm_max_epi32(xmm_a0, xmm_a2);
				xmm_a1 = _mm_max_epi32(xmm_a1, xmm_a3);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t0 = _mm_max_epi32(xmm_t0, xmm_t1);
				xmm_b0 = _mm_max_epi32(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epi32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<unsigned int, cpu_sse41>
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = a + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0));
				xmm_t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1));
				xmm_t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2));
				xmm_t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3));
				for (size_t j = 4; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					// return maximum values
					xmm_t0 = _mm_max_epu32(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_epu32(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_epu32(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_epu32(xmm_t3, xmm_a3);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
				xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
				xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
				xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
				xmm_a0 = _mm_max_epu32(xmm_a0, xmm_a2);
				xmm_a1 = _mm_max_epu32(xmm_a1, xmm_a3);
				xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
				xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
				xmm_t0 = _mm_max_epu32(xmm_t0, xmm_t1);
				xmm_b0 = _mm_max_epu32(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_epu32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_max<float, cpu_sse>
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128 xmm_b0 = _mm_loadu_ps(b);

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_ps(ptr_a0);
				xmm_t1 = _mm_loadu_ps(ptr_a1);
				xmm_t2 = _mm_loadu_ps(ptr_a2);
				xmm_t3 = _mm_loadu_ps(ptr_a3);
				for (size_t j = 4; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
					// return maximum values
					xmm_t0 = _mm_max_ps(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_ps(xmm_t1, xmm_a1);
					xmm_t2 = _mm_max_ps(xmm_t2, xmm_a2);
					xmm_t3 = _mm_max_ps(xmm_t3, xmm_a3);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(1, 0, 1, 0));
				xmm_a1 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(1, 0, 1, 0));
				xmm_a2 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(3, 2, 3, 2));
				xmm_a3 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(3, 2, 3, 2));
				xmm_a0 = _mm_max_ps(xmm_a0, xmm_a2);
				xmm_a1 = _mm_max_ps(xmm_a1, xmm_a3);
				xmm_t0 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(2, 0, 2, 0));
				xmm_t1 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(3, 1, 3, 1));
				xmm_t0 = _mm_max_ps(xmm_t0, xmm_t1);
				xmm_b0 = _mm_max_ps(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_ps(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_ps(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_ps(b, xmm_b0);
		}
	};

	template<>
	struct block_max<double, cpu_sse2>
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_b0 = _mm_loadu_pd(b);

			if (aligned_n > 0)
			{
				xmm_t0 = _mm_loadu_pd(ptr_a0);
				xmm_t1 = _mm_loadu_pd(ptr_a1);
				for (size_t j = 2; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(ptr_a0 + j);
					xmm_a1 = _mm_loadu_pd(ptr_a1 + j);
					// return maximum values
					xmm_t0 = _mm_max_pd(xmm_t0, xmm_a0);
					xmm_t1 = _mm_max_pd(xmm_t1, xmm_a1);
				}
				// return horizontal maximum values
				xmm_a0 = _mm_shuffle_pd(xmm_t0, xmm_t1, _MM_SHUFFLE(0, 0, 0, 0));
				xmm_a1 = _mm_shuffle_pd(xmm_t0, xmm_t1, _MM_SHUFFLE(0, 0, 3, 3));
				xmm_t0 = _mm_max_pd(xmm_a0, xmm_a1);
				xmm_b0 = _mm_max_pd(xmm_b0, xmm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_pd(ptr_a1[j], ptr_a0[j]);
					// return maximum values
					xmm_b0 = _mm_max_pd(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
		}
	};

	template<>
	struct block_max<signed char, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af,
				ymm_ag, ymm_ah, ymm_ai, ymm_aj, ymm_ak, ymm_al, ymm_am, ymm_an, ymm_ao, ymm_ap, ymm_aq, ymm_ar, ymm_as, ymm_at, ymm_au, ymm_av;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf,
				ymm_tg, ymm_th, ymm_ti, ymm_tj, ymm_tk, ymm_tl, ymm_tm, ymm_tn, ymm_to, ymm_tp, ymm_tq, ymm_tr, ymm_ts, ymm_tt, ymm_tu, ymm_tv;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				ymm_t8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8));
				ymm_t9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9));
				ymm_ta = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa));
				ymm_tb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab));
				ymm_tc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac));
				ymm_td = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad));
				ymm_te = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae));
				ymm_tf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af));
				ymm_tg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ag));
				ymm_th = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ah));
				ymm_ti = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ai));
				ymm_tj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aj));
				ymm_tk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ak));
				ymm_tl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_al));
				ymm_tm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_am));
				ymm_tn = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_an));
				ymm_to = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ao));
				ymm_tp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ap));
				ymm_tq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aq));
				ymm_tr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ar));
				ymm_ts = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_as));
				ymm_tt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_at));
				ymm_tu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_au));
				ymm_tv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_av));
				for (size_t j = 32; j < aligned_n; j += 32)
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
					// return maximum values
					ymm_t0 = _mm256_max_epi8(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epi8(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epi8(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epi8(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epi8(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epi8(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epi8(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epi8(ymm_t7, ymm_a7);
					ymm_t8 = _mm256_max_epi8(ymm_t8, ymm_a8);
					ymm_t9 = _mm256_max_epi8(ymm_t9, ymm_a9);
					ymm_ta = _mm256_max_epi8(ymm_ta, ymm_aa);
					ymm_tb = _mm256_max_epi8(ymm_tb, ymm_ab);
					ymm_tc = _mm256_max_epi8(ymm_tc, ymm_ac);
					ymm_td = _mm256_max_epi8(ymm_td, ymm_ad);
					ymm_te = _mm256_max_epi8(ymm_te, ymm_ae);
					ymm_tf = _mm256_max_epi8(ymm_tf, ymm_af);
					ymm_tg = _mm256_max_epi8(ymm_tg, ymm_ag);
					ymm_th = _mm256_max_epi8(ymm_th, ymm_ah);
					ymm_ti = _mm256_max_epi8(ymm_ti, ymm_ai);
					ymm_tj = _mm256_max_epi8(ymm_tj, ymm_aj);
					ymm_tk = _mm256_max_epi8(ymm_tk, ymm_ak);
					ymm_tl = _mm256_max_epi8(ymm_tl, ymm_al);
					ymm_tm = _mm256_max_epi8(ymm_tm, ymm_am);
					ymm_tn = _mm256_max_epi8(ymm_tn, ymm_an);
					ymm_to = _mm256_max_epi8(ymm_to, ymm_ao);
					ymm_tp = _mm256_max_epi8(ymm_tp, ymm_ap);
					ymm_tq = _mm256_max_epi8(ymm_tq, ymm_aq);
					ymm_tr = _mm256_max_epi8(ymm_tr, ymm_ar);
					ymm_ts = _mm256_max_epi8(ymm_ts, ymm_as);
					ymm_tt = _mm256_max_epi8(ymm_tt, ymm_at);
					ymm_tu = _mm256_max_epi8(ymm_tu, ymm_au);
					ymm_tv = _mm256_max_epi8(ymm_tv, ymm_av);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_tg, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_th, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_ti, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_tj, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t4, ymm_tk, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t5, ymm_tl, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t6, ymm_tm, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t7, ymm_tn, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a8 = _mm256_permute2f128_si256(ymm_t8, ymm_to, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a9 = _mm256_permute2f128_si256(ymm_t9, ymm_tp, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_aa = _mm256_permute2f128_si256(ymm_ta, ymm_tq, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ab = _mm256_permute2f128_si256(ymm_tb, ymm_tr, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ac = _mm256_permute2f128_si256(ymm_tc, ymm_ts, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ad = _mm256_permute2f128_si256(ymm_td, ymm_tt, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ae = _mm256_permute2f128_si256(ymm_te, ymm_tu, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_af = _mm256_permute2f128_si256(ymm_tf, ymm_tv, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ag = _mm256_permute2f128_si256(ymm_t0, ymm_tg, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ah = _mm256_permute2f128_si256(ymm_t1, ymm_th, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ai = _mm256_permute2f128_si256(ymm_t2, ymm_ti, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aj = _mm256_permute2f128_si256(ymm_t3, ymm_tj, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ak = _mm256_permute2f128_si256(ymm_t4, ymm_tk, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_al = _mm256_permute2f128_si256(ymm_t5, ymm_tl, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_am = _mm256_permute2f128_si256(ymm_t6, ymm_tm, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_an = _mm256_permute2f128_si256(ymm_t7, ymm_tn, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ao = _mm256_permute2f128_si256(ymm_t8, ymm_to, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ap = _mm256_permute2f128_si256(ymm_t9, ymm_tp, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aq = _mm256_permute2f128_si256(ymm_ta, ymm_tq, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ar = _mm256_permute2f128_si256(ymm_tb, ymm_tr, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_as = _mm256_permute2f128_si256(ymm_tc, ymm_ts, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_at = _mm256_permute2f128_si256(ymm_td, ymm_tt, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_au = _mm256_permute2f128_si256(ymm_te, ymm_tu, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_av = _mm256_permute2f128_si256(ymm_tf, ymm_tv, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epi8(ymm_a0, ymm_ag);
				ymm_a1 = _mm256_max_epi8(ymm_a1, ymm_ah);
				ymm_a2 = _mm256_max_epi8(ymm_a2, ymm_ai);
				ymm_a3 = _mm256_max_epi8(ymm_a3, ymm_aj);
				ymm_a4 = _mm256_max_epi8(ymm_a4, ymm_ak);
				ymm_a5 = _mm256_max_epi8(ymm_a5, ymm_al);
				ymm_a6 = _mm256_max_epi8(ymm_a6, ymm_am);
				ymm_a7 = _mm256_max_epi8(ymm_a7, ymm_an);
				ymm_a8 = _mm256_max_epi8(ymm_a8, ymm_ao);
				ymm_a9 = _mm256_max_epi8(ymm_a9, ymm_ap);
				ymm_aa = _mm256_max_epi8(ymm_aa, ymm_aq);
				ymm_ab = _mm256_max_epi8(ymm_ab, ymm_ar);
				ymm_ac = _mm256_max_epi8(ymm_ac, ymm_as);
				ymm_ad = _mm256_max_epi8(ymm_ad, ymm_at);
				ymm_ae = _mm256_max_epi8(ymm_ae, ymm_au);
				ymm_af = _mm256_max_epi8(ymm_af, ymm_av);
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
				ymm_t0 = _mm256_max_epi8(ymm_t0, ymm_t8);
				ymm_t1 = _mm256_max_epi8(ymm_t1, ymm_t9);
				ymm_t2 = _mm256_max_epi8(ymm_t2, ymm_ta);
				ymm_t3 = _mm256_max_epi8(ymm_t3, ymm_tb);
				ymm_t4 = _mm256_max_epi8(ymm_t4, ymm_tc);
				ymm_t5 = _mm256_max_epi8(ymm_t5, ymm_td);
				ymm_t6 = _mm256_max_epi8(ymm_t6, ymm_te);
				ymm_t7 = _mm256_max_epi8(ymm_t7, ymm_tf);
				ymm_a0 = _mm256_unpacklo_epi16(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi16(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpacklo_epi16(ymm_t4, ymm_t5);
				ymm_a3 = _mm256_unpacklo_epi16(ymm_t6, ymm_t7);
				ymm_a4 = _mm256_unpackhi_epi16(ymm_t0, ymm_t1);
				ymm_a5 = _mm256_unpackhi_epi16(ymm_t2, ymm_t3);
				ymm_a6 = _mm256_unpackhi_epi16(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi16(ymm_t6, ymm_t7);
				ymm_a0 = _mm256_max_epi8(ymm_a0, ymm_a4);
				ymm_a1 = _mm256_max_epi8(ymm_a1, ymm_a5);
				ymm_a2 = _mm256_max_epi8(ymm_a2, ymm_a6);
				ymm_a3 = _mm256_max_epi8(ymm_a3, ymm_a7);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t0 = _mm256_max_epi8(ymm_t0, ymm_t2);
				ymm_t1 = _mm256_max_epi8(ymm_t1, ymm_t3);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_t0 = _mm256_max_epi8(ymm_a0, ymm_a1);
				ymm_b0 = _mm256_max_epi8(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi8(
						ptr_av[j], ptr_au[j], ptr_at[j], ptr_as[j], ptr_ar[j], ptr_aq[j], ptr_ap[j], ptr_ao[j], ptr_an[j], ptr_am[j], ptr_al[j], ptr_ak[j], ptr_aj[j], ptr_ai[j], ptr_ah[j], ptr_ag[j],
						ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epi8(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<unsigned char, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af,
				ymm_ag, ymm_ah, ymm_ai, ymm_aj, ymm_ak, ymm_al, ymm_am, ymm_an, ymm_ao, ymm_ap, ymm_aq, ymm_ar, ymm_as, ymm_at, ymm_au, ymm_av;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf,
				ymm_tg, ymm_th, ymm_ti, ymm_tj, ymm_tk, ymm_tl, ymm_tm, ymm_tn, ymm_to, ymm_tp, ymm_tq, ymm_tr, ymm_ts, ymm_tt, ymm_tu, ymm_tv;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				ymm_t8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8));
				ymm_t9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9));
				ymm_ta = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa));
				ymm_tb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab));
				ymm_tc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac));
				ymm_td = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad));
				ymm_te = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae));
				ymm_tf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af));
				ymm_tg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ag));
				ymm_th = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ah));
				ymm_ti = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ai));
				ymm_tj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aj));
				ymm_tk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ak));
				ymm_tl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_al));
				ymm_tm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_am));
				ymm_tn = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_an));
				ymm_to = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ao));
				ymm_tp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ap));
				ymm_tq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aq));
				ymm_tr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ar));
				ymm_ts = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_as));
				ymm_tt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_at));
				ymm_tu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_au));
				ymm_tv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_av));
				ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
				for (size_t j = 32; j < aligned_n; j += 32)
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
					// return maximum values
					ymm_t0 = _mm256_max_epu8(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epu8(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epu8(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epu8(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epu8(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epu8(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epu8(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epu8(ymm_t7, ymm_a7);
					ymm_t8 = _mm256_max_epu8(ymm_t8, ymm_a8);
					ymm_t9 = _mm256_max_epu8(ymm_t9, ymm_a9);
					ymm_ta = _mm256_max_epu8(ymm_ta, ymm_aa);
					ymm_tb = _mm256_max_epu8(ymm_tb, ymm_ab);
					ymm_tc = _mm256_max_epu8(ymm_tc, ymm_ac);
					ymm_td = _mm256_max_epu8(ymm_td, ymm_ad);
					ymm_te = _mm256_max_epu8(ymm_te, ymm_ae);
					ymm_tf = _mm256_max_epu8(ymm_tf, ymm_af);
					ymm_tg = _mm256_max_epu8(ymm_tg, ymm_ag);
					ymm_th = _mm256_max_epu8(ymm_th, ymm_ah);
					ymm_ti = _mm256_max_epu8(ymm_ti, ymm_ai);
					ymm_tj = _mm256_max_epu8(ymm_tj, ymm_aj);
					ymm_tk = _mm256_max_epu8(ymm_tk, ymm_ak);
					ymm_tl = _mm256_max_epu8(ymm_tl, ymm_al);
					ymm_tm = _mm256_max_epu8(ymm_tm, ymm_am);
					ymm_tn = _mm256_max_epu8(ymm_tn, ymm_an);
					ymm_to = _mm256_max_epu8(ymm_to, ymm_ao);
					ymm_tp = _mm256_max_epu8(ymm_tp, ymm_ap);
					ymm_tq = _mm256_max_epu8(ymm_tq, ymm_aq);
					ymm_tr = _mm256_max_epu8(ymm_tr, ymm_ar);
					ymm_ts = _mm256_max_epu8(ymm_ts, ymm_as);
					ymm_tt = _mm256_max_epu8(ymm_tt, ymm_at);
					ymm_tu = _mm256_max_epu8(ymm_tu, ymm_au);
					ymm_tv = _mm256_max_epu8(ymm_tv, ymm_av);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_tg, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_th, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_ti, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_tj, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t4, ymm_tk, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t5, ymm_tl, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t6, ymm_tm, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t7, ymm_tn, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a8 = _mm256_permute2f128_si256(ymm_t8, ymm_to, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a9 = _mm256_permute2f128_si256(ymm_t9, ymm_tp, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_aa = _mm256_permute2f128_si256(ymm_ta, ymm_tq, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ab = _mm256_permute2f128_si256(ymm_tb, ymm_tr, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ac = _mm256_permute2f128_si256(ymm_tc, ymm_ts, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ad = _mm256_permute2f128_si256(ymm_td, ymm_tt, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ae = _mm256_permute2f128_si256(ymm_te, ymm_tu, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_af = _mm256_permute2f128_si256(ymm_tf, ymm_tv, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_ag = _mm256_permute2f128_si256(ymm_t0, ymm_tg, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ah = _mm256_permute2f128_si256(ymm_t1, ymm_th, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ai = _mm256_permute2f128_si256(ymm_t2, ymm_ti, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aj = _mm256_permute2f128_si256(ymm_t3, ymm_tj, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ak = _mm256_permute2f128_si256(ymm_t4, ymm_tk, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_al = _mm256_permute2f128_si256(ymm_t5, ymm_tl, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_am = _mm256_permute2f128_si256(ymm_t6, ymm_tm, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_an = _mm256_permute2f128_si256(ymm_t7, ymm_tn, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ao = _mm256_permute2f128_si256(ymm_t8, ymm_to, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ap = _mm256_permute2f128_si256(ymm_t9, ymm_tp, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aq = _mm256_permute2f128_si256(ymm_ta, ymm_tq, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ar = _mm256_permute2f128_si256(ymm_tb, ymm_tr, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_as = _mm256_permute2f128_si256(ymm_tc, ymm_ts, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_at = _mm256_permute2f128_si256(ymm_td, ymm_tt, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_au = _mm256_permute2f128_si256(ymm_te, ymm_tu, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_av = _mm256_permute2f128_si256(ymm_tf, ymm_tv, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epu8(ymm_a0, ymm_ag);
				ymm_a1 = _mm256_max_epu8(ymm_a1, ymm_ah);
				ymm_a2 = _mm256_max_epu8(ymm_a2, ymm_ai);
				ymm_a3 = _mm256_max_epu8(ymm_a3, ymm_aj);
				ymm_a4 = _mm256_max_epu8(ymm_a4, ymm_ak);
				ymm_a5 = _mm256_max_epu8(ymm_a5, ymm_al);
				ymm_a6 = _mm256_max_epu8(ymm_a6, ymm_am);
				ymm_a7 = _mm256_max_epu8(ymm_a7, ymm_an);
				ymm_a8 = _mm256_max_epu8(ymm_a8, ymm_ao);
				ymm_a9 = _mm256_max_epu8(ymm_a9, ymm_ap);
				ymm_aa = _mm256_max_epu8(ymm_aa, ymm_aq);
				ymm_ab = _mm256_max_epu8(ymm_ab, ymm_ar);
				ymm_ac = _mm256_max_epu8(ymm_ac, ymm_as);
				ymm_ad = _mm256_max_epu8(ymm_ad, ymm_at);
				ymm_ae = _mm256_max_epu8(ymm_ae, ymm_au);
				ymm_af = _mm256_max_epu8(ymm_af, ymm_av);
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
				ymm_t0 = _mm256_max_epu8(ymm_t0, ymm_t8);
				ymm_t1 = _mm256_max_epu8(ymm_t1, ymm_t9);
				ymm_t2 = _mm256_max_epu8(ymm_t2, ymm_ta);
				ymm_t3 = _mm256_max_epu8(ymm_t3, ymm_tb);
				ymm_t4 = _mm256_max_epu8(ymm_t4, ymm_tc);
				ymm_t5 = _mm256_max_epu8(ymm_t5, ymm_td);
				ymm_t6 = _mm256_max_epu8(ymm_t6, ymm_te);
				ymm_t7 = _mm256_max_epu8(ymm_t7, ymm_tf);
				ymm_a0 = _mm256_unpacklo_epi16(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi16(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpacklo_epi16(ymm_t4, ymm_t5);
				ymm_a3 = _mm256_unpacklo_epi16(ymm_t6, ymm_t7);
				ymm_a4 = _mm256_unpackhi_epi16(ymm_t0, ymm_t1);
				ymm_a5 = _mm256_unpackhi_epi16(ymm_t2, ymm_t3);
				ymm_a6 = _mm256_unpackhi_epi16(ymm_t4, ymm_t5);
				ymm_a7 = _mm256_unpackhi_epi16(ymm_t6, ymm_t7);
				ymm_a0 = _mm256_max_epu8(ymm_a0, ymm_a4);
				ymm_a1 = _mm256_max_epu8(ymm_a1, ymm_a5);
				ymm_a2 = _mm256_max_epu8(ymm_a2, ymm_a6);
				ymm_a3 = _mm256_max_epu8(ymm_a3, ymm_a7);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t0 = _mm256_max_epu8(ymm_t0, ymm_t2);
				ymm_t1 = _mm256_max_epu8(ymm_t1, ymm_t3);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_t0 = _mm256_max_epu8(ymm_a0, ymm_a1);
				ymm_b0 = _mm256_max_epu8(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi8(
						ptr_av[j], ptr_au[j], ptr_at[j], ptr_as[j], ptr_ar[j], ptr_aq[j], ptr_ap[j], ptr_ao[j], ptr_an[j], ptr_am[j], ptr_al[j], ptr_ak[j], ptr_aj[j], ptr_ai[j], ptr_ah[j], ptr_ag[j],
						ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epu8(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<signed short, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				ymm_t8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8));
				ymm_t9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9));
				ymm_ta = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa));
				ymm_tb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab));
				ymm_tc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac));
				ymm_td = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad));
				ymm_te = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae));
				ymm_tf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af));
				for (size_t j = 16; j < aligned_n; j += 16)
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
					// return maximum values
					ymm_t0 = _mm256_max_epi16(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epi16(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epi16(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epi16(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epi16(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epi16(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epi16(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epi16(ymm_t7, ymm_a7);
					ymm_t8 = _mm256_max_epi16(ymm_t8, ymm_a8);
					ymm_t9 = _mm256_max_epi16(ymm_t9, ymm_a9);
					ymm_ta = _mm256_max_epi16(ymm_ta, ymm_aa);
					ymm_tb = _mm256_max_epi16(ymm_tb, ymm_ab);
					ymm_tc = _mm256_max_epi16(ymm_tc, ymm_ac);
					ymm_td = _mm256_max_epi16(ymm_td, ymm_ad);
					ymm_te = _mm256_max_epi16(ymm_te, ymm_ae);
					ymm_tf = _mm256_max_epi16(ymm_tf, ymm_af);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_t8, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_t9, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_ta, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_tb, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t4, ymm_tc, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t5, ymm_td, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t6, ymm_te, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t7, ymm_tf, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a8 = _mm256_permute2f128_si256(ymm_t0, ymm_t8, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a9 = _mm256_permute2f128_si256(ymm_t1, ymm_t9, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aa = _mm256_permute2f128_si256(ymm_t2, ymm_ta, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ab = _mm256_permute2f128_si256(ymm_t3, ymm_tb, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ac = _mm256_permute2f128_si256(ymm_t4, ymm_tc, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ad = _mm256_permute2f128_si256(ymm_t5, ymm_td, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ae = _mm256_permute2f128_si256(ymm_t6, ymm_te, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_af = _mm256_permute2f128_si256(ymm_t7, ymm_tf, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epi16(ymm_a0, ymm_a8);
				ymm_a1 = _mm256_max_epi16(ymm_a1, ymm_a9);
				ymm_a2 = _mm256_max_epi16(ymm_a2, ymm_aa);
				ymm_a3 = _mm256_max_epi16(ymm_a3, ymm_ab);
				ymm_a4 = _mm256_max_epi16(ymm_a4, ymm_ac);
				ymm_a5 = _mm256_max_epi16(ymm_a5, ymm_ad);
				ymm_a6 = _mm256_max_epi16(ymm_a6, ymm_ae);
				ymm_a7 = _mm256_max_epi16(ymm_a7, ymm_af);
				ymm_t0 = _mm256_unpacklo_epi16(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi16(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi16(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi16(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi16(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi16(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi16(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi16(ymm_a6, ymm_a7);
				ymm_t0 = _mm256_max_epi16(ymm_t0, ymm_t4);
				ymm_t1 = _mm256_max_epi16(ymm_t1, ymm_t5);
				ymm_t2 = _mm256_max_epi16(ymm_t2, ymm_t6);
				ymm_t3 = _mm256_max_epi16(ymm_t3, ymm_t7);
				ymm_a0 = _mm256_unpacklo_epi32(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi32(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi32(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi32(ymm_t2, ymm_t3);
				ymm_a0 = _mm256_max_epi16(ymm_a0, ymm_a2);
				ymm_a1 = _mm256_max_epi16(ymm_a1, ymm_a3);
				ymm_t0 = _mm256_unpacklo_epi64(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpackhi_epi64(ymm_a0, ymm_a1);
				ymm_t0 = _mm256_max_epi16(ymm_t0, ymm_t1);
				ymm_b0 = _mm256_max_epi16(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi16(ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epi16(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<unsigned short, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				ymm_t8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a8));
				ymm_t9 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a9));
				ymm_ta = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_aa));
				ymm_tb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ab));
				ymm_tc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ac));
				ymm_td = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ad));
				ymm_te = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_ae));
				ymm_tf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_af));
				for (size_t j = 16; j < aligned_n; j += 16)
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
					// return maximum values
					ymm_t0 = _mm256_max_epu16(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epu16(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epu16(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epu16(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epu16(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epu16(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epu16(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epu16(ymm_t7, ymm_a7);
					ymm_t8 = _mm256_max_epu16(ymm_t8, ymm_a8);
					ymm_t9 = _mm256_max_epu16(ymm_t9, ymm_a9);
					ymm_ta = _mm256_max_epu16(ymm_ta, ymm_aa);
					ymm_tb = _mm256_max_epu16(ymm_tb, ymm_ab);
					ymm_tc = _mm256_max_epu16(ymm_tc, ymm_ac);
					ymm_td = _mm256_max_epu16(ymm_td, ymm_ad);
					ymm_te = _mm256_max_epu16(ymm_te, ymm_ae);
					ymm_tf = _mm256_max_epu16(ymm_tf, ymm_af);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_t8, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_t9, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_ta, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_tb, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t4, ymm_tc, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t5, ymm_td, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t6, ymm_te, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t7, ymm_tf, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a8 = _mm256_permute2f128_si256(ymm_t0, ymm_t8, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a9 = _mm256_permute2f128_si256(ymm_t1, ymm_t9, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_aa = _mm256_permute2f128_si256(ymm_t2, ymm_ta, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ab = _mm256_permute2f128_si256(ymm_t3, ymm_tb, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ac = _mm256_permute2f128_si256(ymm_t4, ymm_tc, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ad = _mm256_permute2f128_si256(ymm_t5, ymm_td, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_ae = _mm256_permute2f128_si256(ymm_t6, ymm_te, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_af = _mm256_permute2f128_si256(ymm_t7, ymm_tf, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epu16(ymm_a0, ymm_a8);
				ymm_a1 = _mm256_max_epu16(ymm_a1, ymm_a9);
				ymm_a2 = _mm256_max_epu16(ymm_a2, ymm_aa);
				ymm_a3 = _mm256_max_epu16(ymm_a3, ymm_ab);
				ymm_a4 = _mm256_max_epu16(ymm_a4, ymm_ac);
				ymm_a5 = _mm256_max_epu16(ymm_a5, ymm_ad);
				ymm_a6 = _mm256_max_epu16(ymm_a6, ymm_ae);
				ymm_a7 = _mm256_max_epu16(ymm_a7, ymm_af);
				ymm_t0 = _mm256_unpacklo_epi16(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi16(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpacklo_epi16(ymm_a4, ymm_a5);
				ymm_t3 = _mm256_unpacklo_epi16(ymm_a6, ymm_a7);
				ymm_t4 = _mm256_unpackhi_epi16(ymm_a0, ymm_a1);
				ymm_t5 = _mm256_unpackhi_epi16(ymm_a2, ymm_a3);
				ymm_t6 = _mm256_unpackhi_epi16(ymm_a4, ymm_a5);
				ymm_t7 = _mm256_unpackhi_epi16(ymm_a6, ymm_a7);
				ymm_t0 = _mm256_max_epu16(ymm_t0, ymm_t4);
				ymm_t1 = _mm256_max_epu16(ymm_t1, ymm_t5);
				ymm_t2 = _mm256_max_epu16(ymm_t2, ymm_t6);
				ymm_t3 = _mm256_max_epu16(ymm_t3, ymm_t7);
				ymm_a0 = _mm256_unpacklo_epi32(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpacklo_epi32(ymm_t2, ymm_t3);
				ymm_a2 = _mm256_unpackhi_epi32(ymm_t0, ymm_t1);
				ymm_a3 = _mm256_unpackhi_epi32(ymm_t2, ymm_t3);
				ymm_a0 = _mm256_max_epu16(ymm_a0, ymm_a2);
				ymm_a1 = _mm256_max_epu16(ymm_a1, ymm_a3);
				ymm_t0 = _mm256_unpacklo_epi64(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpackhi_epi64(ymm_a0, ymm_a1);
				ymm_t0 = _mm256_max_epu16(ymm_t0, ymm_t1);
				ymm_b0 = _mm256_max_epu16(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi16(ptr_af[j], ptr_ae[j], ptr_ad[j], ptr_ac[j], ptr_ab[j], ptr_aa[j], ptr_a9[j], ptr_a8[j], ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epu16(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<signed int, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				for (size_t j = 8; j < aligned_n; j += 8)
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
					// return maximum values
					ymm_t0 = _mm256_max_epi32(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epi32(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epi32(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epi32(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epi32(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epi32(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epi32(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epi32(ymm_t7, ymm_a7);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epi32(ymm_a0, ymm_a4);
				ymm_a1 = _mm256_max_epi32(ymm_a1, ymm_a5);
				ymm_a2 = _mm256_max_epi32(ymm_a2, ymm_a6);
				ymm_a3 = _mm256_max_epi32(ymm_a3, ymm_a7);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t0 = _mm256_max_epi32(ymm_t0, ymm_t2);
				ymm_t1 = _mm256_max_epi32(ymm_t1, ymm_t3);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_t0 = _mm256_max_epi32(ymm_a0, ymm_a1);
				ymm_b0 = _mm256_max_epi32(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epi32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<unsigned int, cpu_avx2>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a0));
				ymm_t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a1));
				ymm_t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a2));
				ymm_t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a3));
				ymm_t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a4));
				ymm_t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a5));
				ymm_t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a6));
				ymm_t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr_a7));
				for (size_t j = 8; j < aligned_n; j += 8)
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
					// return maximum values
					ymm_t0 = _mm256_max_epu32(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_epu32(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_epu32(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_epu32(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_epu32(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_epu32(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_epu32(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_epu32(ymm_t7, ymm_a7);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_si256(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_si256(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_si256(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a5 = _mm256_permute2f128_si256(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a6 = _mm256_permute2f128_si256(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a7 = _mm256_permute2f128_si256(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_epu32(ymm_a0, ymm_a4);
				ymm_a1 = _mm256_max_epu32(ymm_a1, ymm_a5);
				ymm_a2 = _mm256_max_epu32(ymm_a2, ymm_a6);
				ymm_a3 = _mm256_max_epu32(ymm_a3, ymm_a7);
				ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
				ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
				ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
				ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
				ymm_t0 = _mm256_max_epu32(ymm_t0, ymm_t2);
				ymm_t1 = _mm256_max_epu32(ymm_t1, ymm_t3);
				ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
				ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
				ymm_t0 = _mm256_max_epu32(ymm_a0, ymm_a1);
				ymm_b0 = _mm256_max_epu32(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_epu32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_max<float, cpu_avx>
	{
		// b[i] = max(b[i], a[i][j])
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
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256 ymm_b0 = _mm256_loadu_ps(b);

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_ps(ptr_a0);
				ymm_t1 = _mm256_loadu_ps(ptr_a1);
				ymm_t2 = _mm256_loadu_ps(ptr_a2);
				ymm_t3 = _mm256_loadu_ps(ptr_a3);
				ymm_t4 = _mm256_loadu_ps(ptr_a4);
				ymm_t5 = _mm256_loadu_ps(ptr_a5);
				ymm_t6 = _mm256_loadu_ps(ptr_a6);
				ymm_t7 = _mm256_loadu_ps(ptr_a7);
				for (size_t j = 8; j < aligned_n; j += 8)
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
					// return maximum values
					ymm_t0 = _mm256_max_ps(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_ps(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_ps(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_ps(ymm_t3, ymm_a3);
					ymm_t4 = _mm256_max_ps(ymm_t4, ymm_a4);
					ymm_t5 = _mm256_max_ps(ymm_t5, ymm_a5);
					ymm_t6 = _mm256_max_ps(ymm_t6, ymm_a6);
					ymm_t7 = _mm256_max_ps(ymm_t7, ymm_a7);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_ps(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_ps(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a3 = _mm256_permute2f128_ps(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a4 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a5 = _mm256_permute2f128_ps(ymm_t1, ymm_t5, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a6 = _mm256_permute2f128_ps(ymm_t2, ymm_t6, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a7 = _mm256_permute2f128_ps(ymm_t3, ymm_t7, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_ps(ymm_a0, ymm_a4);
				ymm_a1 = _mm256_max_ps(ymm_a1, ymm_a5);
				ymm_a2 = _mm256_max_ps(ymm_a2, ymm_a6);
				ymm_a3 = _mm256_max_ps(ymm_a3, ymm_a7);
				ymm_t0 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t1 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(1, 0, 1, 0));
				ymm_t2 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_t3 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(3, 2, 3, 2));
				ymm_t0 = _mm256_max_ps(ymm_t0, ymm_t2);
				ymm_t1 = _mm256_max_ps(ymm_t1, ymm_t3);
				ymm_a0 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(2, 0, 2, 0));
				ymm_a1 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(3, 1, 3, 1));
				ymm_t0 = _mm256_max_ps(ymm_a0, ymm_a1);
				ymm_b0 = _mm256_max_ps(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_ps(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_ps(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_ps(b, ymm_b0);
		}
	};

	template<>
	struct block_max<double, cpu_avx>
	{
		// b[i] = max(b[i], a[i][j])
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_b0 = _mm256_loadu_pd(b);

			if (aligned_n > 0)
			{
				ymm_t0 = _mm256_loadu_pd(ptr_a0);
				ymm_t1 = _mm256_loadu_pd(ptr_a1);
				ymm_t2 = _mm256_loadu_pd(ptr_a2);
				ymm_t3 = _mm256_loadu_pd(ptr_a3);
				for (size_t j = 4; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(ptr_a0 + j);
					ymm_a1 = _mm256_loadu_pd(ptr_a1 + j);
					ymm_a2 = _mm256_loadu_pd(ptr_a2 + j);
					ymm_a3 = _mm256_loadu_pd(ptr_a3 + j);
					// return maximum values
					ymm_t0 = _mm256_max_pd(ymm_t0, ymm_a0);
					ymm_t1 = _mm256_max_pd(ymm_t1, ymm_a1);
					ymm_t2 = _mm256_max_pd(ymm_t2, ymm_a2);
					ymm_t3 = _mm256_max_pd(ymm_t3, ymm_a3);
				}
				// return horizontal maximum values
				ymm_a0 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a1 = _mm256_permute2f128_pd(ymm_t1, ymm_t3, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_a2 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a3 = _mm256_permute2f128_pd(ymm_t1, ymm_t3, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_a0 = _mm256_max_pd(ymm_a0, ymm_a2);
				ymm_a1 = _mm256_max_pd(ymm_a1, ymm_a3);
				ymm_t0 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 0, 0));
				ymm_t1 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 3, 3));
				ymm_t0 = _mm256_max_pd(ymm_t0, ymm_t1);
				ymm_b0 = _mm256_max_pd(ymm_b0, ymm_t0);
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_pd(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return maximum values
					ymm_b0 = _mm256_max_pd(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
		}
	};

} // namespace core

#endif
