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

#ifndef __CORE_CPU_KERNEL_BLOCK_SUM_DOUBLE_H__
#define __CORE_CPU_KERNEL_BLOCK_SUM_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_sum_double

	template<class T, cpu_inst_type inst>
	struct block_sum_double
	{
		void operator()(size_t, size_t n, const T *a, size_t rsa, double *b) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = a + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			double val_b0 = b[0];
			double val_b1 = b[1];
			double val_b2 = b[2];
			double val_b3 = b[3];

			for (size_t j = 0; j < n; ++j)
			{
				val_b0 += static_cast<double>(ptr_a0[j]);
				val_b1 += static_cast<double>(ptr_a1[j]);
				val_b2 += static_cast<double>(ptr_a2[j]);
				val_b3 += static_cast<double>(ptr_a3[j]);
			}
			b[0] = val_b0;
			b[1] = val_b1;
			b[2] = val_b2;
			b[3] = val_b3;
		}
	};

	template<>
	struct block_sum_double<signed char, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, double *b) const
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
			__m128d xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);
			__m128d xmm_b2 = _mm_loadu_pd(b + 4);
			__m128d xmm_b3 = _mm_loadu_pd(b + 6);

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
					// return the horizontal summation
					xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a8);
					xmm_a1 = _mm_add_epi16(xmm_a1, xmm_a9);
					xmm_a2 = _mm_add_epi16(xmm_a2, xmm_aa);
					xmm_a3 = _mm_add_epi16(xmm_a3, xmm_ab);
					xmm_a4 = _mm_add_epi16(xmm_a4, xmm_ac);
					xmm_a5 = _mm_add_epi16(xmm_a5, xmm_ad);
					xmm_a6 = _mm_add_epi16(xmm_a6, xmm_ae);
					xmm_a7 = _mm_add_epi16(xmm_a7, xmm_af);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi16(xmm_a2, xmm_a3);
					xmm_a4 = _mm_hadd_epi16(xmm_a4, xmm_a5);
					xmm_a6 = _mm_hadd_epi16(xmm_a6, xmm_a7);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a2);
					xmm_a4 = _mm_hadd_epi16(xmm_a4, xmm_a6);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a4);
					// data-type conversion
					xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, _mm_cvtepi32_pd(xmm_a0));
					xmm_b1 = _mm_add_pd(xmm_b1, _mm_cvtepi32_pd(xmm_a1));
					xmm_b2 = _mm_add_pd(xmm_b2, _mm_cvtepi32_pd(xmm_a2));
					xmm_b3 = _mm_add_pd(xmm_b3, _mm_cvtepi32_pd(xmm_a3));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					xmm_t2 = _mm_set_pd(static_cast<double>(ptr_a5[j]), static_cast<double>(ptr_a4[j]));
					xmm_t3 = _mm_set_pd(static_cast<double>(ptr_a7[j]), static_cast<double>(ptr_a6[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
					xmm_b2 = _mm_add_pd(xmm_b2, xmm_t2);
					xmm_b3 = _mm_add_pd(xmm_b3, xmm_t3);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
			_mm_storeu_pd(b + 4, xmm_b2);
			_mm_storeu_pd(b + 6, xmm_b3);
		}
	};

	template<>
	struct block_sum_double<unsigned char, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, double *b) const
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
			__m128d xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);
			__m128d xmm_b2 = _mm_loadu_pd(b + 4);
			__m128d xmm_b3 = _mm_loadu_pd(b + 6);

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
					// return the horizontal summation
					xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a8);
					xmm_a1 = _mm_add_epi16(xmm_a1, xmm_a9);
					xmm_a2 = _mm_add_epi16(xmm_a2, xmm_aa);
					xmm_a3 = _mm_add_epi16(xmm_a3, xmm_ab);
					xmm_a4 = _mm_add_epi16(xmm_a4, xmm_ac);
					xmm_a5 = _mm_add_epi16(xmm_a5, xmm_ad);
					xmm_a6 = _mm_add_epi16(xmm_a6, xmm_ae);
					xmm_a7 = _mm_add_epi16(xmm_a7, xmm_af);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi16(xmm_a2, xmm_a3);
					xmm_a4 = _mm_hadd_epi16(xmm_a4, xmm_a5);
					xmm_a6 = _mm_hadd_epi16(xmm_a6, xmm_a7);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a2);
					xmm_a4 = _mm_hadd_epi16(xmm_a4, xmm_a6);
					xmm_a0 = _mm_hadd_epi16(xmm_a0, xmm_a4);
					// data-type conversion
					xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, _mm_cvtepi32_pd(xmm_a0));
					xmm_b1 = _mm_add_pd(xmm_b1, _mm_cvtepi32_pd(xmm_a1));
					xmm_b2 = _mm_add_pd(xmm_b2, _mm_cvtepi32_pd(xmm_a2));
					xmm_b3 = _mm_add_pd(xmm_b3, _mm_cvtepi32_pd(xmm_a3));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					xmm_t2 = _mm_set_pd(static_cast<double>(ptr_a5[j]), static_cast<double>(ptr_a4[j]));
					xmm_t3 = _mm_set_pd(static_cast<double>(ptr_a7[j]), static_cast<double>(ptr_a6[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
					xmm_b2 = _mm_add_pd(xmm_b2, xmm_t2);
					xmm_b3 = _mm_add_pd(xmm_b3, xmm_t3);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
			_mm_storeu_pd(b + 4, xmm_b2);
			_mm_storeu_pd(b + 6, xmm_b3);
		}
	};

	template<>
	struct block_sum_double<signed short, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, double *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = ptr_a0 + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					// return the horizontal summation
					xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a4);
					xmm_a1 = _mm_add_epi32(xmm_a1, xmm_a5);
					xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a6);
					xmm_a3 = _mm_add_epi32(xmm_a3, xmm_a7);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a2);
					// data-type conversion
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, _mm_cvtepi32_pd(xmm_a0));
					xmm_b1 = _mm_add_pd(xmm_b1, _mm_cvtepi32_pd(xmm_a1));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<unsigned short, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, double *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = ptr_a0 + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 8)
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
					// return the horizontal summation
					xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a4);
					xmm_a1 = _mm_add_epi32(xmm_a1, xmm_a5);
					xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a6);
					xmm_a3 = _mm_add_epi32(xmm_a3, xmm_a7);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a2);
					// data-type conversion
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, _mm_cvtepi32_pd(xmm_a0));
					xmm_b1 = _mm_add_pd(xmm_b1, _mm_cvtepi32_pd(xmm_a1));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<signed int, cpu_sse3>
	{
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, double *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
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
					xmm_t0 = _mm_cvtepi32_pd(xmm_a0);
					xmm_t1 = _mm_cvtepi32_pd(xmm_a1);
					xmm_t2 = _mm_cvtepi32_pd(xmm_a2);
					xmm_t3 = _mm_cvtepi32_pd(xmm_a3);
					xmm_t4 = _mm_cvtepi32_pd(xmm_a4);
					xmm_t5 = _mm_cvtepi32_pd(xmm_a5);
					xmm_t6 = _mm_cvtepi32_pd(xmm_a6);
					xmm_t7 = _mm_cvtepi32_pd(xmm_a7);
					// return the horizontal summation
					xmm_t0 = _mm_add_pd(xmm_t0, xmm_t4);
					xmm_t1 = _mm_add_pd(xmm_t1, xmm_t5);
					xmm_t2 = _mm_add_pd(xmm_t2, xmm_t6);
					xmm_t3 = _mm_add_pd(xmm_t3, xmm_t7);
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_pd(xmm_t2, xmm_t3);
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t2);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<unsigned int, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, double *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi64x(0x41e0000000000000LL);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3, xmm_s4, xmm_s5, xmm_s6, xmm_s7;
			__m128d xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
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
					xmm_s4 = _mm_shuffle_epi32(xmm_s0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_s5 = _mm_shuffle_epi32(xmm_s1, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_s6 = _mm_shuffle_epi32(xmm_s2, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_s7 = _mm_shuffle_epi32(xmm_s3, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a4 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a5 = _mm_shuffle_epi32(xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a6 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a7 = _mm_shuffle_epi32(xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_s0 = _mm_cvtepi32_epi64(xmm_s0);
					xmm_s1 = _mm_cvtepi32_epi64(xmm_s1);
					xmm_s2 = _mm_cvtepi32_epi64(xmm_s2);
					xmm_s3 = _mm_cvtepi32_epi64(xmm_s3);
					xmm_s4 = _mm_cvtepi32_epi64(xmm_s4);
					xmm_s5 = _mm_cvtepi32_epi64(xmm_s5);
					xmm_s6 = _mm_cvtepi32_epi64(xmm_s6);
					xmm_s7 = _mm_cvtepi32_epi64(xmm_s7);
					xmm_t0 = _mm_castsi128_pd(_mm_and_si128(xmm_s0, val));
					xmm_t1 = _mm_castsi128_pd(_mm_and_si128(xmm_s1, val));
					xmm_t2 = _mm_castsi128_pd(_mm_and_si128(xmm_s2, val));
					xmm_t3 = _mm_castsi128_pd(_mm_and_si128(xmm_s3, val));
					xmm_t4 = _mm_castsi128_pd(_mm_and_si128(xmm_s4, val));
					xmm_t5 = _mm_castsi128_pd(_mm_and_si128(xmm_s5, val));
					xmm_t6 = _mm_castsi128_pd(_mm_and_si128(xmm_s6, val));
					xmm_t7 = _mm_castsi128_pd(_mm_and_si128(xmm_s7, val));
					xmm_t0 = _mm_add_pd(xmm_t0, _mm_cvtepi32_pd(xmm_a0));
					xmm_t1 = _mm_add_pd(xmm_t1, _mm_cvtepi32_pd(xmm_a1));
					xmm_t2 = _mm_add_pd(xmm_t2, _mm_cvtepi32_pd(xmm_a2));
					xmm_t3 = _mm_add_pd(xmm_t3, _mm_cvtepi32_pd(xmm_a3));
					xmm_t4 = _mm_add_pd(xmm_t4, _mm_cvtepi32_pd(xmm_a4));
					xmm_t5 = _mm_add_pd(xmm_t5, _mm_cvtepi32_pd(xmm_a5));
					xmm_t6 = _mm_add_pd(xmm_t6, _mm_cvtepi32_pd(xmm_a6));
					xmm_t7 = _mm_add_pd(xmm_t7, _mm_cvtepi32_pd(xmm_a7));
					// return the horizontal summation
					xmm_t0 = _mm_add_pd(xmm_t0, xmm_t4);
					xmm_t1 = _mm_add_pd(xmm_t1, xmm_t5);
					xmm_t2 = _mm_add_pd(xmm_t2, xmm_t6);
					xmm_t3 = _mm_add_pd(xmm_t3, xmm_t7);
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_pd(xmm_t2, xmm_t3);
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t2);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<float, cpu_sse3>
	{
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, double *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
					// data-type conversion
					xmm_a4 = _mm_shuffle_ps(xmm_a0, xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a5 = _mm_shuffle_ps(xmm_a1, xmm_a1, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a6 = _mm_shuffle_ps(xmm_a2, xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a7 = _mm_shuffle_ps(xmm_a3, xmm_a3, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_t0 = _mm_cvtps_pd(xmm_a0);
					xmm_t1 = _mm_cvtps_pd(xmm_a1);
					xmm_t2 = _mm_cvtps_pd(xmm_a2);
					xmm_t3 = _mm_cvtps_pd(xmm_a3);
					xmm_t4 = _mm_cvtps_pd(xmm_a4);
					xmm_t5 = _mm_cvtps_pd(xmm_a5);
					xmm_t6 = _mm_cvtps_pd(xmm_a6);
					xmm_t7 = _mm_cvtps_pd(xmm_a7);
					// return the horizontal summation
					xmm_t0 = _mm_add_pd(xmm_t0, xmm_t4);
					xmm_t1 = _mm_add_pd(xmm_t1, xmm_t5);
					xmm_t2 = _mm_add_pd(xmm_t2, xmm_t6);
					xmm_t3 = _mm_add_pd(xmm_t3, xmm_t7);
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_pd(xmm_t2, xmm_t3);
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t2);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(static_cast<double>(ptr_a1[j]), static_cast<double>(ptr_a0[j]));
					xmm_t1 = _mm_set_pd(static_cast<double>(ptr_a3[j]), static_cast<double>(ptr_a2[j]));
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<double, cpu_sse3>
	{
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m128d xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_b0 = _mm_loadu_pd(b);
			__m128d xmm_b1 = _mm_loadu_pd(b + 2);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n;)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(ptr_a0 + j);
					xmm_a1 = _mm_loadu_pd(ptr_a1 + j);
					xmm_a2 = _mm_loadu_pd(ptr_a2 + j);
					xmm_a3 = _mm_loadu_pd(ptr_a3 + j);
					j += 2;
					xmm_a4 = _mm_loadu_pd(ptr_a0 + j);
					xmm_a5 = _mm_loadu_pd(ptr_a1 + j);
					xmm_a6 = _mm_loadu_pd(ptr_a2 + j);
					xmm_a7 = _mm_loadu_pd(ptr_a3 + j);
					j += 2;
					// return the horizontal summation
					xmm_a0 = _mm_add_pd(xmm_a0, xmm_a4);
					xmm_a1 = _mm_add_pd(xmm_a1, xmm_a5);
					xmm_a2 = _mm_add_pd(xmm_a2, xmm_a6);
					xmm_a3 = _mm_add_pd(xmm_a3, xmm_a7);
					xmm_a0 = _mm_hadd_pd(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_pd(xmm_a2, xmm_a3);
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_a0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_a2);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_t0 = _mm_set_pd(ptr_a1[j], ptr_a0[j]);
					xmm_t1 = _mm_set_pd(ptr_a3[j], ptr_a2[j]);
					// return the summation
					xmm_b0 = _mm_add_pd(xmm_b0, xmm_t0);
					xmm_b1 = _mm_add_pd(xmm_b1, xmm_t1);
				}
			}
			// store data into memory
			_mm_storeu_pd(b, xmm_b0);
			_mm_storeu_pd(b + 2, xmm_b1);
		}
	};

	template<>
	struct block_sum_double<signed char, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, double *b) const
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
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);
			__m256d ymm_b2 = _mm256_loadu_pd(b + 8);
			__m256d ymm_b3 = _mm256_loadu_pd(b + 12);

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
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_epi16(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_epi16(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_a9);
					ymm_aa = _mm256_hadd_epi16(ymm_aa, ymm_ab);
					ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ad);
					ymm_ae = _mm256_hadd_epi16(ymm_ae, ymm_af);
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_aa);
					ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ae);
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_ac);
					ymm_a1 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a9 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_epi16(ymm_a1, ymm_a9);
					// data-type conversion
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					xmm_a2 = _mm256_extracti128_si256(ymm_a1, 0);
					xmm_a3 = _mm256_extracti128_si256(ymm_a1, 1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, _mm256_cvtepi32_pd(xmm_a0));
					ymm_b1 = _mm256_add_pd(ymm_b1, _mm256_cvtepi32_pd(xmm_a1));
					ymm_b2 = _mm256_add_pd(ymm_b2, _mm256_cvtepi32_pd(xmm_a2));
					ymm_b3 = _mm256_add_pd(ymm_b3, _mm256_cvtepi32_pd(xmm_a3));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_t0 = _mm256_set_pd(
						static_cast<double>(ptr_a3[j]),
						static_cast<double>(ptr_a2[j]),
						static_cast<double>(ptr_a1[j]),
						static_cast<double>(ptr_a0[j]));
					ymm_t1 = _mm256_set_pd(
						static_cast<double>(ptr_a7[j]),
						static_cast<double>(ptr_a6[j]),
						static_cast<double>(ptr_a5[j]),
						static_cast<double>(ptr_a4[j]));
					ymm_t2 = _mm256_set_pd(
						static_cast<double>(ptr_ab[j]),
						static_cast<double>(ptr_aa[j]),
						static_cast<double>(ptr_a9[j]),
						static_cast<double>(ptr_a8[j]));
					ymm_t3 = _mm256_set_pd(
						static_cast<double>(ptr_af[j]),
						static_cast<double>(ptr_ae[j]),
						static_cast<double>(ptr_ad[j]),
						static_cast<double>(ptr_ac[j]));
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_t0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_t1);
					ymm_b2 = _mm256_add_pd(ymm_b2, ymm_t2);
					ymm_b3 = _mm256_add_pd(ymm_b3, ymm_t3);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
			_mm256_storeu_pd(b + 8, ymm_b2);
			_mm256_storeu_pd(b + 12, ymm_b3);
		}
	};

	template<>
	struct block_sum_double<unsigned char, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, double *b) const
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
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);
			__m256d ymm_b2 = _mm256_loadu_pd(b + 8);
			__m256d ymm_b3 = _mm256_loadu_pd(b + 12);

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
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_epi16(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_epi16(ymm_a6, ymm_a7);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_a9);
					ymm_aa = _mm256_hadd_epi16(ymm_aa, ymm_ab);
					ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ad);
					ymm_ae = _mm256_hadd_epi16(ymm_ae, ymm_af);
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_hadd_epi16(ymm_a4, ymm_a6);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_aa);
					ymm_ac = _mm256_hadd_epi16(ymm_ac, ymm_ae);
					ymm_a0 = _mm256_hadd_epi16(ymm_a0, ymm_a4);
					ymm_a8 = _mm256_hadd_epi16(ymm_a8, ymm_ac);
					ymm_a1 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a9 = _mm256_permute2f128_si256(ymm_a0, ymm_a8, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_epi16(ymm_a1, ymm_a9);
					// data-type conversion
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					xmm_a2 = _mm256_extracti128_si256(ymm_a1, 0);
					xmm_a3 = _mm256_extracti128_si256(ymm_a1, 1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, _mm256_cvtepi32_pd(xmm_a0));
					ymm_b1 = _mm256_add_pd(ymm_b1, _mm256_cvtepi32_pd(xmm_a1));
					ymm_b2 = _mm256_add_pd(ymm_b2, _mm256_cvtepi32_pd(xmm_a2));
					ymm_b3 = _mm256_add_pd(ymm_b3, _mm256_cvtepi32_pd(xmm_a3));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_t0 = _mm256_set_pd(
						static_cast<double>(ptr_a3[j]),
						static_cast<double>(ptr_a2[j]),
						static_cast<double>(ptr_a1[j]),
						static_cast<double>(ptr_a0[j]));
					ymm_t1 = _mm256_set_pd(
						static_cast<double>(ptr_a7[j]),
						static_cast<double>(ptr_a6[j]),
						static_cast<double>(ptr_a5[j]),
						static_cast<double>(ptr_a4[j]));
					ymm_t2 = _mm256_set_pd(
						static_cast<double>(ptr_ab[j]),
						static_cast<double>(ptr_aa[j]),
						static_cast<double>(ptr_a9[j]),
						static_cast<double>(ptr_a8[j]));
					ymm_t3 = _mm256_set_pd(
						static_cast<double>(ptr_af[j]),
						static_cast<double>(ptr_ae[j]),
						static_cast<double>(ptr_ad[j]),
						static_cast<double>(ptr_ac[j]));
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_t0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_t1);
					ymm_b2 = _mm256_add_pd(ymm_b2, ymm_t2);
					ymm_b3 = _mm256_add_pd(ymm_b3, ymm_t3);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
			_mm256_storeu_pd(b + 8, ymm_b2);
			_mm256_storeu_pd(b + 12, ymm_b3);
		}
	};

	template<>
	struct block_sum_double<signed short, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, double *b) const
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
			__m256d ymm_t0, ymm_t1;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

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
					// data-type conversion
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					ymm_a2 = _mm256_cvtepi16_epi32(xmm_a2);
					ymm_a3 = _mm256_cvtepi16_epi32(xmm_a3);
					ymm_a4 = _mm256_cvtepi16_epi32(xmm_a4);
					ymm_a5 = _mm256_cvtepi16_epi32(xmm_a5);
					ymm_a6 = _mm256_cvtepi16_epi32(xmm_a6);
					ymm_a7 = _mm256_cvtepi16_epi32(xmm_a7);
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_epi32(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_epi32(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_epi32(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_epi32(ymm_a6, ymm_a7);
					ymm_a0 = _mm256_hadd_epi32(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_hadd_epi32(ymm_a4, ymm_a6);
					ymm_a1 = _mm256_permute2f128_si256(ymm_a0, ymm_a4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a5 = _mm256_permute2f128_si256(ymm_a0, ymm_a4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_epi32(ymm_a1, ymm_a5);
					// data-type conversion
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, _mm256_cvtepi32_pd(xmm_a0));
					ymm_b1 = _mm256_add_pd(ymm_b1, _mm256_cvtepi32_pd(xmm_a1));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_t0 = _mm256_set_pd(
						static_cast<double>(ptr_a3[j]),
						static_cast<double>(ptr_a2[j]),
						static_cast<double>(ptr_a1[j]),
						static_cast<double>(ptr_a0[j]));
					ymm_t1 = _mm256_set_pd(
						static_cast<double>(ptr_a7[j]),
						static_cast<double>(ptr_a6[j]),
						static_cast<double>(ptr_a5[j]),
						static_cast<double>(ptr_a4[j]));
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_t0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_t1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

	template<>
	struct block_sum_double<unsigned short, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, double *b) const
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
			__m256d ymm_t0, ymm_t1;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

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
					// data-type conversion
					ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepu16_epi32(xmm_a1);
					ymm_a2 = _mm256_cvtepu16_epi32(xmm_a2);
					ymm_a3 = _mm256_cvtepu16_epi32(xmm_a3);
					ymm_a4 = _mm256_cvtepu16_epi32(xmm_a4);
					ymm_a5 = _mm256_cvtepu16_epi32(xmm_a5);
					ymm_a6 = _mm256_cvtepu16_epi32(xmm_a6);
					ymm_a7 = _mm256_cvtepu16_epi32(xmm_a7);
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_epi32(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_epi32(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_epi32(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_epi32(ymm_a6, ymm_a7);
					ymm_a0 = _mm256_hadd_epi32(ymm_a0, ymm_a2);
					ymm_a4 = _mm256_hadd_epi32(ymm_a4, ymm_a6);
					ymm_a1 = _mm256_permute2f128_si256(ymm_a0, ymm_a4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a5 = _mm256_permute2f128_si256(ymm_a0, ymm_a4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_epi32(ymm_a1, ymm_a5);
					// data-type conversion
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, _mm256_cvtepi32_pd(xmm_a0));
					ymm_b1 = _mm256_add_pd(ymm_b1, _mm256_cvtepi32_pd(xmm_a1));
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_t0 = _mm256_set_pd(
						static_cast<double>(ptr_a3[j]),
						static_cast<double>(ptr_a2[j]),
						static_cast<double>(ptr_a1[j]),
						static_cast<double>(ptr_a0[j]));
					ymm_t1 = _mm256_set_pd(
						static_cast<double>(ptr_a7[j]),
						static_cast<double>(ptr_a6[j]),
						static_cast<double>(ptr_a5[j]),
						static_cast<double>(ptr_a4[j]));
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_t0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_t1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

	template<>
	struct block_sum_double<signed int, cpu_avx>
	{
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, double *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = ptr_a0 + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			const signed int *ptr_a4 = ptr_a3 + rsa;
			const signed int *ptr_a5 = ptr_a4 + rsa;
			const signed int *ptr_a6 = ptr_a5 + rsa;
			const signed int *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256d ymm_t0, ymm_t1;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

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
					// data-type conversion
					ymm_a0 = _mm256_cvtepi32_pd(xmm_a0);
					ymm_a1 = _mm256_cvtepi32_pd(xmm_a1);
					ymm_a2 = _mm256_cvtepi32_pd(xmm_a2);
					ymm_a3 = _mm256_cvtepi32_pd(xmm_a3);
					ymm_a4 = _mm256_cvtepi32_pd(xmm_a4);
					ymm_a5 = _mm256_cvtepi32_pd(xmm_a5);
					ymm_a6 = _mm256_cvtepi32_pd(xmm_a6);
					ymm_a7 = _mm256_cvtepi32_pd(xmm_a7);
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_pd(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_pd(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_pd(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_pd(ymm_a6, ymm_a7);
					ymm_a1 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a3 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a5 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a7 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_pd(ymm_a1, ymm_a3);
					ymm_a1 = _mm256_add_pd(ymm_a5, ymm_a7);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					xmm_a1 = _mm_set_epi32(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j]);
					// data-type conversion
					ymm_a0 = _mm256_cvtepi32_pd(xmm_a0);
					ymm_a1 = _mm256_cvtepi32_pd(xmm_a1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_t0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_t1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

	template<>
	struct block_sum_double<unsigned int, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, double *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = ptr_a0 + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			const unsigned int *ptr_a4 = ptr_a3 + rsa;
			const unsigned int *ptr_a5 = ptr_a4 + rsa;
			const unsigned int *ptr_a6 = ptr_a5 + rsa;
			const unsigned int *ptr_a7 = ptr_a6 + rsa;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi64x(0x41e0000000000000LL);
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_s0, xmm_s1, xmm_s2, xmm_s3, xmm_s4, xmm_s5, xmm_s6, xmm_s7;
			__m256i ymm_s0, ymm_s1, ymm_s2, ymm_s3, ymm_s4, ymm_s5, ymm_s6, ymm_s7;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
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
					xmm_s0 = _mm_srai_epi32(xmm_a0, 31);
					xmm_s1 = _mm_srai_epi32(xmm_a1, 31);
					xmm_s2 = _mm_srai_epi32(xmm_a2, 31);
					xmm_s3 = _mm_srai_epi32(xmm_a3, 31);
					xmm_s4 = _mm_srai_epi32(xmm_a4, 31);
					xmm_s5 = _mm_srai_epi32(xmm_a5, 31);
					xmm_s6 = _mm_srai_epi32(xmm_a6, 31);
					xmm_s7 = _mm_srai_epi32(xmm_a7, 31);
					xmm_a0 = _mm_and_si128(xmm_a0, abs);
					xmm_a1 = _mm_and_si128(xmm_a1, abs);
					xmm_a2 = _mm_and_si128(xmm_a2, abs);
					xmm_a3 = _mm_and_si128(xmm_a3, abs);
					xmm_a4 = _mm_and_si128(xmm_a4, abs);
					xmm_a5 = _mm_and_si128(xmm_a5, abs);
					xmm_a6 = _mm_and_si128(xmm_a6, abs);
					xmm_a7 = _mm_and_si128(xmm_a7, abs);
					ymm_s0 = _mm256_cvtepi32_epi64(xmm_s0);
					ymm_s1 = _mm256_cvtepi32_epi64(xmm_s1);
					ymm_s2 = _mm256_cvtepi32_epi64(xmm_s2);
					ymm_s3 = _mm256_cvtepi32_epi64(xmm_s3);
					ymm_s4 = _mm256_cvtepi32_epi64(xmm_s4);
					ymm_s5 = _mm256_cvtepi32_epi64(xmm_s5);
					ymm_s6 = _mm256_cvtepi32_epi64(xmm_s6);
					ymm_s7 = _mm256_cvtepi32_epi64(xmm_s7);
					ymm_a0 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s0, val));
					ymm_a1 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s1, val));
					ymm_a2 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s2, val));
					ymm_a3 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s3, val));
					ymm_a4 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s4, val));
					ymm_a5 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s5, val));
					ymm_a6 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s6, val));
					ymm_a7 = _mm256_castsi256_pd(_mm256_and_si256(ymm_s7, val));
					ymm_a0 = _mm256_add_pd(ymm_a0, _mm256_cvtepi32_pd(xmm_a0));
					ymm_a1 = _mm256_add_pd(ymm_a1, _mm256_cvtepi32_pd(xmm_a1));
					ymm_a2 = _mm256_add_pd(ymm_a2, _mm256_cvtepi32_pd(xmm_a2));
					ymm_a3 = _mm256_add_pd(ymm_a3, _mm256_cvtepi32_pd(xmm_a3));
					ymm_a4 = _mm256_add_pd(ymm_a4, _mm256_cvtepi32_pd(xmm_a4));
					ymm_a5 = _mm256_add_pd(ymm_a5, _mm256_cvtepi32_pd(xmm_a5));
					ymm_a6 = _mm256_add_pd(ymm_a6, _mm256_cvtepi32_pd(xmm_a6));
					ymm_a7 = _mm256_add_pd(ymm_a7, _mm256_cvtepi32_pd(xmm_a7));
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_pd(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_pd(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_pd(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_pd(ymm_a6, ymm_a7);
					ymm_a1 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a3 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a5 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a7 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_pd(ymm_a1, ymm_a3);
					ymm_a1 = _mm256_add_pd(ymm_a5, ymm_a7);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_pd(
						static_cast<double>(ptr_a3[j]),
						static_cast<double>(ptr_a2[j]),
						static_cast<double>(ptr_a1[j]),
						static_cast<double>(ptr_a0[j]));
					ymm_a1 = _mm256_set_pd(
						static_cast<double>(ptr_a7[j]),
						static_cast<double>(ptr_a6[j]),
						static_cast<double>(ptr_a5[j]),
						static_cast<double>(ptr_a4[j]));
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

	template<>
	struct block_sum_double<float, cpu_avx>
	{
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, double *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + j);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + j);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + j);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + j);
					xmm_a4 = _mm_loadu_ps(ptr_a4 + j);
					xmm_a5 = _mm_loadu_ps(ptr_a5 + j);
					xmm_a6 = _mm_loadu_ps(ptr_a6 + j);
					xmm_a7 = _mm_loadu_ps(ptr_a7 + j);
					// data-type conversion
					ymm_a0 = _mm256_cvtps_pd(xmm_a0);
					ymm_a1 = _mm256_cvtps_pd(xmm_a1);
					ymm_a2 = _mm256_cvtps_pd(xmm_a2);
					ymm_a3 = _mm256_cvtps_pd(xmm_a3);
					ymm_a4 = _mm256_cvtps_pd(xmm_a4);
					ymm_a5 = _mm256_cvtps_pd(xmm_a5);
					ymm_a6 = _mm256_cvtps_pd(xmm_a6);
					ymm_a7 = _mm256_cvtps_pd(xmm_a7);
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_pd(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_pd(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_pd(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_pd(ymm_a6, ymm_a7);
					ymm_a1 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a3 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a5 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a7 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_pd(ymm_a1, ymm_a3);
					ymm_a1 = _mm256_add_pd(ymm_a5, ymm_a7);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_ps(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					xmm_a1 = _mm_set_ps(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j]);
					// data-type conversion
					ymm_a0 = _mm256_cvtps_pd(xmm_a0);
					ymm_a1 = _mm256_cvtps_pd(xmm_a1);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

	template<>
	struct block_sum_double<double, cpu_avx>
	{
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_a4 = ptr_a3 + rsa;
			const double *ptr_a5 = ptr_a4 + rsa;
			const double *ptr_a6 = ptr_a5 + rsa;
			const double *ptr_a7 = ptr_a6 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256d ymm_b0 = _mm256_loadu_pd(b);
			__m256d ymm_b1 = _mm256_loadu_pd(b + 4);

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(ptr_a0 + j);
					ymm_a1 = _mm256_loadu_pd(ptr_a1 + j);
					ymm_a2 = _mm256_loadu_pd(ptr_a2 + j);
					ymm_a3 = _mm256_loadu_pd(ptr_a3 + j);
					ymm_a4 = _mm256_loadu_pd(ptr_a4 + j);
					ymm_a5 = _mm256_loadu_pd(ptr_a5 + j);
					ymm_a6 = _mm256_loadu_pd(ptr_a6 + j);
					ymm_a7 = _mm256_loadu_pd(ptr_a7 + j);
					// return the horizontal summation
					ymm_a0 = _mm256_hadd_pd(ymm_a0, ymm_a1);
					ymm_a2 = _mm256_hadd_pd(ymm_a2, ymm_a3);
					ymm_a4 = _mm256_hadd_pd(ymm_a4, ymm_a5);
					ymm_a6 = _mm256_hadd_pd(ymm_a6, ymm_a7);
					ymm_a1 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a3 = _mm256_permute2f128_pd(ymm_a0, ymm_a2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a5 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_a7 = _mm256_permute2f128_pd(ymm_a4, ymm_a6, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_a0 = _mm256_add_pd(ymm_a1, ymm_a3);
					ymm_a1 = _mm256_add_pd(ymm_a5, ymm_a7);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_pd(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					ymm_a1 = _mm256_set_pd(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j]);
					// return the summation
					ymm_b0 = _mm256_add_pd(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_pd(ymm_b1, ymm_a1);
				}
			}
			// store data into memory
			_mm256_storeu_pd(b, ymm_b0);
			_mm256_storeu_pd(b + 4, ymm_b1);
		}
	};

} // namespace core

#endif
