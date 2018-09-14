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

#ifndef __CORE_CPU_KERNEL_BLOCK_SUM_INT32_H__
#define __CORE_CPU_KERNEL_BLOCK_SUM_INT32_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_sum_int32

	template<class T, cpu_inst_type inst>
	struct block_sum_int32
	{
		void operator()(size_t, size_t n, const T *a, size_t rsa, signed int *b) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = a + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			signed int val_b0 = b[0];
			signed int val_b1 = b[1];
			signed int val_b2 = b[2];
			signed int val_b3 = b[3];

			for (size_t j = 0; j < n; ++j)
			{
				val_b0 += static_cast<signed int>(ptr_a0[j]);
				val_b1 += static_cast<signed int>(ptr_a1[j]);
				val_b2 += static_cast<signed int>(ptr_a2[j]);
				val_b3 += static_cast<signed int>(ptr_a3[j]);
			}
			b[0] = val_b0;
			b[1] = val_b1;
			b[2] = val_b2;
			b[3] = val_b3;
		}
	};

	template<>
	struct block_sum_int32<signed char, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed int *b) const
		{
			const signed char *ptr_a0 = a;
			const signed char *ptr_a1 = a + rsa;
			const signed char *ptr_a2 = ptr_a1 + rsa;
			const signed char *ptr_a3 = ptr_a2 + rsa;
			const signed char *ptr_a4 = ptr_a3 + rsa;
			const signed char *ptr_a5 = ptr_a4 + rsa;
			const signed char *ptr_a6 = ptr_a5 + rsa;
			const signed char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
			__m128i xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);

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
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
					xmm_b1 = _mm_add_epi32(xmm_b1, xmm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					xmm_a1 = _mm_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
					xmm_b1 = _mm_add_epi32(xmm_b1, xmm_a1);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
		}
	};

	template<>
	struct block_sum_int32<unsigned char, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, signed int *b) const
		{
			const unsigned char *ptr_a0 = a;
			const unsigned char *ptr_a1 = a + rsa;
			const unsigned char *ptr_a2 = ptr_a1 + rsa;
			const unsigned char *ptr_a3 = ptr_a2 + rsa;
			const unsigned char *ptr_a4 = ptr_a3 + rsa;
			const unsigned char *ptr_a5 = ptr_a4 + rsa;
			const unsigned char *ptr_a6 = ptr_a5 + rsa;
			const unsigned char *ptr_a7 = ptr_a6 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7, xmm_a8, xmm_a9, xmm_aa, xmm_ab, xmm_ac, xmm_ad, xmm_ae, xmm_af;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
			__m128i xmm_b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b) + 1);

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
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
					xmm_b1 = _mm_add_epi32(xmm_b1, xmm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					xmm_a1 = _mm_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
					xmm_b1 = _mm_add_epi32(xmm_b1, xmm_a1);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b) + 1, xmm_b1);
		}
	};

	template<>
	struct block_sum_int32<signed short, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed int *b) const
		{
			const signed short *ptr_a0 = a;
			const signed short *ptr_a1 = a + rsa;
			const signed short *ptr_a2 = ptr_a1 + rsa;
			const signed short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

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
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<unsigned short, cpu_sse41>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, signed int *b) const
		{
			const unsigned short *ptr_a0 = a;
			const unsigned short *ptr_a1 = a + rsa;
			const unsigned short *ptr_a2 = ptr_a1 + rsa;
			const unsigned short *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

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
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<signed int, cpu_ssse3>
	{
		void operator()(size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			const signed int *ptr_a0 = a;
			const signed int *ptr_a1 = a + rsa;
			const signed int *ptr_a2 = ptr_a1 + rsa;
			const signed int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					// return the horizontal summation
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a2);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<unsigned int, cpu_ssse3>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, signed int *b) const
		{
			const unsigned int *ptr_a0 = a;
			const unsigned int *ptr_a1 = a + rsa;
			const unsigned int *ptr_a2 = ptr_a1 + rsa;
			const unsigned int *ptr_a3 = ptr_a2 + rsa;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a0 + j));
					xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a1 + j));
					xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a2 + j));
					xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr_a3 + j));
					// return the horizontal summation
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a1);
					xmm_a2 = _mm_hadd_epi32(xmm_a2, xmm_a3);
					xmm_a0 = _mm_hadd_epi32(xmm_a0, xmm_a2);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_epi32(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<float, cpu_ssse3>
	{
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, signed int *b) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

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
					xmm_t0 = _mm_cvtps_epi32(xmm_a0);
					xmm_t1 = _mm_cvtps_epi32(xmm_a1);
					xmm_t2 = _mm_cvtps_epi32(xmm_a2);
					xmm_t3 = _mm_cvtps_epi32(xmm_a3);
					// return the horizontal summation
					xmm_t0 = _mm_hadd_epi32(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_epi32(xmm_t2, xmm_t3);
					xmm_t0 = _mm_hadd_epi32(xmm_t0, xmm_t2);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_ps(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// data-type conversion
					xmm_t0 = _mm_cvtps_epi32(xmm_a0);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<double, cpu_ssse3>
	{
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, signed int *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m128d xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;
			__m128i xmm_b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

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
					// data-type conversion
					xmm_t0 = _mm_cvtpd_epi32(xmm_a0);
					xmm_t1 = _mm_cvtpd_epi32(xmm_a1);
					xmm_t2 = _mm_cvtpd_epi32(xmm_a2);
					xmm_t3 = _mm_cvtpd_epi32(xmm_a3);
					xmm_t4 = _mm_cvtpd_epi32(xmm_a4);
					xmm_t5 = _mm_cvtpd_epi32(xmm_a5);
					xmm_t6 = _mm_cvtpd_epi32(xmm_a6);
					xmm_t7 = _mm_cvtpd_epi32(xmm_a7);
					xmm_t0 = _mm_unpacklo_epi64(xmm_t0, xmm_t4);
					xmm_t1 = _mm_unpacklo_epi64(xmm_t1, xmm_t5);
					xmm_t2 = _mm_unpacklo_epi64(xmm_t2, xmm_t6);
					xmm_t3 = _mm_unpacklo_epi64(xmm_t3, xmm_t7);
					// return the horizontal summation
					xmm_t0 = _mm_hadd_epi32(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_epi32(xmm_t2, xmm_t3);
					xmm_t0 = _mm_hadd_epi32(xmm_t0, xmm_t2);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_a0 = _mm_set_pd(ptr_a1[j], ptr_a0[j]);
					xmm_a1 = _mm_set_pd(ptr_a3[j], ptr_a2[j]);
					// data-type conversion
					xmm_t0 = _mm_cvtpd_epi32(xmm_a0);
					xmm_t1 = _mm_cvtpd_epi32(xmm_a1);
					xmm_t0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
			}
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(b), xmm_b0);
		}
	};

	template<>
	struct block_sum_int32<signed char, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
			__m256i ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_epi32(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]),
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					ymm_a1 = _mm256_set_epi32(
						static_cast<signed int>(ptr_af[j]),
						static_cast<signed int>(ptr_ae[j]),
						static_cast<signed int>(ptr_ad[j]),
						static_cast<signed int>(ptr_ac[j]),
						static_cast<signed int>(ptr_ab[j]),
						static_cast<signed int>(ptr_aa[j]),
						static_cast<signed int>(ptr_a9[j]),
						static_cast<signed int>(ptr_a8[j]));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_epi32(ymm_b1, ymm_a1);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
		}
	};

	template<>
	struct block_sum_int32<unsigned char, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
			__m256i ymm_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b) + 1);

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_epi32(ymm_b1, ymm_a1);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]),
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					ymm_a1 = _mm256_set_epi32(
						static_cast<signed int>(ptr_af[j]),
						static_cast<signed int>(ptr_ae[j]),
						static_cast<signed int>(ptr_ad[j]),
						static_cast<signed int>(ptr_ac[j]),
						static_cast<signed int>(ptr_ab[j]),
						static_cast<signed int>(ptr_aa[j]),
						static_cast<signed int>(ptr_a9[j]),
						static_cast<signed int>(ptr_a8[j]));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
					ymm_b1 = _mm256_add_epi32(ymm_b1, ymm_a1);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b) + 1, ymm_b1);
		}
	};

	template<>
	struct block_sum_int32<signed short, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]),
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_sum_int32<unsigned short, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, signed int *b) const
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
			__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(
						static_cast<signed int>(ptr_a7[j]),
						static_cast<signed int>(ptr_a6[j]),
						static_cast<signed int>(ptr_a5[j]),
						static_cast<signed int>(ptr_a4[j]),
						static_cast<signed int>(ptr_a3[j]),
						static_cast<signed int>(ptr_a2[j]),
						static_cast<signed int>(ptr_a1[j]),
						static_cast<signed int>(ptr_a0[j]));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_sum_int32<signed int, cpu_avx2>
	{
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
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_sum_int32<unsigned int, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, signed int *b) const
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
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

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
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_epi32(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_sum_int32<float, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, signed int *b) const
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
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

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
					// data-type conversion
					ymm_t0 = _mm256_cvtps_epi32(ymm_a0);
					ymm_t1 = _mm256_cvtps_epi32(ymm_a1);
					ymm_t2 = _mm256_cvtps_epi32(ymm_a2);
					ymm_t3 = _mm256_cvtps_epi32(ymm_a3);
					ymm_t4 = _mm256_cvtps_epi32(ymm_a4);
					ymm_t5 = _mm256_cvtps_epi32(ymm_a5);
					ymm_t6 = _mm256_cvtps_epi32(ymm_a6);
					ymm_t7 = _mm256_cvtps_epi32(ymm_a7);
					// return the horizontal summation
					ymm_t0 = _mm256_hadd_epi32(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_epi32(ymm_t2, ymm_t3);
					ymm_t4 = _mm256_hadd_epi32(ymm_t4, ymm_t5);
					ymm_t6 = _mm256_hadd_epi32(ymm_t6, ymm_t7);
					ymm_t0 = _mm256_hadd_epi32(ymm_t0, ymm_t2);
					ymm_t4 = _mm256_hadd_epi32(ymm_t4, ymm_t6);
					ymm_t1 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t5 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_epi32(ymm_t1, ymm_t5);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_ps(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j], ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					// data-type conversion
					ymm_t0 = _mm256_cvtps_epi32(ymm_a0);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

	template<>
	struct block_sum_int32<double, cpu_avx2>
	{
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, signed int *b) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_a4 = ptr_a3 + rsa;
			const double *ptr_a5 = ptr_a4 + rsa;
			const double *ptr_a6 = ptr_a5 + rsa;
			const double *ptr_a7 = ptr_a6 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
			__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7, xmm_t8, xmm_t9, xmm_ta, xmm_tb, xmm_tc, xmm_td, xmm_te, xmm_tf;
			__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256i ymm_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

			if (aligned_n > 0)
			{
				for (size_t j = 0; j < aligned_n;)
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
					j += 4;
					ymm_a8 = _mm256_loadu_pd(ptr_a0 + j);
					ymm_a9 = _mm256_loadu_pd(ptr_a1 + j);
					ymm_aa = _mm256_loadu_pd(ptr_a2 + j);
					ymm_ab = _mm256_loadu_pd(ptr_a3 + j);
					ymm_ac = _mm256_loadu_pd(ptr_a4 + j);
					ymm_ad = _mm256_loadu_pd(ptr_a5 + j);
					ymm_ae = _mm256_loadu_pd(ptr_a6 + j);
					ymm_af = _mm256_loadu_pd(ptr_a7 + j);
					j += 4;
					// data-type conversion
					xmm_t0 = _mm256_cvtpd_epi32(ymm_a0);
					xmm_t1 = _mm256_cvtpd_epi32(ymm_a1);
					xmm_t2 = _mm256_cvtpd_epi32(ymm_a2);
					xmm_t3 = _mm256_cvtpd_epi32(ymm_a3);
					xmm_t4 = _mm256_cvtpd_epi32(ymm_a4);
					xmm_t5 = _mm256_cvtpd_epi32(ymm_a5);
					xmm_t6 = _mm256_cvtpd_epi32(ymm_a6);
					xmm_t7 = _mm256_cvtpd_epi32(ymm_a7);
					xmm_t8 = _mm256_cvtpd_epi32(ymm_a8);
					xmm_t9 = _mm256_cvtpd_epi32(ymm_a9);
					xmm_ta = _mm256_cvtpd_epi32(ymm_aa);
					xmm_tb = _mm256_cvtpd_epi32(ymm_ab);
					xmm_tc = _mm256_cvtpd_epi32(ymm_ac);
					xmm_td = _mm256_cvtpd_epi32(ymm_ad);
					xmm_te = _mm256_cvtpd_epi32(ymm_ae);
					xmm_tf = _mm256_cvtpd_epi32(ymm_af);
					ymm_t0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t0), xmm_t8, 1);
					ymm_t1 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t1), xmm_t9, 1);
					ymm_t2 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t2), xmm_ta, 1);
					ymm_t3 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t3), xmm_tb, 1);
					ymm_t4 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t4), xmm_tc, 1);
					ymm_t5 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t5), xmm_td, 1);
					ymm_t6 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t6), xmm_te, 1);
					ymm_t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t7), xmm_tf, 1);
					// return the horizontal summation
					ymm_t0 = _mm256_hadd_epi32(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_epi32(ymm_t2, ymm_t3);
					ymm_t4 = _mm256_hadd_epi32(ymm_t4, ymm_t5);
					ymm_t6 = _mm256_hadd_epi32(ymm_t6, ymm_t7);
					ymm_t0 = _mm256_hadd_epi32(ymm_t0, ymm_t2);
					ymm_t4 = _mm256_hadd_epi32(ymm_t4, ymm_t6);
					ymm_t1 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t5 = _mm256_permute2f128_si256(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_epi32(ymm_t1, ymm_t5);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_a0 = _mm256_set_pd(ptr_a3[j], ptr_a2[j], ptr_a1[j], ptr_a0[j]);
					ymm_a1 = _mm256_set_pd(ptr_a7[j], ptr_a6[j], ptr_a5[j], ptr_a4[j]);
					// data-type conversion
					xmm_t0 = _mm256_cvtpd_epi32(ymm_a0);
					xmm_t1 = _mm256_cvtpd_epi32(ymm_a1);
					ymm_t0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t0), xmm_t1, 1);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
			}
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), ymm_b0);
		}
	};

} // namespace core

#endif
