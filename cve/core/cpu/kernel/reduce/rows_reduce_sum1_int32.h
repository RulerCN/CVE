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

#ifndef __CORE_CPU_KERNEL_ROWS_REDUCE_SUM1_INT32_H__
#define __CORE_CPU_KERNEL_ROWS_REDUCE_SUM1_INT32_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_reduce_sum1_int32
	template<class T, cpu_inst_type inst>
	struct rows_reduce_sum1_int32
	{
		void operator()(size_t m, size_t n, const T *a, size_t rsa, signed int *b) const
		{
			const T *ptr_a;
			signed int val_b0, val_b1, val_b2, val_b3;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_a = a;
				val_b0 = 0;
				val_b1 = 0;
				val_b2 = 0;
				val_b3 = 0;
				for (size_t j = 0; j < n; j += 4)
				{
					val_b0 += static_cast<signed int>(ptr_a[0]);
					val_b1 += static_cast<signed int>(ptr_a[1]);
					val_b2 += static_cast<signed int>(ptr_a[2]);
					val_b3 += static_cast<signed int>(ptr_a[3]);
					ptr_a += 4;
				}
				val_b0 += val_b1;
				val_b2 += val_b3;
				val_b0 += val_b2;
				b[i] += val_b0;
			}
		}
	};

	template<>
	struct rows_reduce_sum1_int32<signed char, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const signed char *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0 = _mm_setzero_si128();

			for (size_t j = 0; j < n; j += 16)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepi8_epi16(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
			}
			// return the horizontal summation
			xmm_a0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
			xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(2, 3, 0, 1));
			xmm_b0 = _mm_add_epi32(xmm_a0, xmm_a1);
			// store data into memory
			*b += reinterpret_cast<signed int*>(&xmm_b0)[0];
		}
	};

	template<>
	struct line_reduce_sum1_int32<unsigned char, cpu_sse41>
	{
		void operator()(size_t n, const signed char *a, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0 = _mm_setzero_si128();

			for (size_t j = 0; j < n; j += 16)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu8_epi16(xmm_a0);
				xmm_a1 = _mm_cvtepu8_epi16(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi16(xmm_a0, xmm_a1);
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
			}
			// return the horizontal summation
			xmm_a0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
			xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(2, 3, 0, 1));
			xmm_b0 = _mm_add_epi32(xmm_a0, xmm_a1);
			// store data into memory
			*b += reinterpret_cast<signed int*>(&xmm_b0)[0];
		}
	};

	template<>
	struct line_reduce_sum1_int32<signed short, cpu_sse41>
	{
		void operator()(size_t n, const signed short *a, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0 = _mm_setzero_si128();

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
			}
			// return the horizontal summation
			xmm_a0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
			xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(2, 3, 0, 1));
			xmm_b0 = _mm_add_epi32(xmm_a0, xmm_a1);
			// store data into memory
			*b += reinterpret_cast<signed int*>(&xmm_b0)[0];
		}
	};

	template<>
	struct line_reduce_sum1_int32<unsigned short, cpu_sse41>
	{
		void operator()(size_t n, const unsigned short *a, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0 = _mm_setzero_si128();

			for (size_t j = 0; j < n; j += 8)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
				// data-type conversion
				xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
				xmm_a0 = _mm_cvtepu16_epi32(xmm_a0);
				xmm_a1 = _mm_cvtepu16_epi32(xmm_a1);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
			}
			// return the horizontal summation
			xmm_a0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
			xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(2, 3, 0, 1));
			xmm_b0 = _mm_add_epi32(xmm_a0, xmm_a1);
			// store data into memory
			*b += reinterpret_cast<signed int*>(&xmm_b0)[0];
		}
	};

	template<>
	struct block_reduce_sum1_int32<signed int, cpu_sse41>
	{
		void operator()(size_t n, const signed int *a, signed int *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128i xmm_b0 = _mm_setzero_si128();

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 1);
				xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 2);
				xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + 3);
				// return the summation
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a1);
				xmm_a2 = _mm_add_epi32(xmm_a2, xmm_a3);
				xmm_a0 = _mm_add_epi32(xmm_a0, xmm_a2);
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				a += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
				// return the summation
				xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				a += bit;
				n -= bit;
			}
			// return the horizontal summation
			xmm_a0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
			xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(2, 3, 0, 1));
			xmm_b0 = _mm_add_epi32(xmm_a0, xmm_a1);
			// store data into memory
			*b += reinterpret_cast<signed int*>(&xmm_b0)[0];
		}
	};

	// Function template kernel_reduce_sum1_int32

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sum1_int32(size_t m, size_t n, const T *a, size_t rsa, signed int *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const struct common_reduce_sum1_int32<T> functor;
		const struct block_reduce_sum1_int32<T, inst> special_functor;

		for (size_t i = 0; i < aligned_m; i += block_m)
		{
			if (aligned_n > 0)
				special_functor(aligned_n, a, rsa, b);
			if (surplus_n > 0)
				functor(block_m, surplus_n, a + aligned_n, rsa, b);
			a += block_rsa;
			b += block_m;
		}
		if (surplus_m > 0)
			functor(surplus_m, n, a, rsa, b);
	}

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sum1_int32(size_t l, size_t m, size_t n, const T *a, size_t rsa, signed int *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const size_t surplus_rsa = surplus_m * rsa;
		const struct common_reduce_sum1_int32<T> functor;
		const struct block_reduce_sum1_int32<T, inst> special_functor;

		for (size_t j = 0; j < l; j++)
		{
			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_n > 0)
					special_functor(aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(block_m, surplus_n, a + aligned_n, rsa, b);
				a += block_rsa;
				b += block_m;
			}
			if (surplus_m > 0)
			{
				functor(surplus_m, n, a, rsa, b);
				a += surplus_rsa;
				b += surplus_m;
			}
		}
	}

} // namespace core

#endif
