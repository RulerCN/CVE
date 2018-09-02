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

#ifndef __CORE_CPU_KERNEL_ROWS_SUM_INT32_H__
#define __CORE_CPU_KERNEL_ROWS_SUM_INT32_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_sum_int32

	template<class T, cpu_inst_type inst>
	struct rows_sum_int32
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
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed char, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const signed char *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
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
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned char, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const unsigned char *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
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
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed short, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const signed short *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
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
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned short, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const unsigned short *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
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
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed int, cpu_ssse3>
	{
		void operator()(size_t m, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
				// return the horizontal summation
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned int, cpu_ssse3>
	{
		void operator()(size_t m, size_t n, const unsigned int *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_a0);
				}
				// return the horizontal summation
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<float, cpu_ssse3>
	{
		void operator()(size_t m, size_t n, const float *a, size_t rsa, signed int *b) const
		{
			__m128 xmm_a0;
			__m128i xmm_t0;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(a + j);
					// data-type conversion
					xmm_t0 = _mm_cvtps_epi32(xmm_a0);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
				// return the horizontal summation
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<double, cpu_ssse3>
	{
		void operator()(size_t m, size_t n, const double *a, size_t rsa, signed int *b) const
		{
			__m128d xmm_a0, xmm_a1;
			__m128i xmm_t0, xmm_t1;
			__m128i xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_b0 = _mm_setzero_si128();
				for (size_t j = 0; j < n;)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(a + j);
					j += 2;
					xmm_a1 = _mm_loadu_pd(a + j);
					j += 2;
					// data-type conversion
					xmm_t0 = _mm_cvtpd_epi32(xmm_a0);
					xmm_t1 = _mm_cvtpd_epi32(xmm_a1);
					xmm_t0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
					// return the summation
					xmm_b0 = _mm_add_epi32(xmm_b0, xmm_t0);
				}
				// return the horizontal summation
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				xmm_b0 = _mm_hadd_epi32(xmm_b0, xmm_b0);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&xmm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed char, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const signed char *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0, ymm_a1;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// data-type conversion
					ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					// return the summation
					ymm_a0 = _mm256_add_epi32(ymm_a0, ymm_a1);
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned char, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned char *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0, ymm_a1;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// data-type conversion
					ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					// return the summation
					ymm_a0 = _mm256_add_epi32(ymm_a0, ymm_a1);
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed short, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const signed short *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0;
			__m256i ymm_a0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// data-type conversion
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned short, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned short *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a0;
			__m256i ymm_a0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// data-type conversion
					ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<signed int, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			__m256i ymm_a0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<unsigned int, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned int *a, size_t rsa, signed int *b) const
		{
			__m256i ymm_a0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_a0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<float, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const float *a, size_t rsa, signed int *b) const
		{
			__m256 ymm_a0;
			__m256i ymm_t0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_ps(a + j);
					// data-type conversion
					ymm_t0 = _mm256_cvtps_epi32(ymm_a0);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_sum_int32<double, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const double *a, size_t rsa, signed int *b) const
		{
			__m256d ymm_a0, ymm_a1;
			__m128i xmm_t0, xmm_t1;
			__m256i ymm_t0;
			__m256i ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_b0 = _mm256_setzero_si256();
				for (size_t j = 0; j < n;)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(a + j);
					j += 4;
					ymm_a1 = _mm256_loadu_pd(a + j);
					j += 4;
					// data-type conversion
					xmm_t0 = _mm256_cvtpd_epi32(ymm_a0);
					xmm_t1 = _mm256_cvtpd_epi32(ymm_a1);
					ymm_t0 = _mm256_insertf128_si256(_mm256_castsi128_si256(xmm_t0), xmm_t1, 1);
					// return the summation
					ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_t0);
				}
				// return the horizontal summation
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b0 = _mm256_hadd_epi32(ymm_b0, ymm_b0);
				ymm_b1 = _mm256_permute2f128_si256(ymm_b0, ymm_b0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_b0 = _mm256_add_epi32(ymm_b0, ymm_b1);
				// store data into memory
				b[i] += reinterpret_cast<signed int*>(&ymm_b0)[0];
				a += rsa;
			}
		}
	};

} // namespace core

#endif
