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

#ifndef __CORE_CPU_KERNEL_ROWS_REDUCE_SUMT_FLOAT_H__
#define __CORE_CPU_KERNEL_ROWS_REDUCE_SUMT_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_reduce_sumt_float

	template<class T, cpu_inst_type inst>
	struct rows_reduce_sumt_float
	{
		void operator()(size_t m, size_t n, const T *a, size_t rsa, float *b) const
		{
			const T *ptr_a;
			float *ptr_b;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_a = a;
				ptr_b = b;
				for (size_t j = 0; j < n; j += 4)
				{
					ptr_b[0] += static_cast<float>(ptr_a[0]);
					ptr_b[1] += static_cast<float>(ptr_a[1]);
					ptr_b[2] += static_cast<float>(ptr_a[2]);
					ptr_b[3] += static_cast<float>(ptr_a[3]);
					ptr_a += 4;
					ptr_b += 4;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed char, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const signed char *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					xmm_b1 = _mm_loadu_ps(ptr_b + 4);
					xmm_b2 = _mm_loadu_ps(ptr_b + 8);
					xmm_b3 = _mm_loadu_ps(ptr_b + 12);
					// data-type conversion
					xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi8_epi16(xmm_a0);
					xmm_a2 = _mm_cvtepi8_epi16(xmm_a2);
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
					xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
					xmm_a3 = _mm_cvtepi16_epi32(xmm_a3);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, _mm_cvtepi32_ps(xmm_a0));
					xmm_b1 = _mm_add_ps(xmm_b1, _mm_cvtepi32_ps(xmm_a1));
					xmm_b2 = _mm_add_ps(xmm_b2, _mm_cvtepi32_ps(xmm_a2));
					xmm_b3 = _mm_add_ps(xmm_b3, _mm_cvtepi32_ps(xmm_a3));
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					_mm_storeu_ps(ptr_b + 4, xmm_b1);
					_mm_storeu_ps(ptr_b + 8, xmm_b2);
					_mm_storeu_ps(ptr_b + 12, xmm_b3);
					ptr_b += 16;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned char, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const unsigned char *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					xmm_b1 = _mm_loadu_ps(ptr_b + 4);
					xmm_b2 = _mm_loadu_ps(ptr_b + 8);
					xmm_b3 = _mm_loadu_ps(ptr_b + 12);
					// data-type conversion
					xmm_a2 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepu8_epi16(xmm_a0);
					xmm_a2 = _mm_cvtepu8_epi16(xmm_a2);
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a3 = _mm_shuffle_epi32(xmm_a2, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
					xmm_a2 = _mm_cvtepi16_epi32(xmm_a2);
					xmm_a3 = _mm_cvtepi16_epi32(xmm_a3);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, _mm_cvtepi32_ps(xmm_a0));
					xmm_b1 = _mm_add_ps(xmm_b1, _mm_cvtepi32_ps(xmm_a1));
					xmm_b2 = _mm_add_ps(xmm_b2, _mm_cvtepi32_ps(xmm_a2));
					xmm_b3 = _mm_add_ps(xmm_b3, _mm_cvtepi32_ps(xmm_a3));
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					_mm_storeu_ps(ptr_b + 4, xmm_b1);
					_mm_storeu_ps(ptr_b + 8, xmm_b2);
					_mm_storeu_ps(ptr_b + 12, xmm_b3);
					ptr_b += 16;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed short, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const signed short *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1;
			__m128 xmm_b0, xmm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					xmm_b1 = _mm_loadu_ps(ptr_b + 4);
					// data-type conversion
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepi16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepi16_epi32(xmm_a1);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, _mm_cvtepi32_ps(xmm_a0));
					xmm_b1 = _mm_add_ps(xmm_b1, _mm_cvtepi32_ps(xmm_a1));
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					_mm_storeu_ps(ptr_b + 4, xmm_b1);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned short, cpu_sse41>
	{
		void operator()(size_t m, size_t n, const unsigned short *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1;
			__m128 xmm_b0, xmm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					xmm_b1 = _mm_loadu_ps(ptr_b + 4);
					// data-type conversion
					xmm_a1 = _mm_shuffle_epi32(xmm_a0, _MM_SHUFFLE(1, 0, 3, 2));
					xmm_a0 = _mm_cvtepu16_epi32(xmm_a0);
					xmm_a1 = _mm_cvtepu16_epi32(xmm_a1);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, _mm_cvtepi32_ps(xmm_a0));
					xmm_b1 = _mm_add_ps(xmm_b1, _mm_cvtepi32_ps(xmm_a1));
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					_mm_storeu_ps(ptr_b + 4, xmm_b1);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed int, cpu_sse2>
	{
		void operator()(size_t m, size_t n, const signed int *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0;
			__m128 xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, _mm_cvtepi32_ps(xmm_a0));
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					ptr_b += 4;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned int, cpu_sse2>
	{
		void operator()(size_t m, size_t n, const unsigned int *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			const __m128i abs = _mm_set1_epi32(0x7fffffff);
			const __m128i val = _mm_set1_epi32(0x4f000000);
			__m128i xmm_a0;
			__m128i xmm_s0;
			__m128 xmm_t0;
			__m128 xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					xmm_b0 = _mm_loadu_ps(ptr_b);
					// data-type conversion
					xmm_s0 = _mm_srai_epi32(xmm_a0, 31);
					xmm_a0 = _mm_and_si128(xmm_a0, abs);
					xmm_t0 = _mm_castsi128_ps(_mm_and_si128(xmm_s0, val));
					xmm_t0 = _mm_add_ps(xmm_t0, _mm_cvtepi32_ps(xmm_a0));
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, xmm_t0);
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					ptr_b += 4;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<float, cpu_sse>
	{
		void operator()(size_t m, size_t n, const float *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128 xmm_a0;
			__m128 xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(a + j);
					xmm_b0 = _mm_loadu_ps(ptr_b);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, xmm_a0);
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					ptr_b += 4;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<double, cpu_sse2>
	{
		void operator()(size_t m, size_t n, const double *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128d xmm_a0, xmm_a1;
			__m128 xmm_t0, xmm_t1;
			__m128 xmm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n;)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(a + j);
					j += 2;
					xmm_a1 = _mm_loadu_pd(a + j);
					j += 2;
					xmm_b0 = _mm_loadu_ps(ptr_b);
					// data-type conversion
					xmm_t0 = _mm_cvtpd_ps(xmm_a0);
					xmm_t1 = _mm_cvtpd_ps(xmm_a1);
					xmm_t0 = _mm_movelh_ps(xmm_t0, xmm_t1);
					// return the summation
					xmm_b0 = _mm_add_ps(xmm_b0, xmm_t0);
					// store data into memory
					_mm_storeu_ps(ptr_b, xmm_b0);
					ptr_b += 4;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed char, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const signed char *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0, ymm_a1;
			__m256 ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					ymm_b1 = _mm256_loadu_ps(ptr_b + 8);
					// data-type conversion
					ymm_a0 = _mm256_cvtepi8_epi16(xmm_a0);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, _mm256_cvtepi32_ps(ymm_a0));
					ymm_b1 = _mm256_add_ps(ymm_b1, _mm256_cvtepi32_ps(ymm_a1));
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					_mm256_storeu_ps(ptr_b + 8, ymm_b1);
					ptr_b += 16;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned char, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned char *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0, xmm_a1;
			__m256i ymm_a0, ymm_a1;
			__m256 ymm_b0, ymm_b1;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 16)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					ymm_b1 = _mm256_loadu_ps(ptr_b + 8);
					// data-type conversion
					ymm_a0 = _mm256_cvtepu8_epi16(xmm_a0);
					xmm_a0 = _mm256_extracti128_si256(ymm_a0, 0);
					xmm_a1 = _mm256_extracti128_si256(ymm_a0, 1);
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					ymm_a1 = _mm256_cvtepi16_epi32(xmm_a1);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, _mm256_cvtepi32_ps(ymm_a0));
					ymm_b1 = _mm256_add_ps(ymm_b1, _mm256_cvtepi32_ps(ymm_a1));
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					_mm256_storeu_ps(ptr_b + 8, ymm_b1);
					ptr_b += 16;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed short, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const signed short *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0;
			__m256i ymm_a0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// data-type conversion
					ymm_a0 = _mm256_cvtepi16_epi32(xmm_a0);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, _mm256_cvtepi32_ps(ymm_a0));
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned short, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned short *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m128i xmm_a0;
			__m256i ymm_a0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// data-type conversion
					ymm_a0 = _mm256_cvtepu16_epi32(xmm_a0);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, _mm256_cvtepi32_ps(ymm_a0));
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<signed int, cpu_avx>
	{
		void operator()(size_t m, size_t n, const signed int *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m256i ymm_a0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, _mm256_cvtepi32_ps(ymm_a0));
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<unsigned int, cpu_avx2>
	{
		void operator()(size_t m, size_t n, const unsigned int *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			const __m256i abs = _mm256_set1_epi32(0x7fffffff);
			const __m256i val = _mm256_set1_epi32(0x4f000000);
			__m256i ymm_a0;
			__m256i ymm_s0;
			__m256 ymm_t0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// data-type conversion
					ymm_s0 = _mm256_srai_epi32(ymm_a0, 31);
					ymm_a0 = _mm256_and_si256(ymm_a0, abs);
					ymm_t0 = _mm256_castsi256_ps(_mm256_and_si256(ymm_s0, val));
					ymm_t0 = _mm256_add_ps(ymm_t0, _mm256_cvtepi32_ps(ymm_a0));
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, ymm_t0);
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<float, cpu_avx>
	{
		void operator()(size_t m, size_t n, const float *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m256 ymm_a0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n; j += 8)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_ps(a + j);
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, ymm_a0);
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_reduce_sumt_float<double, cpu_avx>
	{
		void operator()(size_t m, size_t n, const double *a, size_t rsa, float *b) const
		{
			float *ptr_b;
			__m256d ymm_a0, ymm_a1;
			__m128 xmm_t0, xmm_t1;
			__m256 ymm_t0;
			__m256 ymm_b0;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < n;)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(a + j);
					j += 4;
					ymm_a1 = _mm256_loadu_pd(a + j);
					j += 4;
					ymm_b0 = _mm256_loadu_ps(ptr_b);
					// data-type conversion
					xmm_t0 = _mm256_cvtpd_ps(ymm_a0);
					xmm_t1 = _mm256_cvtpd_ps(ymm_a1);
					ymm_t0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm_t0), xmm_t1, 1);
					// return the summation
					ymm_b0 = _mm256_add_ps(ymm_b0, ymm_t0);
					// store data into memory
					_mm256_storeu_ps(ptr_b, ymm_b0);
					ptr_b += 8;
				}
				a += rsa;
			}
		}
	};

} // namespace core

#endif
