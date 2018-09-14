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

#ifndef __CORE_CPU_ROWS_MIN_H__
#define __CORE_CPU_ROWS_MIN_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_min

	template<class T, cpu_inst_type inst>
	struct rows_min
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const T *a, size_t rsa, T *b) const
		{
			const T *ptr_a;
			T val_b0, val_b1, val_b2, val_b3;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_a = a;
				val_b0 = b[i];
				if (aligned_n > 0)
				{
					val_b1 = val_b2 = val_b3 = val_b0;
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						val_b0 = ptr_a[0] < val_b0 ? ptr_a[0] : val_b0;
						val_b1 = ptr_a[1] < val_b1 ? ptr_a[1] : val_b1;
						val_b2 = ptr_a[2] < val_b2 ? ptr_a[2] : val_b2;
						val_b3 = ptr_a[3] < val_b3 ? ptr_a[3] : val_b3;
						ptr_a += 4;
					}
					val_b0 = val_b1 < val_b0 ? val_b1 : val_b0;
					val_b2 = val_b3 < val_b2 ? val_b3 : val_b2;
					val_b0 = val_b2 < val_b0 ? val_b2 : val_b0;
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b0 = a[j] < val_b0 ? a[j] : val_b0;
				}
				b[i] = val_b0;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed char, cpu_sse41>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			signed char val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 16; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epi8(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi8(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi8(xmm_b, xmm_b);
					xmm_b = _mm_min_epi8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi16(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi16(xmm_b, xmm_b);
					xmm_b = _mm_min_epi8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epi8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epi8(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<signed char*>(&xmm_b)[0] ? val_b : reinterpret_cast<signed char*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned char, cpu_sse2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			unsigned char val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 16; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epu8(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi8(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi8(xmm_b, xmm_b);
					xmm_b = _mm_min_epu8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi16(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi16(xmm_b, xmm_b);
					xmm_b = _mm_min_epu8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epu8(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epu8(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<unsigned char*>(&xmm_b)[0] ? val_b : reinterpret_cast<unsigned char*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed short, cpu_sse2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			signed short val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 8; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epi16(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi16(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi16(xmm_b, xmm_b);
					xmm_b = _mm_min_epi16(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epi16(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epi16(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<signed short*>(&xmm_b)[0] ? val_b : reinterpret_cast<signed short*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned short, cpu_sse41>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			unsigned short val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 8; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epu16(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi16(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi16(xmm_b, xmm_b);
					xmm_b = _mm_min_epu16(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epu16(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epu16(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<unsigned short*>(&xmm_b)[0] ? val_b : reinterpret_cast<unsigned short*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed int, cpu_sse41>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			signed int val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 4; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epi32(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epi32(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epi32(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<signed int*>(&xmm_b)[0] ? val_b : reinterpret_cast<signed int*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned int, cpu_sse41>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			unsigned int val_b;
			__m128i xmm_a, xmm_b;
			__m128i xmm_h, xmm_l;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
					for (size_t j = 4; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						// return minimum values
						xmm_b = _mm_min_epu32(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_unpacklo_epi32(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi32(xmm_b, xmm_b);
					xmm_b = _mm_min_epu32(xmm_l, xmm_h);
					xmm_l = _mm_unpacklo_epi64(xmm_b, xmm_b);
					xmm_h = _mm_unpackhi_epi64(xmm_b, xmm_b);
					xmm_b = _mm_min_epu32(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<unsigned int*>(&xmm_b)[0] ? val_b : reinterpret_cast<unsigned int*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<float, cpu_sse>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			float val_b;
			__m128 xmm_a, xmm_b;
			__m128 xmm_l, xmm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_ps(a);
					for (size_t j = 4; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_ps(a + j);
						// return minimum values
						xmm_b = _mm_min_ps(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_shuffle_ps(xmm_b, xmm_b, _MM_SHUFFLE(1, 0, 1, 0));
					xmm_h = _mm_shuffle_ps(xmm_b, xmm_b, _MM_SHUFFLE(3, 2, 3, 2));
					xmm_b = _mm_min_ps(xmm_l, xmm_h);
					xmm_l = _mm_shuffle_ps(xmm_b, xmm_b, _MM_SHUFFLE(2, 0, 2, 0));
					xmm_h = _mm_shuffle_ps(xmm_b, xmm_b, _MM_SHUFFLE(3, 1, 3, 1));
					xmm_b = _mm_min_ps(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<float*>(&xmm_b)[0] ? val_b : reinterpret_cast<float*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<double, cpu_sse2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			double val_b;
			__m128d xmm_a, xmm_b;
			__m128d xmm_l, xmm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					xmm_b = _mm_loadu_pd(a);
					for (size_t j = 2; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_a = _mm_loadu_pd(a + j);
						// return minimum values
						xmm_b = _mm_min_pd(xmm_a, xmm_b);
					}
					// return horizontal minimum values
					xmm_l = _mm_shuffle_pd(xmm_b, xmm_b, _MM_SHUFFLE(0, 0, 0, 0));
					xmm_h = _mm_shuffle_pd(xmm_b, xmm_b, _MM_SHUFFLE(0, 0, 3, 3));
					xmm_b = _mm_min_pd(xmm_l, xmm_h);
					val_b = val_b < reinterpret_cast<double*>(&xmm_b)[0] ? val_b : reinterpret_cast<double*>(&xmm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed char, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			signed char val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 32; j < aligned_n; j += 32)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epi8(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epi8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi8(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi8(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi16(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi16(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi8(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<signed char*>(&ymm_b)[0] ? val_b : reinterpret_cast<signed char*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned char, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			unsigned char val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 32; j < aligned_n; j += 32)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epu8(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epu8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi8(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi8(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi16(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi16(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu8(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu8(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<unsigned char*>(&ymm_b)[0] ? val_b : reinterpret_cast<unsigned char*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed short, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			signed short val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 32; j < aligned_n; j += 32)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epi16(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epi16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi16(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi16(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi16(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<signed short*>(&ymm_b)[0] ? val_b : reinterpret_cast<signed short*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned short, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			unsigned short val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 32; j < aligned_n; j += 32)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epu16(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epu16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi16(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi16(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu16(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu16(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<unsigned short*>(&ymm_b)[0] ? val_b : reinterpret_cast<unsigned short*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<signed int, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			signed int val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 8; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epi32(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epi32(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi32(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epi32(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<signed int*>(&ymm_b)[0] ? val_b : reinterpret_cast<signed int*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<unsigned int, cpu_avx2>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			unsigned int val_b;
			__m256i ymm_a, ymm_b;
			__m256i ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
					for (size_t j = 8; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						// return minimum values
						ymm_b = _mm256_min_epu32(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_si256(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_epu32(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi32(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi32(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu32(ymm_l, ymm_h);
					ymm_l = _mm256_unpacklo_epi64(ymm_b, ymm_b);
					ymm_h = _mm256_unpackhi_epi64(ymm_b, ymm_b);
					ymm_b = _mm256_min_epu32(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<unsigned int*>(&ymm_b)[0] ? val_b : reinterpret_cast<unsigned int*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<float, cpu_avx>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			float val_b;
			__m256 ymm_a, ymm_b;
			__m256 ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_ps(a);
					for (size_t j = 8; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_a = _mm256_loadu_ps(a + j);
						// return minimum values
						ymm_b = _mm256_min_ps(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_ps(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_ps(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_ps(ymm_l, ymm_h);
					ymm_l = _mm256_shuffle_ps(ymm_b, ymm_b, _MM_SHUFFLE(1, 0, 1, 0));
					ymm_h = _mm256_shuffle_ps(ymm_b, ymm_b, _MM_SHUFFLE(3, 2, 3, 2));
					ymm_b = _mm256_min_ps(ymm_l, ymm_h);
					ymm_l = _mm256_shuffle_ps(ymm_b, ymm_b, _MM_SHUFFLE(2, 0, 2, 0));
					ymm_h = _mm256_shuffle_ps(ymm_b, ymm_b, _MM_SHUFFLE(3, 1, 3, 1));
					ymm_b = _mm256_min_ps(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<float*>(&ymm_b)[0] ? val_b : reinterpret_cast<float*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

	template<>
	struct rows_min<double, cpu_avx>
	{
		// b[i] = min(b[i], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			double val_b;
			__m256d ymm_a, ymm_b;
			__m256d ymm_l, ymm_h;

			for (size_t i = 0; i < m; ++i)
			{
				val_b = b[i];
				if (aligned_n > 0)
				{
					ymm_b = _mm256_loadu_pd(a);
					for (size_t j = 4; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_a = _mm256_loadu_pd(a + j);
						// return minimum values
						ymm_b = _mm256_min_pd(ymm_a, ymm_b);
					}
					// return horizontal minimum values
					ymm_l = _mm256_permute2f128_pd(ymm_b, ymm_b, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_h = _mm256_permute2f128_pd(ymm_b, ymm_b, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_b = _mm256_min_pd(ymm_l, ymm_h);
					ymm_l = _mm256_shuffle_pd(ymm_b, ymm_b, _MM_SHUFFLE(0, 0, 0, 0));
					ymm_h = _mm256_shuffle_pd(ymm_b, ymm_b, _MM_SHUFFLE(0, 0, 3, 3));
					ymm_b = _mm256_min_pd(ymm_l, ymm_h);
					val_b = val_b < reinterpret_cast<double*>(&ymm_b)[0] ? val_b : reinterpret_cast<double*>(&ymm_b)[0];
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						val_b = a[j] < val_b ? a[j] : val_b;
				}
				// store data into memory
				b[i] = val_b;
				a += rsa;
			}
		}
	};

} // namespace core

#endif
