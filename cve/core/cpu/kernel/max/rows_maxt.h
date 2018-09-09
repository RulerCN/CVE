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

#ifndef __CORE_CPU_ROWS_MAXT_H__
#define __CORE_CPU_ROWS_MAXT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_maxt
	template<class T, cpu_inst_type inst>
	struct rows_maxt
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const T *a, size_t rsa, T *b) const
		{
			for (size_t i = 0; i < m; ++i)
			{
				for (size_t j = 0; j < aligned_n;)
				{
					b[j] = a[j] > b[j] ? a[j] : b[j]; ++j;
					b[j] = a[j] > b[j] ? a[j] : b[j]; ++j;
					b[j] = a[j] > b[j] ? a[j] : b[j]; ++j;
					b[j] = a[j] > b[j] ? a[j] : b[j]; ++j;
				}
				for (size_t j = aligned_n; j < n; ++j)
					b[j] = a[j] > b[j] ? a[j] : b[j];
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed char, cpu_sse41>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epi8(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned char, cpu_sse2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epu8(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed short, cpu_sse2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epi16(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned short, cpu_sse41>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epu16(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed int, cpu_sse41>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epi32(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned int, cpu_sse41>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			__m128i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
						xmm_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
						// return maximum values
						xmm_b = _mm_max_epu32(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_si128(reinterpret_cast<__m128i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<float, cpu_sse>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			__m128 xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_ps(a + j);
						xmm_b = _mm_loadu_ps(b + j);
						// return maximum values
						xmm_b = _mm_max_ps(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_ps(b + j, xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<double, cpu_sse2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			__m128d xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_a = _mm_loadu_pd(a + j);
						xmm_b = _mm_loadu_pd(b + j);
						// return maximum values
						xmm_b = _mm_max_pd(xmm_a, xmm_b);
						// store data into memory
						_mm_storeu_pd(b + j, xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed char, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed char *a, size_t rsa, signed char *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 32)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epi8(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned char, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned char *a, size_t rsa, unsigned char *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 32)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epu8(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed short, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed short *a, size_t rsa, signed short *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epi16(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned short, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned short *a, size_t rsa, unsigned short *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 16)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epu16(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<signed int, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const signed int *a, size_t rsa, signed int *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epi32(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<unsigned int, cpu_avx2>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const unsigned int *a, size_t rsa, unsigned int *b) const
		{
			__m256i xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
						xmm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
						// return maximum values
						xmm_b = _mm256_max_epu32(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(b + j), xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<float, cpu_avx>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, size_t rsa, float *b) const
		{
			__m256 xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						xmm_a = _mm256_loadu_ps(a + j);
						xmm_b = _mm256_loadu_ps(b + j);
						// return maximum values
						xmm_b = _mm256_max_ps(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_ps(b + j, xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

	template<>
	struct rows_maxt<double, cpu_avx>
	{
		// b[j] = max(b[j], a[i][j])
		void operator()(size_t m, size_t aligned_n, size_t n, const double *a, size_t rsa, double *b) const
		{
			__m256d xmm_a, xmm_b;

			for (size_t i = 0; i < m; ++i)
			{
				if (aligned_n > 0)
				{
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_a = _mm256_loadu_pd(a + j);
						xmm_b = _mm256_loadu_pd(b + j);
						// return maximum values
						xmm_b = _mm256_max_pd(xmm_a, xmm_b);
						// store data into memory
						_mm256_storeu_pd(b + j, xmm_b);
					}
				}
				if (aligned_n < n)
				{
					for (size_t j = aligned_n; j < n; ++j)
						b[j] = a[j] > b[j] ? a[j] : b[j];
				}
				a += rsa;
			}
		}
	};

} // namespace core

#endif
