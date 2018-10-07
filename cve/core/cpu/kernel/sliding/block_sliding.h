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

#ifndef __CORE_CPU_KERNEL_BOCK_SLIDING_H__
#define __CORE_CPU_KERNEL_BOCK_SLIDING_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_sliding

	template<class T, cpu_inst_type inst>
	struct block_sliding
	{
		void operator()(const T m, const T n, T *a, const T b) const
		{
			T val_b = b;
			T *ptr_c = a + n;

			for (T i = 1; i < m; ++i)
			{
				for (T j = 0; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<signed int, cpu_sse2>
	{
		void operator()(const signed int m, const signed int n, signed int *a, const signed int b) const
		{
			constexpr signed int block = 4;
			const signed int aligned = n & ~(block - 1);
			signed int val_b = b;
			signed int *ptr_c = a + n;
			__m128i xmm_a, xmm_b, xmm_c;

			for (signed int i = 1; i < m; ++i)
			{
				xmm_b = _mm_set1_epi32(val_b);
				for (signed int j = 0; j < aligned; j += block)
				{
					// load data from memory
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// c = a + b;
					xmm_c = _mm_add_epi32(xmm_a, xmm_b);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr_c + j), xmm_c);
				}
				for (signed int j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<unsigned int, cpu_sse2>
	{
		void operator()(const unsigned int m, const unsigned int n, unsigned int *a, const unsigned int b) const
		{
			constexpr unsigned int block = 4;
			const unsigned int aligned = n & ~(block - 1);
			unsigned int val_b = b;
			unsigned int *ptr_c = a + n;
			__m128i xmm_a, xmm_b, xmm_c;

			for (unsigned int i = 1; i < m; ++i)
			{
				xmm_b = _mm_set1_epi32(val_b);
				for (unsigned int j = 0; j < aligned; j += block)
				{
					// load data from memory
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// c = a + b;
					xmm_c = _mm_add_epi32(xmm_a, xmm_b);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr_c + j), xmm_c);
				}
				for (unsigned int j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<signed long long, cpu_sse2>
	{
		void operator()(const signed long long m, const signed long long n, signed long long *a, const signed long long b) const
		{
			constexpr signed long long block = 2;
			const signed long long aligned = n & ~(block - 1);
			signed long long val_b = b;
			signed long long *ptr_c = a + n;
			__m128i xmm_a, xmm_b, xmm_c;

			for (signed long long i = 1; i < m; ++i)
			{
				xmm_b = _mm_set1_epi64x(val_b);
				for (signed long long j = 0; j < aligned; j += block)
				{
					// load data from memory
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// c = a + b;
					xmm_c = _mm_add_epi64(xmm_a, xmm_b);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr_c + j), xmm_c);
				}
				for (signed long long j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<unsigned long long, cpu_sse2>
	{
		void operator()(const unsigned long long m, const unsigned long long n, unsigned long long *a, const unsigned long long b) const
		{
			constexpr unsigned long long block = 2;
			const unsigned long long aligned = n & ~(block - 1);
			unsigned long long val_b = b;
			unsigned long long *ptr_c = a + n;
			__m128i xmm_a, xmm_b, xmm_c;

			for (unsigned long long i = 1; i < m; ++i)
			{
				xmm_b = _mm_set1_epi64x(val_b);
				for (unsigned long long j = 0; j < aligned; j += block)
				{
					// load data from memory
					xmm_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
					// c = a + b;
					xmm_c = _mm_add_epi64(xmm_a, xmm_b);
					// store data into memory
					_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr_c + j), xmm_c);
				}
				for (unsigned long long j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<signed int, cpu_avx2>
	{
		void operator()(const signed int m, const signed int n, signed int *a, const signed int b) const
		{
			constexpr signed int block = 8;
			const signed int aligned = n & ~(block - 1);
			signed int val_b = b;
			signed int *ptr_c = a + n;
			__m256i ymm_a, ymm_b, ymm_c;

			for (signed int i = 1; i < m; ++i)
			{
				ymm_b = _mm256_set1_epi32(val_b);
				for (signed int j = 0; j < aligned; j += block)
				{
					// load data from memory
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// c = a + b;
					ymm_c = _mm256_add_epi32(ymm_a, ymm_b);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr_c + j), ymm_c);
				}
				for (signed int j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<unsigned int, cpu_avx2>
	{
		void operator()(const unsigned int m, const unsigned int n, unsigned int *a, const unsigned int b) const
		{
			constexpr unsigned int block = 8;
			const unsigned int aligned = n & ~(block - 1);
			unsigned int val_b = b;
			unsigned int *ptr_c = a + n;
			__m256i ymm_a, ymm_b, ymm_c;

			for (unsigned int i = 1; i < m; ++i)
			{
				ymm_b = _mm256_set1_epi32(val_b);
				for (unsigned int j = 0; j < aligned; j += block)
				{
					// load data from memory
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// c = a + b;
					ymm_c = _mm256_add_epi32(ymm_a, ymm_b);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr_c + j), ymm_c);
				}
				for (unsigned int j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<signed long long, cpu_avx2>
	{
		void operator()(const signed long long m, const signed long long n, signed long long *a, const signed long long b) const
		{
			constexpr signed long long block = 4;
			const signed long long aligned = n & ~(block - 1);
			signed long long val_b = b;
			signed long long *ptr_c = a + n;
			__m256i ymm_a, ymm_b, ymm_c;

			for (signed long long i = 1; i < m; ++i)
			{
				ymm_b = _mm256_set1_epi64x(val_b);
				for (signed long long j = 0; j < aligned; j += block)
				{
					// load data from memory
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// c = a + b;
					ymm_c = _mm256_add_epi64(ymm_a, ymm_b);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr_c + j), ymm_c);
				}
				for (signed long long j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

	template<>
	struct block_sliding<unsigned long long, cpu_avx2>
	{
		void operator()(const unsigned long long m, const unsigned long long n, unsigned long long *a, const unsigned long long b) const
		{
			constexpr unsigned long long block = 4;
			const unsigned long long aligned = n & ~(block - 1);
			unsigned long long val_b = b;
			unsigned long long *ptr_c = a + n;
			__m256i ymm_a, ymm_b, ymm_c;

			for (unsigned long long i = 1; i < m; ++i)
			{
				ymm_b = _mm256_set1_epi64x(val_b);
				for (unsigned long long j = 0; j < aligned; j += block)
				{
					// load data from memory
					ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
					// c = a + b;
					ymm_c = _mm256_add_epi64(ymm_a, ymm_b);
					// store data into memory
					_mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr_c + j), ymm_c);
				}
				for (unsigned long long j = aligned; j < n; ++j)
					ptr_c[j] = a[j] + val_b;
				val_b += b;
				ptr_c += n;
			}
		}
	};

} // namespace core

#endif
