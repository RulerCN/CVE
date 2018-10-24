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

#ifndef __CORE_CPU_KERNEL_FILL_LINEAR_H__
#define __CORE_CPU_KERNEL_FILL_LINEAR_H__

#include "../../cpu_inst.h"

namespace core
{
	// Function template kernel_fill_linear

	template<class T, cpu_inst_type inst>
	void kernel_fill_linear(size_t n, T a, T b, T *c)
	{
		constexpr size_t block = 4;
		const T val_b = b * 4;
		T val_c0 = a;
		T val_c1 = a + b;
		T val_c2 = val_c1 + b;
		T val_c3 = val_c2 + b;

		while (n >= block)
		{
			c[0] = val_c0;
			c[1] = val_c1;
			c[2] = val_c2;
			c[3] = val_c3;
			val_c0 += val_b;
			val_c1 += val_b;
			val_c2 += val_b;
			val_c3 += val_b;
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
		{
			c[i] = val_c0;
			val_c0 += b;
		}
	}

	template<>
	void kernel_fill_linear<signed char, cpu_sse2>(size_t n, signed char a, signed char b, signed char *c)
	{
		constexpr size_t block = 16;
		const __m128i xmm_b = _mm_set1_epi8(b << 4);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed char*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi8(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed char*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned char, cpu_sse2>(size_t n, unsigned char a, unsigned char b, unsigned char *c)
	{
		constexpr size_t block = 16;
		const __m128i xmm_b = _mm_set1_epi8(b << 4);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned char*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi8(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned char*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed short, cpu_sse2>(size_t n, signed short a, signed short b, signed short *c)
	{
		constexpr size_t block = 8;
		const __m128i xmm_b = _mm_set1_epi16(b << 3);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed short*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi16(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed short*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned short, cpu_sse2>(size_t n, unsigned short a, unsigned short b, unsigned short *c)
	{
		constexpr size_t block = 8;
		const __m128i xmm_b = _mm_set1_epi16(b << 3);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned short*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi16(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned short*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed int, cpu_sse2>(size_t n, signed int a, signed int b, signed int *c)
	{
		constexpr size_t block = 4;
		const __m128i xmm_b = _mm_set1_epi32(b << 2);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed int*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi32(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed int*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned int, cpu_sse2>(size_t n, unsigned int a, unsigned int b, unsigned int *c)
	{
		constexpr size_t block = 4;
		const __m128i xmm_b = _mm_set1_epi32(b << 2);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned int*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi32(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned int*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed long long, cpu_sse2>(size_t n, signed long long a, signed long long b, signed long long *c)
	{
		constexpr size_t block = 2;
		const __m128i xmm_b = _mm_set1_epi64x(b << 1);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed long long*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi64(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed long long*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned long long, cpu_sse2>(size_t n, unsigned long long a, unsigned long long b, unsigned long long *c)
	{
		constexpr size_t block = 2;
		const __m128i xmm_b = _mm_set1_epi64x(b << 1);
		__m128i xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned long long*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(c), xmm_c);
			// c += b;
			xmm_c = _mm_add_epi64(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned long long*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<float, cpu_sse>(size_t n, float a, float b, float *c)
	{
		constexpr size_t block = 4;
		const __m128 xmm_b = _mm_set1_ps(b * 4.0F);
		__m128 xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<float*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_ps(c, xmm_c);
			// c += b;
			xmm_c = _mm_add_ps(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<float*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<double, cpu_sse2>(size_t n, double a, double b, double *c)
	{
		constexpr size_t block = 2;
		const __m128d xmm_b = _mm_set1_pd(b * 2.0);
		__m128d xmm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<double*>(&xmm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm_storeu_pd(c, xmm_c);
			// c += b;
			xmm_c = _mm_add_pd(xmm_c, xmm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<double*>(&xmm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed char, cpu_avx2>(size_t n, signed char a, signed char b, signed char *c)
	{
		constexpr size_t block = 32;
		const __m256i ymm_b = _mm256_set1_epi8(b << 5);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed char*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi8(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed char*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned char, cpu_avx2>(size_t n, unsigned char a, unsigned char b, unsigned char *c)
	{
		constexpr size_t block = 32;
		const __m256i ymm_b = _mm256_set1_epi8(b << 5);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned char*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi8(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned char*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed short, cpu_avx2>(size_t n, signed short a, signed short b, signed short *c)
	{
		constexpr size_t block = 16;
		const __m256i ymm_b = _mm256_set1_epi16(b << 4);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed short*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi16(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed short*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned short, cpu_avx2>(size_t n, unsigned short a, unsigned short b, unsigned short *c)
	{
		constexpr size_t block = 16;
		const __m256i ymm_b = _mm256_set1_epi16(b << 4);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned short*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi16(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned short*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed int, cpu_avx2>(size_t n, signed int a, signed int b, signed int *c)
	{
		constexpr size_t block = 8;
		const __m256i ymm_b = _mm256_set1_epi32(b << 3);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed int*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi32(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed int*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned int, cpu_avx2>(size_t n, unsigned int a, unsigned int b, unsigned int *c)
	{
		constexpr size_t block = 8;
		const __m256i ymm_b = _mm256_set1_epi32(b << 3);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned int*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi32(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned int*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<signed long long, cpu_avx2>(size_t n, signed long long a, signed long long b, signed long long *c)
	{
		constexpr size_t block = 4;
		const __m256i ymm_b = _mm256_set1_epi64x(b << 2);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<signed long long*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi64(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<signed long long*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<unsigned long long, cpu_avx2>(size_t n, unsigned long long a, unsigned long long b, unsigned long long *c)
	{
		constexpr size_t block = 4;
		const __m256i ymm_b = _mm256_set1_epi64x(b << 2);
		__m256i ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<unsigned long long*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(c), ymm_c);
			// c += b;
			ymm_c = _mm256_add_epi64(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<unsigned long long*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<float, cpu_avx>(size_t n, float a, float b, float *c)
	{
		constexpr size_t block = 8;
		const __m256 ymm_b = _mm256_set1_ps(b * 8.0F);
		__m256 ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<float*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_ps(c, ymm_c);
			// c += b;
			ymm_c = _mm256_add_ps(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<float*>(&ymm_c)[i];
	}

	template<>
	void kernel_fill_linear<double, cpu_avx>(size_t n, double a, double b, double *c)
	{
		constexpr size_t block = 4;
		const __m256d ymm_b = _mm256_set1_pd(b * 4.0);
		__m256d ymm_c;

		for (size_t i = 0; i < block; ++i)
		{
			reinterpret_cast<double*>(&ymm_c)[i] = a;
			a += b;
		}
		while (n >= block)
		{
			// store data into memory
			_mm256_storeu_pd(c, ymm_c);
			// c += b;
			ymm_c = _mm256_add_pd(ymm_c, ymm_b);
			c += block;
			n -= block;
		}
		for (size_t i = 0; i < n; ++i)
			c[i] = reinterpret_cast<double*>(&ymm_c)[i];
	}

} // namespace core

#endif
