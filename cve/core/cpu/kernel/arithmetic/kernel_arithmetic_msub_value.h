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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_MSUB_VALUE_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_MSUB_VALUE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_msub_value

	template<class T, cpu_inst_type inst>
	struct kernel_msub_value
	{
		void operator()(size_t n, T a, const T *b, T *c) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				c[0] = a * b[0] - c[0];
				c[1] = a * b[1] - c[1];
				c[2] = a * b[2] - c[2];
				c[3] = a * b[3] - c[3];
				c[4] = a * b[4] - c[4];
				c[5] = a * b[5] - c[5];
				c[6] = a * b[6] - c[6];
				c[7] = a * b[7] - c[7];
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<float, cpu_sse>
	{
		void operator()(size_t n, float a, const float *b, float *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				xmm_c0 = _mm_loadu_ps(c);
				xmm_c1 = _mm_loadu_ps(c + 4);
				xmm_c2 = _mm_loadu_ps(c + 8);
				xmm_c3 = _mm_loadu_ps(c + 12);
				// c = a * b - c;
				xmm_c0 = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b0), xmm_c0);
				xmm_c1 = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b1), xmm_c1);
				xmm_c2 = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b2), xmm_c2);
				xmm_c3 = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b3), xmm_c3);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				_mm_storeu_ps(c + 4, xmm_c1);
				_mm_storeu_ps(c + 8, xmm_c2);
				_mm_storeu_ps(c + 12, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_c0 = _mm_loadu_ps(c);
				// c = a * b - c;
				xmm_c0 = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b0), xmm_c0);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<float, cpu_sse | cpu_fma>
	{
		void operator()(size_t n, float a, const float *b, float *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				xmm_c0 = _mm_loadu_ps(c);
				xmm_c1 = _mm_loadu_ps(c + 4);
				xmm_c2 = _mm_loadu_ps(c + 8);
				xmm_c3 = _mm_loadu_ps(c + 12);
				// c = a * b - c;
				xmm_c0 = _mm_fmsub_ps(xmm_a, xmm_b0, xmm_c0);
				xmm_c1 = _mm_fmsub_ps(xmm_a, xmm_b1, xmm_c1);
				xmm_c2 = _mm_fmsub_ps(xmm_a, xmm_b2, xmm_c2);
				xmm_c3 = _mm_fmsub_ps(xmm_a, xmm_b3, xmm_c3);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				_mm_storeu_ps(c + 4, xmm_c1);
				_mm_storeu_ps(c + 8, xmm_c2);
				_mm_storeu_ps(c + 12, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(b);
				xmm_c0 = _mm_loadu_ps(c);
				// c = a * b - c;
				xmm_c0 = _mm_fmsub_ps(xmm_a, xmm_b0, xmm_c0);
				// store data into memory
				_mm_storeu_ps(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<double, cpu_sse2>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				xmm_c0 = _mm_loadu_pd(c);
				xmm_c1 = _mm_loadu_pd(c + 2);
				xmm_c2 = _mm_loadu_pd(c + 4);
				xmm_c3 = _mm_loadu_pd(c + 6);
				// c = a * b - c;
				xmm_c0 = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b0), xmm_c0);
				xmm_c1 = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b1), xmm_c1);
				xmm_c2 = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b2), xmm_c2);
				xmm_c3 = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b3), xmm_c3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_c0 = _mm_loadu_pd(c);
				// c = a * b - c;
				xmm_c0 = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b0), xmm_c0);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<double, cpu_sse2 | cpu_fma>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128d xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			while (n > block)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				xmm_c0 = _mm_loadu_pd(c);
				xmm_c1 = _mm_loadu_pd(c + 2);
				xmm_c2 = _mm_loadu_pd(c + 4);
				xmm_c3 = _mm_loadu_pd(c + 6);
				// c = a * b - c;
				xmm_c0 = _mm_fmsub_pd(xmm_a, xmm_b0, xmm_c0);
				xmm_c1 = _mm_fmsub_pd(xmm_a, xmm_b1, xmm_c1);
				xmm_c2 = _mm_fmsub_pd(xmm_a, xmm_b2, xmm_c2);
				xmm_c3 = _mm_fmsub_pd(xmm_a, xmm_b3, xmm_c3);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				_mm_storeu_pd(c + 2, xmm_c1);
				_mm_storeu_pd(c + 4, xmm_c2);
				_mm_storeu_pd(c + 6, xmm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(b);
				xmm_c0 = _mm_loadu_pd(c);
				// c = a * b - c;
				xmm_c0 = _mm_fmsub_pd(xmm_a, xmm_b0, xmm_c0);
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<float, cpu_avx>
	{
		void operator()(size_t n, float a, const float *b, float *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				ymm_c0 = _mm256_loadu_ps(c);
				ymm_c1 = _mm256_loadu_ps(c + 8);
				ymm_c2 = _mm256_loadu_ps(c + 16);
				ymm_c3 = _mm256_loadu_ps(c + 24);
				// c = a * b - c;
				ymm_c0 = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b0), ymm_c0);
				ymm_c1 = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b1), ymm_c1);
				ymm_c2 = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b2), ymm_c2);
				ymm_c3 = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b3), ymm_c3);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				_mm256_storeu_ps(c + 8, ymm_c1);
				_mm256_storeu_ps(c + 16, ymm_c2);
				_mm256_storeu_ps(c + 24, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_c0 = _mm256_loadu_ps(c);
				// c = a * b - c;
				ymm_c0 = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b0), ymm_c0);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<float, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, float a, const float *b, float *c) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				ymm_c0 = _mm256_loadu_ps(c);
				ymm_c1 = _mm256_loadu_ps(c + 8);
				ymm_c2 = _mm256_loadu_ps(c + 16);
				ymm_c3 = _mm256_loadu_ps(c + 24);
				// c = a * b - c;
				ymm_c0 = _mm256_fmsub_ps(ymm_a, ymm_b0, ymm_c0);
				ymm_c1 = _mm256_fmsub_ps(ymm_a, ymm_b1, ymm_c1);
				ymm_c2 = _mm256_fmsub_ps(ymm_a, ymm_b2, ymm_c2);
				ymm_c3 = _mm256_fmsub_ps(ymm_a, ymm_b3, ymm_c3);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				_mm256_storeu_ps(c + 8, ymm_c1);
				_mm256_storeu_ps(c + 16, ymm_c2);
				_mm256_storeu_ps(c + 24, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_c0 = _mm256_loadu_ps(c);
				// c = a * b - c;
				ymm_c0 = _mm256_fmsub_ps(ymm_a, ymm_b0, ymm_c0);
				// store data into memory
				_mm256_storeu_ps(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<double, cpu_avx>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				ymm_c0 = _mm256_loadu_pd(c);
				ymm_c1 = _mm256_loadu_pd(c + 4);
				ymm_c2 = _mm256_loadu_pd(c + 8);
				ymm_c3 = _mm256_loadu_pd(c + 12);
				// c = a * b - c;
				ymm_c0 = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b0), ymm_c0);
				ymm_c1 = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b1), ymm_c1);
				ymm_c2 = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b2), ymm_c2);
				ymm_c3 = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b3), ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_c0 = _mm256_loadu_pd(c);
				// c = a * b - c;
				ymm_c0 = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b0), ymm_c0);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_msub_value<double, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, double a, const double *b, double *c) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			while (n > block)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				ymm_c0 = _mm256_loadu_pd(c);
				ymm_c1 = _mm256_loadu_pd(c + 4);
				ymm_c2 = _mm256_loadu_pd(c + 8);
				ymm_c3 = _mm256_loadu_pd(c + 12);
				// c = a * b - c;
				ymm_c0 = _mm256_fmsub_pd(ymm_a, ymm_b0, ymm_c0);
				ymm_c1 = _mm256_fmsub_pd(ymm_a, ymm_b1, ymm_c1);
				ymm_c2 = _mm256_fmsub_pd(ymm_a, ymm_b2, ymm_c2);
				ymm_c3 = _mm256_fmsub_pd(ymm_a, ymm_b3, ymm_c3);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				_mm256_storeu_pd(c + 4, ymm_c1);
				_mm256_storeu_pd(c + 8, ymm_c2);
				_mm256_storeu_pd(c + 12, ymm_c3);
				b += block;
				c += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_c0 = _mm256_loadu_pd(c);
				// c = a * b - c;
				ymm_c0 = _mm256_fmsub_pd(ymm_a, ymm_b0, ymm_c0);
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				b += bit;
				c += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

} // namespace core

#endif
