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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_DESIGMOID_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_DESIGMOID_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template kernel_desigmoid

	template<class T, cpu_inst_type inst>
	struct kernel_desigmoid
	{
		void operator()(size_t n, const T *a, const T *b, T *c) const
		{
			constexpr size_t block = 4;
			constexpr T one = 1;

			while (n >= block)
			{
				c[0] = a[0] * b[0] * (one - b[0]);
				c[1] = a[1] * b[1] * (one - b[1]);
				c[2] = a[2] * b[2] * (one - b[2]);
				c[3] = a[3] * b[3] * (one - b[3]);
				a += block;
				b += block;
				c += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				c[i] = a[i] * b[i] * (one - b[i]);
		}
	};

	template<>
	struct kernel_desigmoid<float, cpu_sse>
	{
		void operator()(size_t n, const float *a, const float *b, float *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m128 xmm_one = _mm_set1_ps(1.0F);
			__m128 xmm_a, xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				xmm_b = _mm_loadu_ps(b + i);
				// c = a * b * (1 - b);
				xmm_a = _mm_mul_ps(xmm_a, xmm_b);
				xmm_b = _mm_sub_ps(xmm_one, xmm_b);
				xmm_c = _mm_mul_ps(xmm_a, xmm_b);
				// store data into memory
				_mm_storeu_ps(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a[i] * b[i] * (1.0F - b[i]);
		}
	};

	template<>
	struct kernel_desigmoid<double, cpu_sse2>
	{
		void operator()(size_t n, const double *a, const double *b, double *c) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			const __m128d xmm_one = _mm_set1_pd(1.0);
			__m128d xmm_a, xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				xmm_b = _mm_loadu_pd(b + i);
				// c = a * b * (1 - b);
				xmm_a = _mm_mul_pd(xmm_a, xmm_b);
				xmm_b = _mm_sub_pd(xmm_one, xmm_b);
				xmm_c = _mm_mul_pd(xmm_a, xmm_b);
				// store data into memory
				_mm_storeu_pd(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a[i] * b[i] * (1.0 - b[i]);
		}
	};

	template<>
	struct kernel_desigmoid<float, cpu_avx>
	{
		void operator()(size_t n, const float *a, const float *b, float *c) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			const __m256 ymm_one = _mm256_set1_ps(1.0F);
			__m256 ymm_a, ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				ymm_b = _mm256_loadu_ps(b + i);
				// c = a * b * (1 - b);
				ymm_a = _mm256_mul_ps(ymm_a, ymm_b);
				ymm_b = _mm256_sub_ps(ymm_one, ymm_b);
				ymm_c = _mm256_mul_ps(ymm_a, ymm_b);
				// store data into memory
				_mm256_storeu_ps(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a[i] * b[i] * (1.0F - b[i]);
		}
	};

	template<>
	struct kernel_desigmoid<double, cpu_avx>
	{
		void operator()(size_t n, const double *a, const double *b, double *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m256d ymm_one = _mm256_set1_pd(1.0);
			__m256d ymm_a, ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_pd(a + i);
				ymm_b = _mm256_loadu_pd(b + i);
				// c = a * b * (1 - b);
				ymm_a = _mm256_mul_pd(ymm_a, ymm_b);
				ymm_b = _mm256_sub_pd(ymm_one, ymm_b);
				ymm_c = _mm256_mul_pd(ymm_a, ymm_b);
				// store data into memory
				_mm256_storeu_pd(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a[i] * b[i] * (1.0 - b[i]);
		}
	};

} // namespace core

#endif
