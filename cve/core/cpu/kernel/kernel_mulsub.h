/*====================================================================
Copyright (C) 2016-2016 Ruler. All rights reserved.
Author:  Ruler
Address: Nan'an District,Chongqing,China
Contact: 26105499@qq.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. The name of the author may not be used to endorse or promote
products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY RULER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __CORE_CPU_KERNEL_MULSUB_H__
#define __CORE_CPU_KERNEL_MULSUB_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template kernel_mulsub

	template<class T, cpu_inst_type inst>
	struct kernel_mulsub
	{
		void operator()(size_t n, const T a, const T *b, T *c) const
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
	struct kernel_mulsub<float, cpu_sse>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_b = _mm_loadu_ps(b + i);
				xmm_c = _mm_loadu_ps(c + i);
				// c = a * b - c;
				xmm_c = _mm_sub_ps(_mm_mul_ps(xmm_a, xmm_b), xmm_c);
				// store data into memory
				_mm_storeu_ps(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<float, cpu_sse | cpu_fma>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m128 xmm_a = _mm_set1_ps(a);
			__m128 xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_b = _mm_loadu_ps(b + i);
				xmm_c = _mm_loadu_ps(c + i);
				// c = a * b - c;
				xmm_c = _mm_fmsub_ps(xmm_a, xmm_b, xmm_c);
				// store data into memory
				_mm_storeu_ps(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<double, cpu_sse2>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_b = _mm_loadu_pd(b + i);
				xmm_c = _mm_loadu_pd(c + i);
				// c = a * b - c;
				xmm_c = _mm_sub_pd(_mm_mul_pd(xmm_a, xmm_b), xmm_c);
				// store data into memory
				_mm_storeu_pd(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<double, cpu_sse2 | cpu_fma>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			const __m128d xmm_a = _mm_set1_pd(a);
			__m128d xmm_b, xmm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_b = _mm_loadu_pd(b + i);
				xmm_c = _mm_loadu_pd(c + i);
				// c = a * b - c;
				xmm_c = _mm_fmsub_pd(xmm_a, xmm_b, xmm_c);
				// store data into memory
				_mm_storeu_pd(c + i, xmm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<float, cpu_avx>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_b = _mm256_loadu_ps(b + i);
				ymm_c = _mm256_loadu_ps(c + i);
				// c = a * b - c;
				ymm_c = _mm256_sub_ps(_mm256_mul_ps(ymm_a, ymm_b), ymm_c);
				// store data into memory
				_mm256_storeu_ps(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<float, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, const float a, const float *b, float *c) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			const __m256 ymm_a = _mm256_set1_ps(a);
			__m256 ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_b = _mm256_loadu_ps(b + i);
				ymm_c = _mm256_loadu_ps(c + i);
				// c = a * b - c;
				ymm_c = _mm256_fmsub_ps(ymm_a, ymm_b, ymm_c);
				// store data into memory
				_mm256_storeu_ps(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<double, cpu_avx>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_b = _mm256_loadu_pd(b + i);
				ymm_c = _mm256_loadu_pd(c + i);
				// c = a * b - c;
				ymm_c = _mm256_sub_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);
				// store data into memory
				_mm256_storeu_pd(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

	template<>
	struct kernel_mulsub<double, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, const double a, const double *b, double *c) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			const __m256d ymm_a = _mm256_set1_pd(a);
			__m256d ymm_b, ymm_c;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_b = _mm256_loadu_pd(b + i);
				ymm_c = _mm256_loadu_pd(c + i);
				// c = a * b - c;
				ymm_c = _mm256_fmsub_pd(ymm_a, ymm_b, ymm_c);
				// store data into memory
				_mm256_storeu_pd(c + i, ymm_c);
			}
			for (size_t i = aligned; i < n; ++i)
				c[i] = a * b[i] - c[i];
		}
	};

} // namespace core

#endif
