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

#ifndef __CORE_CPU_KERNEL_MUL_H__
#define __CORE_CPU_KERNEL_MUL_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template kernel_mul

	template<class T, cpu_inst_type inst>
	struct kernel_mul
	{
		void operator()(size_t n, const T *a, T *b) const
		{
			constexpr size_t block = 8;

			while (n > block)
			{
				b[0] *= a[0];
				b[1] *= a[1];
				b[2] *= a[2];
				b[3] *= a[3];
				b[4] *= a[4];
				b[5] *= a[5];
				b[6] *= a[6];
				b[7] *= a[7];
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] *= a[i];
		}
	};

	template<>
	struct kernel_mul<float, cpu_sse>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				xmm_a1 = _mm_loadu_ps(a + 4);
				xmm_a2 = _mm_loadu_ps(a + 8);
				xmm_a3 = _mm_loadu_ps(a + 12);
				xmm_b0 = _mm_loadu_ps(b);
				xmm_b1 = _mm_loadu_ps(b + 4);
				xmm_b2 = _mm_loadu_ps(b + 8);
				xmm_b3 = _mm_loadu_ps(b + 12);
				// b = b + a;
				xmm_b0 = _mm_mul_ps(xmm_b0, xmm_a0);
				xmm_b1 = _mm_mul_ps(xmm_b1, xmm_a1);
				xmm_b2 = _mm_mul_ps(xmm_b2, xmm_a2);
				xmm_b3 = _mm_mul_ps(xmm_b3, xmm_a3);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				_mm_storeu_ps(b + 4, xmm_b1);
				_mm_storeu_ps(b + 8, xmm_b2);
				_mm_storeu_ps(b + 12, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(a);
				xmm_b0 = _mm_loadu_ps(b);
				// b = b + a;
				xmm_b0 = _mm_mul_ps(xmm_b0, xmm_a0);
				// store data into memory
				_mm_storeu_ps(b, xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] *= a[i];
		}
	};

	template<>
	struct kernel_mul<double, cpu_sse2>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 8;
			constexpr size_t bit = 2;
			__m128d xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128d xmm_b0, xmm_b1, xmm_b2, xmm_b3;

			while (n > block)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				xmm_a1 = _mm_loadu_pd(a + 2);
				xmm_a2 = _mm_loadu_pd(a + 4);
				xmm_a3 = _mm_loadu_pd(a + 6);
				xmm_b0 = _mm_loadu_pd(b);
				xmm_b1 = _mm_loadu_pd(b + 2);
				xmm_b2 = _mm_loadu_pd(b + 4);
				xmm_b3 = _mm_loadu_pd(b + 6);
				// b = b + a;
				xmm_b0 = _mm_mul_pd(xmm_b0, xmm_a0);
				xmm_b1 = _mm_mul_pd(xmm_b1, xmm_a1);
				xmm_b2 = _mm_mul_pd(xmm_b2, xmm_a2);
				xmm_b3 = _mm_mul_pd(xmm_b3, xmm_a3);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				_mm_storeu_pd(b + 2, xmm_b1);
				_mm_storeu_pd(b + 4, xmm_b2);
				_mm_storeu_pd(b + 6, xmm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(a);
				xmm_b0 = _mm_loadu_pd(b);
				// b = b + a;
				xmm_b0 = _mm_mul_pd(xmm_b0, xmm_a0);
				// store data into memory
				_mm_storeu_pd(b, xmm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] *= a[i];
		}
	};

	template<>
	struct kernel_mul<float, cpu_avx>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 32;
			constexpr size_t bit = 8;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(a);
				ymm_a1 = _mm256_loadu_ps(a + 8);
				ymm_a2 = _mm256_loadu_ps(a + 16);
				ymm_a3 = _mm256_loadu_ps(a + 24);
				ymm_b0 = _mm256_loadu_ps(b);
				ymm_b1 = _mm256_loadu_ps(b + 8);
				ymm_b2 = _mm256_loadu_ps(b + 16);
				ymm_b3 = _mm256_loadu_ps(b + 24);
				// b = b + a;
				ymm_b0 = _mm256_mul_ps(ymm_b0, ymm_a0);
				ymm_b1 = _mm256_mul_ps(ymm_b1, ymm_a1);
				ymm_b2 = _mm256_mul_ps(ymm_b2, ymm_a2);
				ymm_b3 = _mm256_mul_ps(ymm_b3, ymm_a3);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				_mm256_storeu_ps(b + 8, ymm_b1);
				_mm256_storeu_ps(b + 16, ymm_b2);
				_mm256_storeu_ps(b + 24, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(a);
				ymm_b0 = _mm256_loadu_ps(b);
				// b = b + a;
				ymm_b0 = _mm256_mul_ps(ymm_b0, ymm_a0);
				// store data into memory
				_mm256_storeu_ps(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] *= a[i];
		}
	};

	template<>
	struct kernel_mul<double, cpu_avx>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 16;
			constexpr size_t bit = 4;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;

			while (n > block)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				ymm_a1 = _mm256_loadu_pd(a + 4);
				ymm_a2 = _mm256_loadu_pd(a + 8);
				ymm_a3 = _mm256_loadu_pd(a + 12);
				ymm_b0 = _mm256_loadu_pd(b);
				ymm_b1 = _mm256_loadu_pd(b + 4);
				ymm_b2 = _mm256_loadu_pd(b + 8);
				ymm_b3 = _mm256_loadu_pd(b + 12);
				// b = b + a;
				ymm_b0 = _mm256_mul_pd(ymm_b0, ymm_a0);
				ymm_b1 = _mm256_mul_pd(ymm_b1, ymm_a1);
				ymm_b2 = _mm256_mul_pd(ymm_b2, ymm_a2);
				ymm_b3 = _mm256_mul_pd(ymm_b3, ymm_a3);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				_mm256_storeu_pd(b + 4, ymm_b1);
				_mm256_storeu_pd(b + 8, ymm_b2);
				_mm256_storeu_pd(b + 12, ymm_b3);
				a += block;
				b += block;
				n -= block;
			}
			while (n > bit)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(a);
				ymm_b0 = _mm256_loadu_pd(b);
				// b = b + a;
				ymm_b0 = _mm256_mul_pd(ymm_b0, ymm_a0);
				// store data into memory
				_mm256_storeu_pd(b, ymm_b0);
				a += bit;
				b += bit;
				n -= bit;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] *= a[i];
		}
	};

} // namespace core

#endif
