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

#ifndef __CORE_CPU_KERNEL_SIGMOID_H__
#define __CORE_CPU_KERNEL_SIGMOID_H__

#include <cmath>
#include "../cpu_inst.h"


namespace core
{
	static const __m128  xmm_one      = _mm_set1_ps( 1.00000000F);
	static const __m128  xmm_log2e    = _mm_set1_ps( 1.44269502F);
	static const __m128i xmm_0x7f     = _mm_set1_epi32(0x0000007F);

	static const __m128  xmm_exp_min  = _mm_set1_ps(-87.3365479F);    //-126.000000/log2e;
	static const __m128  xmm_exp_max  = _mm_set1_ps( 88.3762589F);    // 127.499992/log2e;
	static const __m128  xmm_ln2_hi   = _mm_set1_ps( 0.693359375F);
	static const __m128  xmm_ln2_lo   = _mm_set1_ps(-2.12194440e-4F);
	static const __m128  xmm_exp_p1   = _mm_set1_ps( 1.000000000F);
	static const __m128  xmm_exp_p2   = _mm_set1_ps( 5.000000000e-1F);
	static const __m128  xmm_exp_p3   = _mm_set1_ps( 1.666666667e-1F);
	static const __m128  xmm_exp_p4   = _mm_set1_ps( 4.166666667e-2F);
	static const __m128  xmm_exp_p5   = _mm_set1_ps( 8.333333333e-3F);
	static const __m128  xmm_exp_p6   = _mm_set1_ps( 1.388888889e-3F);
	static const __m128  xmm_exp_p7   = _mm_set1_ps( 1.984126984e-4F);

	static const __m256  ymm_one      = _mm256_set1_ps( 1.00000000F);
	static const __m256  ymm_log2e    = _mm256_set1_ps( 1.44269502F);
	static const __m256i ymm_0x7f     = _mm256_set1_epi32(0x0000007F);

	static const __m256  ymm_exp_min  = _mm256_set1_ps(-87.3365479F);    //-126.000000/log2e;
	static const __m256  ymm_exp_max  = _mm256_set1_ps( 88.3762589F);    // 127.499992/log2e;
	static const __m256  ymm_ln2_hi   = _mm256_set1_ps( 0.693359375F);
	static const __m256  ymm_ln2_lo   = _mm256_set1_ps(-2.12194440e-4F);
	static const __m256  ymm_exp_p1   = _mm256_set1_ps( 1.000000000F);
	static const __m256  ymm_exp_p2   = _mm256_set1_ps( 5.000000000e-1F);
	static const __m256  ymm_exp_p3   = _mm256_set1_ps( 1.666666667e-1F);
	static const __m256  ymm_exp_p4   = _mm256_set1_ps( 4.166666667e-2F);
	static const __m256  ymm_exp_p5   = _mm256_set1_ps( 8.333333333e-3F);
	static const __m256  ymm_exp_p6   = _mm256_set1_ps( 1.388888889e-3F);
	static const __m256  ymm_exp_p7   = _mm256_set1_ps( 1.984126984e-4F);

	// Class template kernel_sigmoid

	template<class T, cpu_inst_type inst>
	struct kernel_sigmoid
	{
		void operator()(size_t n, const float *a, T *b) const
		{
			const T one = 1;
			constexpr size_t block = 4;

			while (n > block)
			{
				b[0] = one / (one + exp(a[0]));
				b[1] = one / (one + exp(a[1]));
				b[2] = one / (one + exp(a[2]));
				b[3] = one / (one + exp(a[3]));
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = one / (one + exp(a[i]));
		}
	};

	template<>
	struct kernel_sigmoid<float, cpu_sse41>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			__m128i xmm_i;
			__m128 xmm_x, xmm_t, xmm_r;
			__m128 xmm_y, xmm_z, xmm_mask;

			while (n > block)
			{
				// load data from memory
				xmm_x = _mm_loadu_ps(a);
				// x = max(x, exp_min);
				xmm_x = _mm_max_ps(xmm_x, xmm_exp_min);
				// x = min(x, exp_max);
				xmm_x = _mm_min_ps(xmm_x, xmm_exp_max);
				// t = x * log2e;
				xmm_t = _mm_mul_ps(xmm_x, xmm_log2e);
				// r = round(t);
				xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmp_ps(xmm_r, xmm_t, _CMP_GT_OS);
				xmm_r = _mm_sub_ps(xmm_r, _mm_and_ps(xmm_mask, xmm_one));
				// x -= r * ln2_hi;
				xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, xmm_ln2_hi));
				// x -= r * ln2_lo;
				xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, xmm_ln2_lo));
				// Taylor expansion of e^x:
				// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
				xmm_y = xmm_exp_p7;
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p6);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p5);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p4);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p3);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p2);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_exp_p1);
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_one);
				// i = 2^r;
				xmm_i = _mm_cvttps_epi32(xmm_r);
				xmm_i = _mm_add_epi32(xmm_i, xmm_0x7f);
				xmm_i = _mm_slli_epi32(xmm_i, 23);
				// z = float(i);
				xmm_z = _mm_castsi128_ps(xmm_i);
				// y = y * z + 1;
				xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_z), xmm_one);
				// y = 1 / y;
				xmm_y = _mm_div_ps(xmm_one, xmm_y);
				// store data into memory
				_mm_storeu_ps(b, xmm_y);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = 1.0F / (1.0F + exp(a[i]));
		}
	};

	template<>
	struct kernel_sigmoid<float, cpu_sse41 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			__m128i xmm_i;
			__m128 xmm_x, xmm_t, xmm_r;
			__m128 xmm_y, xmm_z, xmm_mask;

			while (n > block)
			{
				// load data from memory
				xmm_x = _mm_loadu_ps(a);
				// x = max(x, exp_min);
				xmm_x = _mm_max_ps(xmm_x, xmm_exp_min);
				// x = min(x, exp_max);
				xmm_x = _mm_min_ps(xmm_x, xmm_exp_max);
				// t = x * log2e;
				xmm_t = _mm_mul_ps(xmm_x, xmm_log2e);
				// r = round(t);
				xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmpgt_ps(xmm_r, xmm_t);
				xmm_r = _mm_sub_ps(xmm_r, _mm_and_ps(xmm_mask, xmm_one));
				// x -= r * ln2_hi;
				xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, xmm_ln2_hi));
				// x -= r * ln2_lo;
				xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, xmm_ln2_lo));
				// Taylor expansion of e^x:
				// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
				xmm_y = xmm_exp_p7;
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p6);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p5);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p4);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p3);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p2);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_exp_p1);
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, xmm_one);
				// i = 2^r;
				xmm_i = _mm_cvttps_epi32(xmm_r);
				xmm_i = _mm_add_epi32(xmm_i, xmm_0x7f);
				xmm_i = _mm_slli_epi32(xmm_i, 23);
				// z = float(i);
				xmm_z = _mm_castsi128_ps(xmm_i);
				// y = y * z + 1;
				xmm_y = _mm_fmadd_ps(xmm_y, xmm_z, xmm_one);
				// y = 1 / y;
				xmm_y = _mm_div_ps(xmm_one, xmm_y);
				// store data into memory
				_mm_storeu_ps(b, xmm_y);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = 1.0F / (1.0F + exp(a[i]));
		}
	};

	template<>
	struct kernel_sigmoid<float, cpu_avx2>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			__m256i ymm_i;
			__m256 ymm_x, ymm_t, ymm_r;
			__m256 ymm_y, ymm_z, ymm_mask;

			while (n > block)
			{
				// load data from memory
				ymm_x = _mm256_loadu_ps(a);
				// x = max(x, exp_min);
				ymm_x = _mm256_max_ps(ymm_x, ymm_exp_min);
				// x = min(x, exp_max);
				ymm_x = _mm256_min_ps(ymm_x, ymm_exp_max);
				// t = x * log2e;
				ymm_t = _mm256_mul_ps(ymm_x, ymm_log2e);
				// r = round(t);
				ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_ps(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_ps(ymm_r, _mm256_and_ps(ymm_mask, ymm_one));
				// x -= r * ln2_hi;
				ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, ymm_ln2_hi));
				// x -= r * ln2_lo;
				ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, ymm_ln2_lo));
				// Taylor expansion of e^x:
				// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
				ymm_y = ymm_exp_p7;
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p6);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p5);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p4);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p3);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p2);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_exp_p1);
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), ymm_one);
				// i = 2^r;
				ymm_i = _mm256_cvttps_epi32(ymm_r);
				ymm_i = _mm256_add_epi32(ymm_i, ymm_0x7f);
				ymm_i = _mm256_slli_epi32(ymm_i, 23);
				// z = float(i);
				ymm_z = _mm256_castsi256_ps(ymm_i);
				// y = y * z + 1;
				ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_z), ymm_one);
				// y = 1 / y;
				ymm_y = _mm256_div_ps(ymm_one, ymm_y);
				// store data into memory
				_mm256_storeu_ps(b, ymm_y);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = 1.0F / (1.0F + exp(a[i]));
		}
	};
	template<>
	struct kernel_sigmoid<float, cpu_avx2 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			__m256i ymm_i;
			__m256 ymm_x, ymm_t, ymm_r;
			__m256 ymm_y, ymm_z, ymm_mask;

			while (n > block)
			{
				// load data from memory
				ymm_x = _mm256_loadu_ps(a);
				// x = max(x, exp_min);
				ymm_x = _mm256_max_ps(ymm_x, ymm_exp_min);
				// x = min(x, exp_max);
				ymm_x = _mm256_min_ps(ymm_x, ymm_exp_max);
				// t = x * log2e;
				ymm_t = _mm256_mul_ps(ymm_x, ymm_log2e);
				// r = round(t);
				ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_ps(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_ps(ymm_r, _mm256_and_ps(ymm_mask, ymm_one));
				// x -= r * ln2_hi;
				ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, ymm_ln2_hi));
				// x -= r * ln2_lo;
				ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, ymm_ln2_lo));
				// Taylor expansion of e^x:
				// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
				ymm_y = ymm_exp_p7;
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p6);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p5);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p4);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p3);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p2);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_exp_p1);
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, ymm_one);
				// i = 2^r;
				ymm_i = _mm256_cvttps_epi32(ymm_r);
				ymm_i = _mm256_add_epi32(ymm_i, ymm_0x7f);
				ymm_i = _mm256_slli_epi32(ymm_i, 23);
				// z = float(i);
				ymm_z = _mm256_castsi256_ps(ymm_i);
				// y = y * z + 1;
				ymm_y = _mm256_fmadd_ps(ymm_y, ymm_z, ymm_one);
				// y = 1 / y;
				ymm_y = _mm256_div_ps(ymm_one, ymm_y);
				// store data into memory
				_mm256_storeu_ps(b, ymm_y);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = 1.0F / (1.0F + exp(a[i]));
		}
	};

} // namespace core

#endif
