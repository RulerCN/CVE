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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_MATH_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_MATH_H__

#include "../../cpu_inst.h"

namespace core
{
	static constexpr float  flt_exp_min = -87.3365479F;              //-126.000000/log2e
	static constexpr float  flt_exp_max =  88.3762589F;              // 127.499992/log2e
	static constexpr float  flt_ln2_hi  =  0.693359375F;
	static constexpr float  flt_ln2_lo  = -2.12194440e-4F;
	static constexpr double dbl_exp_min = -708.39641853226431;       //-1022.0000000000000/log2e
	static constexpr double dbl_exp_max =  709.43613930310414;       // 1023.4999999999999/log2e
	static constexpr double dbl_ln2_hi  =  0.693145751953125;
	static constexpr double dbl_ln2_lo  =  1.428606820309417e-6;
	// float sse
	static const __m128i xmm_flt_sign   = _mm_set1_epi32(core::flt_sign);
	static const __m128i xmm_flt_base   = _mm_set1_epi32(core::flt_base);
	static const __m128  xmm_onef       = _mm_set1_ps(core::flt_one);
	static const __m128  xmm_log2ef     = _mm_set1_ps(core::flt_log2e);
	static const __m128  xmm_expf_min   = _mm_set1_ps(flt_exp_min);
	static const __m128  xmm_expf_max   = _mm_set1_ps(flt_exp_max);
	static const __m128  xmm_ln2f_hi    = _mm_set1_ps(flt_ln2_hi);
	static const __m128  xmm_ln2f_lo    = _mm_set1_ps(flt_ln2_lo);
	static const __m128  xmm_expf_p1    = _mm_set1_ps(core::flt_rcp_fact1);
	static const __m128  xmm_expf_p2    = _mm_set1_ps(core::flt_rcp_fact2);
	static const __m128  xmm_expf_p3    = _mm_set1_ps(core::flt_rcp_fact3);
	static const __m128  xmm_expf_p4    = _mm_set1_ps(core::flt_rcp_fact4);
	static const __m128  xmm_expf_p5    = _mm_set1_ps(core::flt_rcp_fact5);
	static const __m128  xmm_expf_p6    = _mm_set1_ps(core::flt_rcp_fact6);
	static const __m128  xmm_expf_p7    = _mm_set1_ps(core::flt_rcp_fact7);
	// double sse
	static const __m128i xmm_dbl_sign   = _mm_set1_epi64x(core::dbl_sign);
	static const __m128i xmm_dbl_base   = _mm_set1_epi64x(core::dbl_base);
	static const __m128d xmm_oned       = _mm_set1_pd(core::dbl_one);
	static const __m128d xmm_log2ed     = _mm_set1_pd(core::dbl_log2e);
	static const __m128d xmm_expd_min   = _mm_set1_pd(dbl_exp_min);
	static const __m128d xmm_expd_max   = _mm_set1_pd(dbl_exp_max);
	static const __m128d xmm_ln2d_hi    = _mm_set1_pd(dbl_ln2_hi);
	static const __m128d xmm_ln2d_lo    = _mm_set1_pd(dbl_ln2_lo);
	static const __m128d xmm_expd_p1    = _mm_set1_pd(core::dbl_rcp_fact1);
	static const __m128d xmm_expd_p2    = _mm_set1_pd(core::dbl_rcp_fact2);
	static const __m128d xmm_expd_p3    = _mm_set1_pd(core::dbl_rcp_fact3);
	static const __m128d xmm_expd_p4    = _mm_set1_pd(core::dbl_rcp_fact4);
	static const __m128d xmm_expd_p5    = _mm_set1_pd(core::dbl_rcp_fact5);
	static const __m128d xmm_expd_p6    = _mm_set1_pd(core::dbl_rcp_fact6);
	static const __m128d xmm_expd_p7    = _mm_set1_pd(core::dbl_rcp_fact7);
	static const __m128d xmm_expd_p8    = _mm_set1_pd(core::dbl_rcp_fact8);
	static const __m128d xmm_expd_p9    = _mm_set1_pd(core::dbl_rcp_fact9);
	static const __m128d xmm_expd_p10   = _mm_set1_pd(core::dbl_rcp_fact10);
	static const __m128d xmm_expd_p11   = _mm_set1_pd(core::dbl_rcp_fact11);
	static const __m128d xmm_expd_p12   = _mm_set1_pd(core::dbl_rcp_fact12);
	static const __m128d xmm_expd_p13   = _mm_set1_pd(core::dbl_rcp_fact13);
	// float avx
	static const __m256i ymm_flt_sign   = _mm256_set1_epi32(core::flt_sign);
	static const __m256i ymm_flt_base   = _mm256_set1_epi32(core::flt_base);
	static const __m256  ymm_onef       = _mm256_set1_ps(core::flt_one);
	static const __m256  ymm_log2ef     = _mm256_set1_ps(core::flt_log2e);
	static const __m256  ymm_expf_min   = _mm256_set1_ps(flt_exp_min);
	static const __m256  ymm_expf_max   = _mm256_set1_ps(flt_exp_max);
	static const __m256  ymm_ln2f_hi    = _mm256_set1_ps(flt_ln2_hi);
	static const __m256  ymm_ln2f_lo    = _mm256_set1_ps(flt_ln2_lo);
	static const __m256  ymm_expf_p1    = _mm256_set1_ps(core::flt_rcp_fact1);
	static const __m256  ymm_expf_p2    = _mm256_set1_ps(core::flt_rcp_fact2);
	static const __m256  ymm_expf_p3    = _mm256_set1_ps(core::flt_rcp_fact3);
	static const __m256  ymm_expf_p4    = _mm256_set1_ps(core::flt_rcp_fact4);
	static const __m256  ymm_expf_p5    = _mm256_set1_ps(core::flt_rcp_fact5);
	static const __m256  ymm_expf_p6    = _mm256_set1_ps(core::flt_rcp_fact6);
	static const __m256  ymm_expf_p7    = _mm256_set1_ps(core::flt_rcp_fact7);
	// double avx
	static const __m256i ymm_dbl_sign   = _mm256_set1_epi64x(core::dbl_sign);
	static const __m256i ymm_dbl_base   = _mm256_set1_epi64x(core::dbl_base);
	static const __m256d ymm_oned       = _mm256_set1_pd(core::dbl_one);
	static const __m256d ymm_log2ed     = _mm256_set1_pd(core::dbl_log2e);
	static const __m256d ymm_expd_min   = _mm256_set1_pd(dbl_exp_min);
	static const __m256d ymm_expd_max   = _mm256_set1_pd(dbl_exp_max);
	static const __m256d ymm_ln2d_hi    = _mm256_set1_pd(dbl_ln2_hi);
	static const __m256d ymm_ln2d_lo    = _mm256_set1_pd(dbl_ln2_lo);
	static const __m256d ymm_expd_p1    = _mm256_set1_pd(core::dbl_rcp_fact1);
	static const __m256d ymm_expd_p2    = _mm256_set1_pd(core::dbl_rcp_fact2);
	static const __m256d ymm_expd_p3    = _mm256_set1_pd(core::dbl_rcp_fact3);
	static const __m256d ymm_expd_p4    = _mm256_set1_pd(core::dbl_rcp_fact4);
	static const __m256d ymm_expd_p5    = _mm256_set1_pd(core::dbl_rcp_fact5);
	static const __m256d ymm_expd_p6    = _mm256_set1_pd(core::dbl_rcp_fact6);
	static const __m256d ymm_expd_p7    = _mm256_set1_pd(core::dbl_rcp_fact7);
	static const __m256d ymm_expd_p8    = _mm256_set1_pd(core::dbl_rcp_fact8);
	static const __m256d ymm_expd_p9    = _mm256_set1_pd(core::dbl_rcp_fact9);
	static const __m256d ymm_expd_p10   = _mm256_set1_pd(core::dbl_rcp_fact10);
	static const __m256d ymm_expd_p11   = _mm256_set1_pd(core::dbl_rcp_fact11);
	static const __m256d ymm_expd_p12   = _mm256_set1_pd(core::dbl_rcp_fact12);
	static const __m256d ymm_expd_p13   = _mm256_set1_pd(core::dbl_rcp_fact13);

	// Exponential function

	__m128 exp_sse41(__m128 x)
	{
		// x = max(x, min);
		x = _mm_max_ps(x, xmm_expf_min);
		// x = min(x, max);
		x = _mm_min_ps(x, xmm_expf_max);
		// t = x * log2(e);
		__m128 t = _mm_mul_ps(x, xmm_log2ef);
		// r = round(t);
		__m128 r = _mm_round_ps(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m128 mask = _mm_cmp_ps(r, t, _CMP_GT_OS);
		r = _mm_sub_ps(r, _mm_and_ps(mask, xmm_onef));
		// x -= r * ln2_hi;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2f_hi));
		// x -= r * ln2_lo;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2f_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m128 y = xmm_expf_p7;
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p6);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p5);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p4);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p3);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p2);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_expf_p1);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_onef);
		// i = 2^r;
		__m128i i = _mm_cvttps_epi32(r);
		i = _mm_add_epi32(i, xmm_flt_base);
		i = _mm_slli_epi32(i, 23);
		// y *= (float) i;
		y = _mm_mul_ps(y, _mm_castsi128_ps(i));
		return y;
	}

	__m128 exp_sse41_fma(__m128 x)
	{
		// x = max(x, min);
		x = _mm_max_ps(x, xmm_expf_min);
		// x = min(x, max);
		x = _mm_min_ps(x, xmm_expf_max);
		// t = x * log2(e);
		__m128 t = _mm_mul_ps(x, xmm_log2ef);
		// r = round(t);
		__m128 r = _mm_round_ps(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m128 mask = _mm_cmp_ps(r, t, _CMP_GT_OS);
		r = _mm_sub_ps(r, _mm_and_ps(mask, xmm_onef));
		// x -= r * ln2_hi;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2f_hi));
		// x -= r * ln2_lo;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2f_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m128 y = xmm_expf_p7;
		y = _mm_fmadd_ps(y, x, xmm_expf_p6);
		y = _mm_fmadd_ps(y, x, xmm_expf_p5);
		y = _mm_fmadd_ps(y, x, xmm_expf_p4);
		y = _mm_fmadd_ps(y, x, xmm_expf_p3);
		y = _mm_fmadd_ps(y, x, xmm_expf_p2);
		y = _mm_fmadd_ps(y, x, xmm_expf_p1);
		y = _mm_fmadd_ps(y, x, xmm_onef);
		// i = 2^r;
		__m128i i = _mm_cvttps_epi32(r);
		i = _mm_add_epi32(i, xmm_flt_base);
		i = _mm_slli_epi32(i, 23);
		// y *= (float) i;
		y = _mm_mul_ps(y, _mm_castsi128_ps(i));
		return y;
	}

	__m128d exp_sse41(__m128d x)
	{
		// x = max(x, min);
		x = _mm_max_pd(x, xmm_expd_min);
		// x = min(x, max);
		x = _mm_min_pd(x, xmm_expd_max);
		// t = x * log2(e)
		__m128d t = _mm_mul_pd(x, xmm_log2ed);
		// r = round(t)
		__m128d r = _mm_round_pd(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m128d mask = _mm_cmp_pd(r, t, _CMP_GT_OS);
		r = _mm_sub_pd(r, _mm_and_pd(mask, xmm_oned));
		// x -= r * ln2_hi
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmm_ln2d_hi));
		// x -= r * ln2_lo
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmm_ln2d_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		//   + x^8/8! + x^9/9! + x^10/10! + x^11/11! + + x^12/12! + + x^13/13!
		__m128d y = xmm_expd_p13;
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p12);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p11);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p10);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p9);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p8);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p7);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p6);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p5);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p4);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p3);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p2);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_expd_p1);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmm_oned);
		// i = 2^r
		__m128i i = _mm_cvtepi32_epi64(_mm_cvttpd_epi32(r));
		i = _mm_add_epi64(i, xmm_dbl_base);
		i = _mm_slli_epi64(i, 52);
		// y *= (double) i;
		y = _mm_mul_pd(y, _mm_castsi128_pd(i));
		return y;
	}

	__m128d exp_sse41_fma(__m128d x)
	{
		// x = max(x, min);
		x = _mm_max_pd(x, xmm_expd_min);
		// x = min(x, max);
		x = _mm_min_pd(x, xmm_expd_max);
		// t = x * log2(e)
		__m128d t = _mm_mul_pd(x, xmm_log2ed);
		// r = round(t)
		__m128d r = _mm_round_pd(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m128d mask = _mm_cmp_pd(r, t, _CMP_GT_OS);
		r = _mm_sub_pd(r, _mm_and_pd(mask, xmm_oned));
		// x -= r * ln2_hi
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmm_ln2d_hi));
		// x -= r * ln2_lo
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmm_ln2d_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		//   + x^8/8! + x^9/9! + x^10/10! + x^11/11! + + x^12/12! + + x^13/13!
		__m128d y = xmm_expd_p13;
		y = _mm_fmadd_pd(y, x, xmm_expd_p12);
		y = _mm_fmadd_pd(y, x, xmm_expd_p11);
		y = _mm_fmadd_pd(y, x, xmm_expd_p10);
		y = _mm_fmadd_pd(y, x, xmm_expd_p9);
		y = _mm_fmadd_pd(y, x, xmm_expd_p8);
		y = _mm_fmadd_pd(y, x, xmm_expd_p7);
		y = _mm_fmadd_pd(y, x, xmm_expd_p6);
		y = _mm_fmadd_pd(y, x, xmm_expd_p5);
		y = _mm_fmadd_pd(y, x, xmm_expd_p4);
		y = _mm_fmadd_pd(y, x, xmm_expd_p3);
		y = _mm_fmadd_pd(y, x, xmm_expd_p2);
		y = _mm_fmadd_pd(y, x, xmm_expd_p1);
		y = _mm_fmadd_pd(y, x, xmm_oned);
		// i = 2^r
		__m128i i = _mm_cvtepi32_epi64(_mm_cvttpd_epi32(r));
		i = _mm_add_epi64(i, xmm_dbl_base);
		i = _mm_slli_epi64(i, 52);
		// y *= (double) i;
		y = _mm_mul_pd(y, _mm_castsi128_pd(i));
		return y;
	}

	__m256 exp_avx2(__m256 x)
	{
		// x = max(x, min);
		x = _mm256_max_ps(x, ymm_expf_min);
		// x = min(x, max);
		x = _mm256_min_ps(x, ymm_expf_max);
		// t = x * log2(e);
		__m256 t = _mm256_mul_ps(x, ymm_log2ef);
		// r = round(t);
		__m256 r = _mm256_round_ps(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m256 mask = _mm256_cmp_ps(r, t, _CMP_GT_OS);
		r = _mm256_sub_ps(r, _mm256_and_ps(mask, ymm_onef));
		// x -= r * ln2_hi;
		x = _mm256_sub_ps(x, _mm256_mul_ps(r, ymm_ln2f_hi));
		// x -= r * ln2_lo;
		x = _mm256_sub_ps(x, _mm256_mul_ps(r, ymm_ln2f_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m256 y = ymm_expf_p7;
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p6);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p5);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p4);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p3);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p2);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_expf_p1);
		y = _mm256_add_ps(_mm256_mul_ps(y, x), ymm_onef);
		// i = 2^r;
		__m256i i = _mm256_cvttps_epi32(r);
		i = _mm256_add_epi32(i, ymm_flt_base);
		i = _mm256_slli_epi32(i, 23);
		// y *= (float) i;
		y = _mm256_mul_ps(y, _mm256_castsi256_ps(i));
		return y;
	}

	__m256 exp_avx2_fma(__m256 x)
	{
		// x = max(x, min);
		x = _mm256_max_ps(x, ymm_expf_min);
		// x = min(x, max);
		x = _mm256_min_ps(x, ymm_expf_max);
		// t = x * log2(e);
		__m256 t = _mm256_mul_ps(x, ymm_log2ef);
		// r = round(t);
		__m256 r = _mm256_round_ps(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m256 mask = _mm256_cmp_ps(r, t, _CMP_GT_OS);
		r = _mm256_sub_ps(r, _mm256_and_ps(mask, ymm_onef));
		// x -= r * ln2_hi;
		x = _mm256_sub_ps(x, _mm256_mul_ps(r, ymm_ln2f_hi));
		// x -= r * ln2_lo;
		x = _mm256_sub_ps(x, _mm256_mul_ps(r, ymm_ln2f_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m256 y = ymm_expf_p7;
		y = _mm256_fmadd_ps(y, x, ymm_expf_p6);
		y = _mm256_fmadd_ps(y, x, ymm_expf_p5);
		y = _mm256_fmadd_ps(y, x, ymm_expf_p4);
		y = _mm256_fmadd_ps(y, x, ymm_expf_p3);
		y = _mm256_fmadd_ps(y, x, ymm_expf_p2);
		y = _mm256_fmadd_ps(y, x, ymm_expf_p1);
		y = _mm256_fmadd_ps(y, x, ymm_onef);
		// i = 2^r;
		__m256i i = _mm256_cvttps_epi32(r);
		i = _mm256_add_epi32(i, ymm_flt_base);
		i = _mm256_slli_epi32(i, 23);
		// y *= (float) i;
		y = _mm256_mul_ps(y, _mm256_castsi256_ps(i));
		return y;
	}

	__m256d exp_avx2(__m256d x)
	{
		// x = max(x, min);
		x = _mm256_max_pd(x, ymm_expd_min);
		// x = min(x, max);
		x = _mm256_min_pd(x, ymm_expd_max);
		// t = x * log2(e);
		__m256d t = _mm256_mul_pd(x, ymm_log2ed);
		// r = round(t);
		__m256d r = _mm256_round_pd(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m256d mask = _mm256_cmp_pd(r, t, _CMP_GT_OS);
		r = _mm256_sub_pd(r, _mm256_and_pd(mask, ymm_oned));
		// x -= r * ln2_hi;
		x = _mm256_sub_pd(x, _mm256_mul_pd(r, ymm_ln2d_hi));
		// x -= r * ln2_lo;
		x = _mm256_sub_pd(x, _mm256_mul_pd(r, ymm_ln2d_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		//   + x^8/8! + x^9/9! + x^10/10! + x^11/11! + + x^12/12! + + x^13/13!
		__m256d y = ymm_expd_p13;
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p12);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p11);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p10);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p9);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p8);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p7);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p6);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p5);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p4);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p3);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p2);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_expd_p1);
		y = _mm256_add_pd(_mm256_mul_pd(y, x), ymm_oned);
		// i = 2^r;
		__m256i i = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(r));
		i = _mm256_add_epi64(i, ymm_dbl_base);
		i = _mm256_slli_epi64(i, 52);
		// y *= (double) i;
		y = _mm256_mul_pd(y, _mm256_castsi256_pd(i));
		return y;
	}

	__m256d exp_avx2_fma(__m256d x)
	{
		// x = max(x, min);
		x = _mm256_max_pd(x, ymm_expd_min);
		// x = min(x, max);
		x = _mm256_min_pd(x, ymm_expd_max);
		// t = x * log2(e);
		__m256d t = _mm256_mul_pd(x, ymm_log2ed);
		// r = round(t);
		__m256d r = _mm256_round_pd(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m256d mask = _mm256_cmp_pd(r, t, _CMP_GT_OS);
		r = _mm256_sub_pd(r, _mm256_and_pd(mask, ymm_oned));
		// x -= r * ln2_hi;
		x = _mm256_sub_pd(x, _mm256_mul_pd(r, ymm_ln2d_hi));
		// x -= r * ln2_lo;
		x = _mm256_sub_pd(x, _mm256_mul_pd(r, ymm_ln2d_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		//   + x^8/8! + x^9/9! + x^10/10! + x^11/11! + + x^12/12! + + x^13/13!
		__m256d y = ymm_expd_p13;
		y = _mm256_fmadd_pd(y, x, ymm_expd_p12);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p11);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p10);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p9);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p8);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p7);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p6);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p5);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p4);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p3);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p2);
		y = _mm256_fmadd_pd(y, x, ymm_expd_p1);
		y = _mm256_fmadd_pd(y, x, ymm_oned);
		// i = 2^r;
		__m256i i = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(r));
		i = _mm256_add_epi64(i, ymm_dbl_base);
		i = _mm256_slli_epi64(i, 52);
		// y *= (double) i;
		y = _mm256_mul_pd(y, _mm256_castsi256_pd(i));
		return y;
	}

/*
	__m128 _mm_log_ps(__m128 x)
	{
		__m128 invalid_mask = _mm_cmp_ps(x, xmm_zero, _CMP_LE_OS);
		x = _mm_max_ps(x, _mm_castsi128_ps(xmm_min_norm_pos));

		__m128i imm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
		imm0 = _mm_sub_epi32(imm0, xmm_0x7f);
		__m128 e = _mm_cvtepi32_ps(imm0);
		e = _mm_add_ps(e, xmm_one);

		x = _mm_and_ps(x, _mm_castsi128_ps(xmm_inv_mant_mask));
		x = _mm_or_ps(x, xmm_0p5);

		__m128 mask = _mm_cmp_ps(x, xmm_sqrthf, _CMP_LT_OS);
		__m128 tmp = _mm_and_ps(x, mask);
		x = _mm_sub_ps(x, xmm_one);
		e = _mm_sub_ps(e, _mm_and_ps(xmm_one, mask));
		x = _mm_add_ps(x, tmp);

		__m128 z = _mm_mul_ps(x, x);
		__m128 y = xmm_log_p0;
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p1);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p2);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p3);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p4);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p5);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p6);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p7);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_log_p8);
		y = _mm_mul_ps(y, x);
		y = _mm_mul_ps(y, z);

		tmp = _mm_mul_ps(e, xmm_log_q1);
		y = _mm_add_ps(y, tmp);
		tmp = _mm_mul_ps(z, xmm_0p5);
		y = _mm_sub_ps(y, tmp);
		tmp = _mm_mul_ps(e, xmm_log_q2);
		x = _mm_add_ps(x, y);
		x = _mm_add_ps(x, tmp);
		x = _mm_or_ps(x, invalid_mask);
		return x;
	}

	__m256 _mm256_log_ps(__m256 x)
	{
		__m256 invalid_mask = _mm256_cmp_ps(x, ymm_zero, _CMP_LE_OS);
		x = _mm256_max_ps(x, _mm256_castsi256_ps(ymm_min_norm_pos));

		__m256i imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
		imm0 = _mm256_sub_epi32(imm0, ymm_0x7f);
		__m256 e = _mm256_cvtepi32_ps(imm0);
		e = _mm256_add_ps(e, ymm_one);

		x = _mm256_and_ps(x, _mm256_castsi256_ps(ymm_inv_mant_mask));
		x = _mm256_or_ps(x, ymm_0p5);

		__m256 mask = _mm256_cmp_ps(x, ymm_sqrthf, _CMP_LT_OS);
		__m256 tmp = _mm256_and_ps(x, mask);
		x = _mm256_sub_ps(x, ymm_one);
		e = _mm256_sub_ps(e, _mm256_and_ps(ymm_one, mask));
		x = _mm256_add_ps(x, tmp);

		__m256 z = _mm256_mul_ps(x, x);
		__m256 y = ymm_log_p0;
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p1);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p2);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p3);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p4);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p5);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p6);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p7);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_log_p8);
		y = _mm256_mul_ps(y, x);
		y = _mm256_mul_ps(y, z);

		tmp = _mm256_mul_ps(e, ymm_log_q1);
		y = _mm256_add_ps(y, tmp);
		tmp = _mm256_mul_ps(z, ymm_0p5);
		y = _mm256_sub_ps(y, tmp);
		tmp = _mm256_mul_ps(e, ymm_log_q2);
		x = _mm256_add_ps(x, y);
		x = _mm256_add_ps(x, tmp);
		x = _mm256_or_ps(x, invalid_mask);
		return x;
	}
*/
} // namespace core

#endif
