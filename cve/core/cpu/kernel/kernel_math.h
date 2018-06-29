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

#ifndef __CORE_CPU_KERNEL_MATH_H__
#define __CORE_CPU_KERNEL_MATH_H__

#include "../cpu_inst.h"

namespace core
{
	static const __m128  xmm_one      = _mm_set1_ps( 1.00000000F);
	static const __m128  xmm_log2e    = _mm_set1_ps( 1.44269502F);
	static const __m128i xmm_0x7f     = _mm_set1_epi32(0x0000007F);

	static const __m128  xmm_exp_min  = _mm_set1_ps(-87.3365479F);    //-126.000000/log2e;
	static const __m128  xmm_exp_max  = _mm_set1_ps( 88.3762589F);    // 127.499992/log2e;
	static const __m128  xmm_ln2_hi   = _mm_set1_ps( 0.693359375F);
	static const __m128  xmm_ln2_lo   = _mm_set1_ps(-2.12194442e-4F);
	static const __m128  xmm_exp_p1   = _mm_set1_ps( 1.000000000F);
	static const __m128  xmm_exp_p2   = _mm_set1_ps( 5.000000000e-1F);
	static const __m128  xmm_exp_p3   = _mm_set1_ps( 1.666666667e-1F);
	static const __m128  xmm_exp_p4   = _mm_set1_ps( 4.166666667e-2F);
	static const __m128  xmm_exp_p5   = _mm_set1_ps( 8.333333333e-3F);
	static const __m128  xmm_exp_p6   = _mm_set1_ps( 1.388888889e-3F);
	static const __m128  xmm_exp_p7   = _mm_set1_ps( 1.984126984e-4F);

	static const __m128d xmmd_one     = _mm_set1_pd( 1.000000000000000);
	static const __m128d xmmd_log2e   = _mm_set1_pd( 1.442695040888963);
	static const __m128i xmmd_0x3ff   = _mm_set1_epi64x(0x00000000000003FF);

	static const __m128d xmmd_exp_hi  = _mm_set1_pd( 709.78271289338397);
	static const __m128d xmmd_exp_lo  = _mm_set1_pd(-709.78271289338397);
	static const __m128d xmmd_ln2_hi  = _mm_set1_pd( 0.693145751953125);
	static const __m128d xmmd_ln2_lo  = _mm_set1_pd( 1.428606820309417e-6);
	static const __m128d xmmd_exp_p1  = _mm_set1_pd( 1.000000000000000);
	static const __m128d xmmd_exp_p2  = _mm_set1_pd( 5.000000000000000e-1);
	static const __m128d xmmd_exp_p3  = _mm_set1_pd( 1.666666666666667e-1);
	static const __m128d xmmd_exp_p4  = _mm_set1_pd( 4.166666666666667e-2);
	static const __m128d xmmd_exp_p5  = _mm_set1_pd( 8.333333333333333e-3);
	static const __m128d xmmd_exp_p6  = _mm_set1_pd( 1.388888888888889e-3);
	static const __m128d xmmd_exp_p7  = _mm_set1_pd( 1.984126984126984e-4);
	static const __m128d xmmd_exp_p8  = _mm_set1_pd( 2.480158730158730e-5);
	static const __m128d xmmd_exp_p9  = _mm_set1_pd( 2.755731922398589e-6);
	static const __m128d xmmd_exp_p10 = _mm_set1_pd( 2.755731922398589e-7);
	static const __m128d xmmd_exp_p11 = _mm_set1_pd( 2.505210838544172e-8);
	static const __m128d xmmd_exp_p12 = _mm_set1_pd( 2.087675698786810e-9);
	static const __m128d xmmd_exp_p13 = _mm_set1_pd( 1.605904383682161e-10);


	// Exponential function

	__m128 _mm_exp_ps(__m128 x)
	{
		// x = max(x, min);
		x = _mm_max_ps(x, xmm_exp_min);
		// x = min(x, max);
		x = _mm_min_ps(x, xmm_exp_max);
		// t = x * log2(e);
		__m128 t = _mm_mul_ps(x, xmm_log2e);
		// r = round(t);
		__m128 r = _mm_round_ps(t, _MM_FROUND_NINT);
		// if (r > t) r -= 1;
		__m128 mask = _mm_cmpgt_ps(r, t);
		r = _mm_sub_ps(r, _mm_and_ps(mask, xmm_one));
		// x -= r * ln2_hi;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2_hi));
		// x -= r * ln2_lo;
		x = _mm_sub_ps(x, _mm_mul_ps(r, xmm_ln2_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		__m128 y = xmm_exp_p7;
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p6);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p5);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p4);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p3);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p2);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_exp_p1);
		y = _mm_add_ps(_mm_mul_ps(y, x), xmm_one);
		// i = 2^r;
		__m128i i = _mm_cvttps_epi32(r);
		i = _mm_add_epi32(i, xmm_0x7f);
		i = _mm_slli_epi32(i, 23);
		// y += i;
		y = _mm_mul_ps(y, _mm_castsi128_ps(i));
		return y;
	}

	__m128d _mm_exp_pd(__m128d x)
	{
		x = _mm_min_pd(x, xmmd_exp_hi);
		x = _mm_max_pd(x, xmmd_exp_lo);
		// t = x * log2(e)
		__m128d t = _mm_mul_pd(x, xmmd_log2e);
		// r = round(t)
		__m128d r = _mm_round_pd(t, _MM_FROUND_NINT);
		// x -= r * ln2_hi
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmmd_ln2_hi));
		// x -= r * ln2_lo
		x = _mm_sub_pd(x, _mm_mul_pd(r, xmmd_ln2_lo));
		// Taylor expansion of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!
		//   + x^8/8! + x^9/9! + x^10/10! + x^11/11! + + x^12/12! + + x^13/13!
		__m128d y = xmmd_exp_p13;
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p12);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p11);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p10);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p9);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p8);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p7);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p6);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p5);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p4);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p3);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p2);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_exp_p1);
		y = _mm_add_pd(_mm_mul_pd(y, x), xmmd_one);
		// i = 2^r
		__m128i i = _mm_cvtepi32_epi64(_mm_cvttpd_epi32(r));
		i = _mm_add_epi64(i, xmmd_0x3ff);
		i = _mm_slli_epi64(i, 52);
		// y += i
		y = _mm_mul_pd(y, _mm_castsi128_pd(i));
		return y;
	}

/*
	static const __m128i xmm_0x7f          = _mm_set1_epi32(0x0000007f);
	static const __m128i xmm_min_norm_pos  = _mm_set1_epi32(0x00800000);
	static const __m128i xmm_mant_mask     = _mm_set1_epi32(0x7f800000);
	static const __m128i xmm_inv_mant_mask = _mm_set1_epi32(~0x7f800000);
	static const __m128  xmm_zero          = _mm_setzero_ps();
	static const __m128  xmm_0p5           = _mm_set1_ps(0.5F);
	static const __m128  xmm_one           = _mm_set1_ps(1.0F);
	static const __m128  xmm_exp_hi        = _mm_set1_ps(88.3762626647949F);
	static const __m128  xmm_exp_lo        = _mm_set1_ps(-88.3762626647949F);
	static const __m128  xmm_log2ef        = _mm_set1_ps(1.44269504088896341F);
	static const __m128  xmm_exp_c1        = _mm_set1_ps(0.693359375F);
	static const __m128  xmm_exp_c2        = _mm_set1_ps(-2.12194440e-4F);
	static const __m128  xmm_exp_p0        = _mm_set1_ps(1.9875691500e-4F);
	static const __m128  xmm_exp_p1        = _mm_set1_ps(1.3981999507e-3F);
	static const __m128  xmm_exp_p2        = _mm_set1_ps(8.3334519073e-3F);
	static const __m128  xmm_exp_p3        = _mm_set1_ps(4.1665795894e-2F);
	static const __m128  xmm_exp_p4        = _mm_set1_ps(1.6666665459e-1F);
	static const __m128  xmm_exp_p5        = _mm_set1_ps(5.0000001201e-1F);
	static const __m128  xmm_sqrthf        = _mm_set1_ps(0.707106781186547524F);
	static const __m128  xmm_log_p0        = _mm_set1_ps(7.0376836292e-2F);
	static const __m128  xmm_log_p1        = _mm_set1_ps(-1.1514610310e-1F);
	static const __m128  xmm_log_p2        = _mm_set1_ps(1.1676998740e-1F);
	static const __m128  xmm_log_p3        = _mm_set1_ps(-1.2420140846e-1F);
	static const __m128  xmm_log_p4        = _mm_set1_ps(1.4249322787e-1F);
	static const __m128  xmm_log_p5        = _mm_set1_ps(-1.6668057665e-1F);
	static const __m128  xmm_log_p6        = _mm_set1_ps(2.0000714765e-1F);
	static const __m128  xmm_log_p7        = _mm_set1_ps(-2.4999993993e-1F);
	static const __m128  xmm_log_p8        = _mm_set1_ps(3.3333331174e-1F);
	static const __m128  xmm_log_q1        = _mm_set1_ps(-2.12194440e-4F);
	static const __m128  xmm_log_q2        = _mm_set1_ps(0.693359375F);

	static const __m256i ymm_0x7f          = _mm256_set1_epi32(0x0000007f);
	static const __m256i ymm_min_norm_pos  = _mm256_set1_epi32(0x00800000);
	static const __m256i ymm_mant_mask     = _mm256_set1_epi32(0x7f800000);
	static const __m256i ymm_inv_mant_mask = _mm256_set1_epi32(~0x7f800000);
	static const __m256  ymm_zero          = _mm256_setzero_ps();
	static const __m256  ymm_0p5           = _mm256_set1_ps(0.5F);
	static const __m256  ymm_one           = _mm256_set1_ps(1.0F);
	static const __m256  ymm_exp_hi        = _mm256_set1_ps(88.3762626647949F);
	static const __m256  ymm_exp_lo        = _mm256_set1_ps(-88.3762626647949F);
	static const __m256  ymm_log2ef        = _mm256_set1_ps(1.44269504088896341F);
	static const __m256  ymm_exp_c1        = _mm256_set1_ps(0.693359375F);
	static const __m256  ymm_exp_c2        = _mm256_set1_ps(-2.12194440e-4F);
	static const __m256  ymm_exp_p0        = _mm256_set1_ps(1.9875691500e-4F);
	static const __m256  ymm_exp_p1        = _mm256_set1_ps(1.3981999507e-3F);
	static const __m256  ymm_exp_p2        = _mm256_set1_ps(8.3334519073e-3F);
	static const __m256  ymm_exp_p3        = _mm256_set1_ps(4.1665795894e-2F);
	static const __m256  ymm_exp_p4        = _mm256_set1_ps(1.6666665459e-1F);
	static const __m256  ymm_exp_p5        = _mm256_set1_ps(5.0000001201e-1F);
	static const __m256  ymm_sqrthf        = _mm256_set1_ps(0.707106781186547524F);
	static const __m256  ymm_log_p0        = _mm256_set1_ps(7.0376836292e-2F);
	static const __m256  ymm_log_p1        = _mm256_set1_ps(-1.1514610310e-1F);
	static const __m256  ymm_log_p2        = _mm256_set1_ps(1.1676998740e-1F);
	static const __m256  ymm_log_p3        = _mm256_set1_ps(-1.2420140846e-1F);
	static const __m256  ymm_log_p4        = _mm256_set1_ps(1.4249322787e-1F);
	static const __m256  ymm_log_p5        = _mm256_set1_ps(-1.6668057665e-1F);
	static const __m256  ymm_log_p6        = _mm256_set1_ps(2.0000714765e-1F);
	static const __m256  ymm_log_p7        = _mm256_set1_ps(-2.4999993993e-1F);
	static const __m256  ymm_log_p8        = _mm256_set1_ps(3.3333331174e-1F);
	static const __m256  ymm_log_q1        = _mm256_set1_ps(-2.12194440e-4F);
	static const __m256  ymm_log_q2        = _mm256_set1_ps(0.693359375F);

	__m128 _mm_exp_ps(__m128 x)
	{
		x = _mm_min_ps(x, xmm_exp_hi);
		x = _mm_max_ps(x, xmm_exp_lo);

		// express exp(x) as exp(g + n*log(2))
		__m128 fx = _mm_mul_ps(x, xmm_log2ef);
		fx = _mm_add_ps(fx, xmm_0p5);

		__m128 tmp = _mm_floor_ps(fx);
		__m128 mask = _mm_cmp_ps(tmp, fx, _CMP_GT_OS);
		mask = _mm_and_ps(mask, xmm_one);
		fx = _mm_sub_ps(tmp, mask);
		tmp = _mm_mul_ps(fx, xmm_exp_c1);

		__m128 z = _mm_mul_ps(fx, xmm_exp_c2);
		x = _mm_sub_ps(x, tmp);
		x = _mm_sub_ps(x, z);
		z = _mm_mul_ps(x, x);

		__m128 y = xmm_exp_p0;
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_exp_p1);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_exp_p2);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_exp_p3);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_exp_p4);
		y = _mm_mul_ps(y, x);
		y = _mm_add_ps(y, xmm_exp_p5);
		y = _mm_mul_ps(y, z);
		y = _mm_add_ps(y, x);
		y = _mm_add_ps(y, xmm_one);

		__m128i imm0 = _mm_cvttps_epi32(fx);
		imm0 = _mm_add_epi32(imm0, xmm_0x7f);
		imm0 = _mm_slli_epi32(imm0, 23);

		__m128 pow2n = _mm_castsi128_ps(imm0);
		y = _mm_mul_ps(y, pow2n);
		return y;
	}

	__m256 _mm256_exp_ps(__m256 x)
	{
		x = _mm256_min_ps(x, ymm_exp_hi);
		x = _mm256_max_ps(x, ymm_exp_lo);

		// express exp(x) as exp(g + n*log(2))
		__m256 fx = _mm256_mul_ps(x, ymm_log2ef);
		fx = _mm256_add_ps(fx, ymm_0p5);

		__m256 tmp = _mm256_floor_ps(fx);
		__m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
		mask = _mm256_and_ps(mask, ymm_one);
		fx = _mm256_sub_ps(tmp, mask);
		tmp = _mm256_mul_ps(fx, ymm_exp_c1);

		__m256 z = _mm256_mul_ps(fx, ymm_exp_c2);
		x = _mm256_sub_ps(x, tmp);
		x = _mm256_sub_ps(x, z);
		z = _mm256_mul_ps(x, x);

		__m256 y = ymm_exp_p0;
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_exp_p1);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_exp_p2);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_exp_p3);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_exp_p4);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, ymm_exp_p5);
		y = _mm256_mul_ps(y, z);
		y = _mm256_add_ps(y, x);
		y = _mm256_add_ps(y, ymm_one);

		__m256i imm0 = _mm256_cvttps_epi32(fx);
		imm0 = _mm256_add_epi32(imm0, ymm_0x7f);
		imm0 = _mm256_slli_epi32(imm0, 23);

		__m256 pow2n = _mm256_castsi256_ps(imm0);
		y = _mm256_mul_ps(y, pow2n);
		return y;
	}

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
