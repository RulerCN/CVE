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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_MATH_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_MATH_H__

#include <cmath>
#include "../../cpu_inst.h"

namespace core
{
/*
	static constexpr float  flt_exp_min = -87.3365479F;              //-126.000000/log2e
	static constexpr float  flt_exp_max =  88.3762589F;              // 127.499992/log2e
	static constexpr float  flt_ln2_hi  =  6.93359375e-1F;           // ln2 of 11 digit mantissa
	static constexpr float  flt_ln2_lo  = -2.12194440e-4F;           // ln2 - flt_ln2_hi
	static constexpr double dbl_exp_min = -708.39641853226431;       //-1022.0000000000000/log2e
	static constexpr double dbl_exp_max =  709.43613930310414;       // 1023.4999999999999/log2e
	static constexpr double dbl_ln2_hi  =  6.9314718246459961e-1;    // ln2 of 20 digit mantissa
	static constexpr double dbl_ln2_lo  = -1.9046543005827679e-9;    // ln2 - dbl_ln2_hi

	static constexpr float  flt_log_p0 =  2.00000000000000000000e00F;
	static constexpr float  flt_log_p1 = -2.56410256410256410256e00F;
	static constexpr float  flt_log_p2 =  7.91608391608391608392e-1F;
	static constexpr float  flt_log_p3 = -3.40992340992340992341e-2F;
	static constexpr float  flt_log_q0 =  1.00000000000000000000e00F;
	static constexpr float  flt_log_q1 = -1.61538461538461538462e00F;
	static constexpr float  flt_log_q2 =  7.34265734265734265734e-1F;
	static constexpr float  flt_log_q3 = -8.15850815850815850816e-2F;

	static constexpr double dbl_log_p0 =  2.00000000000000000000e00;
	static constexpr double dbl_log_p1 = -3.56862745098039215686e00;
	static constexpr double dbl_log_p2 =  1.95294117647058823529e00;
	static constexpr double dbl_log_p3 = -3.33290239172592113769e-1;
	static constexpr double dbl_log_p4 =  8.55823914647444059209e-3;
	static constexpr double dbl_log_q0 =  1.00000000000000000000e00;
	static constexpr double dbl_log_q1 = -2.11764705882352941176e00;
	static constexpr double dbl_log_q2 =  1.48235294117647058824e00;
	static constexpr double dbl_log_q3 = -3.80090497737556561086e-1;
	static constexpr double dbl_log_q4 =  2.59152612093788564377e-2;

	// float
	static const __m128  xmm_zero       = _mm_set1_ps(core::flt_zero);
	static const __m128  xmm_half       = _mm_set1_ps(core::flt_half);
	static const __m128  xmm_one        = _mm_set1_ps(core::flt_one);
	static const __m128  xmm_two        = _mm_set1_ps(core::flt_two);
	static const __m128  xmm_sqrt2      = _mm_set1_ps(core::flt_sqrt2);
	static const __m128  xmm_ln2f_hi    = _mm_set1_ps(flt_ln2_hi);
	static const __m128  xmm_ln2f_lo    = _mm_set1_ps(flt_ln2_lo);

	static const __m128i xmm_flt_sign   = _mm_set1_epi32(core::flt_sign);
	static const __m128i xmm_flt_base   = _mm_set1_epi32(core::flt_base);
	static const __m128i xmm_flt_exp    = _mm_set1_epi32(core::flt_exp_mask);
	static const __m128i xmm_flt_mant   = _mm_set1_epi32(core::flt_mant_mask);

	static const __m128  xmm_zerof      = _mm_set1_ps(core::flt_zero);
	static const __m128  xmm_halff      = _mm_set1_ps(core::flt_half);
	static const __m128  xmm_onef       = _mm_set1_ps(core::flt_one);
	static const __m128  xmm_twof       = _mm_set1_ps(core::flt_two);
	static const __m128  xmm_log2ef     = _mm_set1_ps(core::flt_log2e);
	static const __m128  xmm_logf_min   = _mm_set1_ps(core::flt_min);
	static const __m128  xmm_logf_p1    = _mm_set1_ps(core::flt_rcp_1);
	static const __m128  xmm_logf_p2    = _mm_set1_ps(core::flt_rcp_3);
	static const __m128  xmm_logf_p3    = _mm_set1_ps(core::flt_rcp_5);
	static const __m128  xmm_logf_p4    = _mm_set1_ps(core::flt_rcp_7);
	static const __m128  xmm_logf_p5    = _mm_set1_ps(core::flt_rcp_9);
	static const __m128  xmm_logf_p6    = _mm_set1_ps(core::flt_rcp_11);
	static const __m128  xmm_logf_p7    = _mm_set1_ps(core::flt_rcp_13);
	static const __m128  xmm_expf_min   = _mm_set1_ps(flt_exp_min);
	static const __m128  xmm_expf_max   = _mm_set1_ps(flt_exp_max);
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
	static const __m256i ymm_flt_exp    = _mm256_set1_epi32(core::flt_exp_mask);
	static const __m256i ymm_flt_mant   = _mm256_set1_epi32(core::flt_mant_mask);
	static const __m256  ymm_zerof      = _mm256_set1_ps(core::flt_zero);
	static const __m256  ymm_onef       = _mm256_set1_ps(core::flt_one);
	static const __m256  ymm_twof       = _mm256_set1_ps(core::flt_two);
	static const __m256  ymm_log2ef     = _mm256_set1_ps(core::flt_log2e);
	static const __m256  ymm_ln2f_hi    = _mm256_set1_ps(flt_ln2_hi);
	static const __m256  ymm_ln2f_lo    = _mm256_set1_ps(flt_ln2_lo);
	static const __m256  ymm_logf_min   = _mm256_set1_ps(core::flt_min);
	static const __m256  ymm_logf_p1    = _mm256_set1_ps(core::flt_rcp_1);
	static const __m256  ymm_logf_p2    = _mm256_set1_ps(core::flt_rcp_3);
	static const __m256  ymm_logf_p3    = _mm256_set1_ps(core::flt_rcp_5);
	static const __m256  ymm_logf_p4    = _mm256_set1_ps(core::flt_rcp_7);
	static const __m256  ymm_logf_p5    = _mm256_set1_ps(core::flt_rcp_9);
	static const __m256  ymm_logf_p6    = _mm256_set1_ps(core::flt_rcp_11);
	static const __m256  ymm_logf_p7    = _mm256_set1_ps(core::flt_rcp_13);
	static const __m256  ymm_expf_min   = _mm256_set1_ps(flt_exp_min);
	static const __m256  ymm_expf_max   = _mm256_set1_ps(flt_exp_max);
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

	// Logarithmic function

	__m128 log_sse2(__m128 xmm_x)
	{
		__m128i xmm_i;
		__m128 xmm_neg, xmm_e, xmm_m, xmm_t, xmm_y;

		// xmm_neg = xmm_x <= xmm_zerof;
		xmm_neg = _mm_cmp_ps(xmm_x, xmm_zerof, _CMP_LE_OS);
		// xmm_x = max(xmm_x, xmm_logf_min);
		xmm_x = _mm_max_ps(xmm_x, xmm_logf_min);
		// keep the exponent part
		xmm_i = _mm_srli_epi32(_mm_castps_si128(xmm_x), 23);
		xmm_i = _mm_sub_epi32(xmm_i, xmm_flt_base);
		xmm_e = _mm_cvtepi32_ps(xmm_i);
		// keep the decimal part
		xmm_m = _mm_and_ps(xmm_x, _mm_castsi128_ps(xmm_flt_mant));
		xmm_m = _mm_or_ps(xmm_m, _mm_castsi128_ps(xmm_flt_exp));
		// xmm_t = (xmm_m - 1) / (xmm_m + 1);
		xmm_t = _mm_div_ps(_mm_sub_ps(xmm_m, xmm_onef), _mm_add_ps(xmm_m, xmm_onef));
		// xmm_x = xmm_t * xmm_t;
		xmm_x = _mm_mul_ps(xmm_t, xmm_t);
		// Maclaurin expansion of ln(x) = ln((t+1)/(t-1)):
		// y = 2t(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
		xmm_y = xmm_logf_p7;
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p6);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p5);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p4);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p3);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p2);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), xmm_logf_p1);
		xmm_y = _mm_mul_ps(xmm_y, xmm_t);
		xmm_y = _mm_mul_ps(xmm_y, xmm_twof);
		// xmm_y = xmm_y + (float) xmm_e * xmm_ln2f_hi;
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_e, xmm_ln2f_hi), xmm_y);
		// xmm_y = xmm_y + (float) xmm_e * xmm_ln2f_lo;
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_e, xmm_ln2f_lo), xmm_y);
		// negative number returns to NAN 
		xmm_y = _mm_or_ps(xmm_y, xmm_neg);
		return xmm_y;
	}

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

	_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
	_PS_CONST(cephes_log_p0, 7.0376836292E-2);    1/11
	_PS_CONST(cephes_log_p1, - 1.1514610310E-1); -1/10
	_PS_CONST(cephes_log_p2, 1.1676998740E-1);    1/9
	_PS_CONST(cephes_log_p3, - 1.2420140846E-1); -1/8
	_PS_CONST(cephes_log_p4, + 1.4249322787E-1);  1/7
	_PS_CONST(cephes_log_p5, - 1.6668057665E-1); -1/6
	_PS_CONST(cephes_log_p6, + 2.0000714765E-1);  1/5
	_PS_CONST(cephes_log_p7, - 2.4999993993E-1); -1/4
	_PS_CONST(cephes_log_p8, + 3.3333331174E-1);  1/3
	_PS_CONST(cephes_log_q1, -2.12194440e-4);
	_PS_CONST(cephes_log_q2, 0.693359375);

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
