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

#ifndef __CORE_CPU_SIGMOID_H__
#define __CORE_CPU_SIGMOID_H__

#include "cpu_exp.h"

namespace core
{
	static constexpr float  flt_sigmoid_min = -1.59423850332565686113e001F; /* ln(epsilon/(1-epslon)) */
	static constexpr float  flt_sigmoid_max =  1.59423851524658653167e001F; /* -ln(epsilon) */
	static constexpr double dbl_sigmoid_min = -3.60436533891171558590e001;  /* ln(epsilon/(1-epslon)) */
	static constexpr double dbl_sigmoid_max =  3.60436533891171560811e001;  /* -ln(epsilon) */

	static constexpr ALIGN(64) float  a8_flt_sigmoid_min[8] = CONST_ARRAY_8(flt_sigmoid_min);
	static constexpr ALIGN(64) float  a8_flt_sigmoid_max[8] = CONST_ARRAY_8(flt_sigmoid_max);
	static constexpr ALIGN(64) double a4_dbl_sigmoid_min[4] = CONST_ARRAY_4(dbl_sigmoid_min);
	static constexpr ALIGN(64) double a4_dbl_sigmoid_max[4] = CONST_ARRAY_4(dbl_sigmoid_max);

	// Function template sigmoid

	extern float sigmoid(float x)
	{
		if (x < flt_sigmoid_min)
			x = flt_sigmoid_min;
		else if (x > flt_sigmoid_max)
			x = flt_sigmoid_max;
		float y = flt_one / (flt_one + core::exp(-x));
		return y;
	}

	extern double sigmoid(double x)
	{
		if (x < dbl_sigmoid_min)
			x = dbl_sigmoid_min;
		else if (x > dbl_sigmoid_max)
			x = dbl_sigmoid_max;
		double y = dbl_one / (dbl_one + core::exp(-x));
		return y;
	}

/*
	template<cpu_inst_type inst>
	__m128 expf4(__m128 xmm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m128 expf4<cpu_sse2>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_log2e));
		// the round number of t
		__m128 xmm_s = _mm_and_ps(xmm_t, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_h = _mm_or_ps(xmm_s, *reinterpret_cast<const __m128*>(a8_flt_half));
		__m128i xmm_i = _mm_cvtps_epi32(_mm_add_ps(xmm_t, xmm_h));
		__m128 xmm_r = _mm_cvtepi32_ps(xmm_i);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m128 xmm_y = *reinterpret_cast<const __m128*>(a8_flt_exp_7);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_6));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_5));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_4));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_3));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_2));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_1));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_0));
		// i = 2^r;
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(a8_flt_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	template<>
	__m128 expf4<cpu_sse2 | cpu_fma>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_log2e));
		// the round number of t
		__m128 xmm_s = _mm_and_ps(xmm_t, *reinterpret_cast<const __m128*>(a8_flt_sign));
		__m128 xmm_h = _mm_or_ps(xmm_s, *reinterpret_cast<const __m128*>(a8_flt_half));
		__m128i xmm_i = _mm_cvtps_epi32(_mm_add_ps(xmm_t, xmm_h));
		__m128 xmm_r = _mm_cvtepi32_ps(xmm_i);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m128 xmm_y = *reinterpret_cast<const __m128*>(a8_flt_exp_7);
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_6));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_5));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_4));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_3));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_2));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_1));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_0));
		// i = 2^r;
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(a8_flt_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	template<>
	__m128 expf4<cpu_sse41>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_log2e));
		// r = round(t);
		__m128 xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m128 xmm_y = *reinterpret_cast<const __m128*>(a8_flt_exp_7);
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_6));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_5));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_4));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_3));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_2));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_1));
		xmm_y = _mm_add_ps(_mm_mul_ps(xmm_y, xmm_x), *reinterpret_cast<const __m128*>(a8_flt_exp_0));
		// i = 2^r;
		__m128i xmm_i = _mm_cvttps_epi32(xmm_r);
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(a8_flt_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	template<>
	__m128 expf4<cpu_sse41 | cpu_fma>(__m128 xmm_x)
	{
		// t = x * log2(e);
		__m128 xmm_t = _mm_mul_ps(xmm_x, *reinterpret_cast<const __m128*>(a8_flt_log2e));
		// r = round(t);
		__m128 xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_hi)));
		xmm_x = _mm_sub_ps(xmm_x, _mm_mul_ps(xmm_r, *reinterpret_cast<const __m128*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m128 xmm_y = *reinterpret_cast<const __m128*>(a8_flt_exp_7);
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_6));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_5));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_4));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_3));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_2));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_1));
		xmm_y = _mm_fmadd_ps(xmm_y, xmm_x, *reinterpret_cast<const __m128*>(a8_flt_exp_0));
		// i = 2^r;
		__m128i xmm_i = _mm_cvttps_epi32(xmm_r);
		xmm_i = _mm_add_epi32(xmm_i, *reinterpret_cast<const __m128i*>(a8_flt_base));
		xmm_i = _mm_slli_epi32(xmm_i, 23);
		// y *= (float) i;
		xmm_y = _mm_mul_ps(xmm_y, _mm_castsi128_ps(xmm_i));
		return xmm_y;
	}

	template<cpu_inst_type inst>
	__m128d expd2(__m128d xmm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m128d expd2<cpu_sse2>(__m128d xmm_x)
	{
		// t = x * log2(e);
		__m128d xmm_t = _mm_mul_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_log2e));
		// the round number of t
		__m128d xmm_s = _mm_and_pd(xmm_t, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_h = _mm_or_pd(xmm_s, *reinterpret_cast<const __m128d*>(a4_dbl_half));
		__m128i xmm_i = _mm_cvtpd_epi64(_mm_add_pd(xmm_t, xmm_h));
		__m128d xmm_r = _mm_cvtepi64_pd(xmm_i);
		// x -= r * ln2;
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_hi)));
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m128d xmm_p = *reinterpret_cast<const __m128d*>(a4_dbl_exp_p6);
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p5));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p4));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p3));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p2));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p1));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m128d xmm_q = *reinterpret_cast<const __m128d*>(a4_dbl_exp_q6);
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q5));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q4));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q3));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q2));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q1));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m128d xmm_y = _mm_div_pd(xmm_p, xmm_q);
		// i = 2^r;
		xmm_i = _mm_add_epi64(xmm_i, *reinterpret_cast<const __m128i*>(a4_dbl_base));
		xmm_i = _mm_slli_epi64(xmm_i, 52);
		// y *= (double) i;
		xmm_y = _mm_mul_pd(xmm_y, _mm_castsi128_pd(xmm_i));
		return xmm_y;
	}

	template<>
	__m128d expd2<cpu_sse2 | cpu_fma>(__m128d xmm_x)
	{
		// t = x * log2(e);
		__m128d xmm_t = _mm_mul_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_log2e));
		// the round number of t
		__m128d xmm_s = _mm_and_pd(xmm_t, *reinterpret_cast<const __m128d*>(a4_dbl_sign));
		__m128d xmm_h = _mm_or_pd(xmm_s, *reinterpret_cast<const __m128d*>(a4_dbl_half));
		__m128i xmm_i = _mm_cvtpd_epi64(_mm_add_pd(xmm_t, xmm_h));
		__m128d xmm_r = _mm_cvtepi64_pd(xmm_i);
		// x -= r * ln2;
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_hi)));
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m128d xmm_p = *reinterpret_cast<const __m128d*>(a4_dbl_exp_p6);
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p5));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p4));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p3));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p2));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p1));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m128d xmm_q = *reinterpret_cast<const __m128d*>(a4_dbl_exp_q6);
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q5));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q4));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q3));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q2));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q1));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m128d xmm_y = _mm_div_pd(xmm_p, xmm_q);
		// i = 2^r;
		xmm_i = _mm_add_epi64(xmm_i, *reinterpret_cast<const __m128i*>(a4_dbl_base));
		xmm_i = _mm_slli_epi64(xmm_i, 52);
		// y *= (double) i;
		xmm_y = _mm_mul_pd(xmm_y, _mm_castsi128_pd(xmm_i));
		return xmm_y;
	}

	template<>
	__m128d expd2<cpu_sse41>(__m128d xmm_x)
	{
		// t = x * log2(e);
		__m128d xmm_t = _mm_mul_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_log2e));
		// r = round(t);
		__m128d xmm_r = _mm_round_pd(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_hi)));
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m128d xmm_p = *reinterpret_cast<const __m128d*>(a4_dbl_exp_p6);
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p5));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p4));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p3));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p2));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p1));
		xmm_p = _mm_add_pd(_mm_mul_pd(xmm_p, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m128d xmm_q = *reinterpret_cast<const __m128d*>(a4_dbl_exp_q6);
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q5));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q4));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q3));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q2));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q1));
		xmm_q = _mm_add_pd(_mm_mul_pd(xmm_q, xmm_x), *reinterpret_cast<const __m128d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m128d xmm_y = _mm_div_pd(xmm_p, xmm_q);
		// i = 2^r;
		__m128i xmm_i = _mm_cvttpd_epi64(xmm_r);
		xmm_i = _mm_add_epi64(xmm_i, *reinterpret_cast<const __m128i*>(a4_dbl_base));
		xmm_i = _mm_slli_epi64(xmm_i, 52);
		// y *= (double) i;
		xmm_y = _mm_mul_pd(xmm_y, _mm_castsi128_pd(xmm_i));
		return xmm_y;
	}

	template<>
	__m128d expd2<cpu_sse41 | cpu_fma>(__m128d xmm_x)
	{
		// t = x * log2(e);
		__m128d xmm_t = _mm_mul_pd(xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_log2e));
		// r = round(t);
		__m128d xmm_r = _mm_round_pd(xmm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_hi)));
		xmm_x = _mm_sub_pd(xmm_x, _mm_mul_pd(xmm_r, *reinterpret_cast<const __m128d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m128d xmm_p = *reinterpret_cast<const __m128d*>(a4_dbl_exp_p6);
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p5));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p4));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p3));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p2));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p1));
		xmm_p = _mm_fmadd_pd(xmm_p, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m128d xmm_q = *reinterpret_cast<const __m128d*>(a4_dbl_exp_q6);
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q5));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q4));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q3));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q2));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q1));
		xmm_q = _mm_fmadd_pd(xmm_q, xmm_x, *reinterpret_cast<const __m128d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m128d xmm_y = _mm_div_pd(xmm_p, xmm_q);
		// i = 2^r;
		__m128i xmm_i = _mm_cvttpd_epi64(xmm_r);
		xmm_i = _mm_add_epi64(xmm_i, *reinterpret_cast<const __m128i*>(a4_dbl_base));
		xmm_i = _mm_slli_epi64(xmm_i, 52);
		// y *= (double) i;
		xmm_y = _mm_mul_pd(xmm_y, _mm_castsi128_pd(xmm_i));
		return xmm_y;
	}

	template<cpu_inst_type inst>
	__m256 expf8(__m256 ymm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m256 expf8<cpu_avx>(__m256 ymm_x)
	{
		// t = x * log2(e);
		__m256 ymm_t = _mm256_mul_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_log2e));
		// r = round(t);
		__m256 ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(a8_flt_ln2_hi)));
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m256 ymm_y = *reinterpret_cast<const __m256*>(a8_flt_exp_7);
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_6));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_5));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_4));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_3));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_2));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_1));
		ymm_y = _mm256_add_ps(_mm256_mul_ps(ymm_y, ymm_x), *reinterpret_cast<const __m256*>(a8_flt_exp_0));
		// i = 2^r;
		__m256i ymm_i = _mm256_cvttps_epi32(ymm_r);
		ymm_i = _mm256_add_epi32(ymm_i, *reinterpret_cast<const __m256i*>(a8_flt_base));
		ymm_i = _mm256_slli_epi32(ymm_i, 23);
		// y *= (float) i;
		ymm_y = _mm256_mul_ps(ymm_y, _mm256_castsi256_ps(ymm_i));
		return ymm_y;
	}

	template<>
	__m256 expf8<cpu_avx | cpu_fma>(__m256 ymm_x)
	{
		// t = x * log2(e);
		__m256 ymm_t = _mm256_mul_ps(ymm_x, *reinterpret_cast<const __m256*>(a8_flt_log2e));
		// r = round(t);
		__m256 ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(a8_flt_ln2_hi)));
		ymm_x = _mm256_sub_ps(ymm_x, _mm256_mul_ps(ymm_r, *reinterpret_cast<const __m256*>(a8_flt_ln2_lo)));
		// Taylor series of e^x:
		// y = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6! + x^7/7!;
		__m256 ymm_y = *reinterpret_cast<const __m256*>(a8_flt_exp_7);
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_6));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_5));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_4));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_3));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_2));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_1));
		ymm_y = _mm256_fmadd_ps(ymm_y, ymm_x, *reinterpret_cast<const __m256*>(a8_flt_exp_0));
		// i = 2^r;
		__m256i ymm_i = _mm256_cvttps_epi32(ymm_r);
		ymm_i = _mm256_add_epi32(ymm_i, *reinterpret_cast<const __m256i*>(a8_flt_base));
		ymm_i = _mm256_slli_epi32(ymm_i, 23);
		// y *= (float) i;
		ymm_y = _mm256_mul_ps(ymm_y, _mm256_castsi256_ps(ymm_i));
		return ymm_y;
	}

	template<cpu_inst_type inst>
	__m256d expd4(__m256d ymm_x)
	{
		throw ::std::invalid_argument(invalid_template_parameter);
	}

	template<>
	__m256d expd4<cpu_avx>(__m256d ymm_x)
	{
		// t = x * log2(e);
		__m256d ymm_t = _mm256_mul_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_log2e));
		// r = round(t);
		__m256d ymm_r = _mm256_round_pd(ymm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		ymm_x = _mm256_sub_pd(ymm_x, _mm256_mul_pd(ymm_r, *reinterpret_cast<const __m256d*>(a4_dbl_ln2_hi)));
		ymm_x = _mm256_sub_pd(ymm_x, _mm256_mul_pd(ymm_r, *reinterpret_cast<const __m256d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m256d ymm_p = *reinterpret_cast<const __m256d*>(a4_dbl_exp_p6);
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p5));
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p4));
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p3));
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p2));
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p1));
		ymm_p = _mm256_add_pd(_mm256_mul_pd(ymm_p, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m256d ymm_q = *reinterpret_cast<const __m256d*>(a4_dbl_exp_q6);
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q5));
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q4));
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q3));
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q2));
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q1));
		ymm_q = _mm256_add_pd(_mm256_mul_pd(ymm_q, ymm_x), *reinterpret_cast<const __m256d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m256d ymm_y = _mm256_div_pd(ymm_p, ymm_q);
		// i = 2^r;
		__m256i ymm_i = _mm256_cvttpd_epi64(ymm_r);
		ymm_i = _mm256_add_epi64(ymm_i, *reinterpret_cast<const __m256i*>(a4_dbl_base));
		ymm_i = _mm256_slli_epi64(ymm_i, 52);
		// y *= (double) i;
		ymm_y = _mm256_mul_pd(ymm_y, _mm256_castsi256_pd(ymm_i));
		return ymm_y;
	}

	template<>
	__m256d expd4<cpu_avx | cpu_fma>(__m256d ymm_x)
	{
		// t = x * log2(e);
		__m256d ymm_t = _mm256_mul_pd(ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_log2e));
		// r = round(t);
		__m256d ymm_r = _mm256_round_pd(ymm_t, _MM_FROUND_NINT);
		// x -= r * ln2;
		ymm_x = _mm256_sub_pd(ymm_x, _mm256_mul_pd(ymm_r, *reinterpret_cast<const __m256d*>(a4_dbl_ln2_hi)));
		ymm_x = _mm256_sub_pd(ymm_x, _mm256_mul_pd(ymm_r, *reinterpret_cast<const __m256d*>(a4_dbl_ln2_lo)));
		// P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5+ p6*x^6;
		__m256d ymm_p = *reinterpret_cast<const __m256d*>(a4_dbl_exp_p6);
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p5));
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p4));
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p3));
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p2));
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p1));
		ymm_p = _mm256_fmadd_pd(ymm_p, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_p0));
		// Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4 + q5*x^5 + q6*x^6;
		__m256d ymm_q = *reinterpret_cast<const __m256d*>(a4_dbl_exp_q6);
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q5));
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q4));
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q3));
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q2));
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q1));
		ymm_q = _mm256_fmadd_pd(ymm_q, ymm_x, *reinterpret_cast<const __m256d*>(a4_dbl_exp_q0));
		// y = P(x) / Q(x);
		__m256d ymm_y = _mm256_div_pd(ymm_p, ymm_q);
		// i = 2^r;
		__m256i ymm_i = _mm256_cvttpd_epi64(ymm_r);
		ymm_i = _mm256_add_epi64(ymm_i, *reinterpret_cast<const __m256i*>(a4_dbl_base));
		ymm_i = _mm256_slli_epi64(ymm_i, 52);
		// y *= (double) i;
		ymm_y = _mm256_mul_pd(ymm_y, _mm256_castsi256_pd(ymm_i));
		return ymm_y;
	}
*/
} // namespace core

#endif
