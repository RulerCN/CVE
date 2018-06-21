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

#ifndef __CORE_CPU_MATH_AVX_H__
#define __CORE_CPU_MATH_AVX_H__

#include "../definition.h"
#include "../instruction.h"

namespace core
{
	// https://stackoverflow.com/questions/45770089/efficient-implementation-of-log2-m256d-in-avx2/45898937#45898937

	static const __m256i mm256_0x7f          = _mm256_set1_epi32(0x0000007f);
	static const __m256i mm256_min_norm_pos  = _mm256_set1_epi32(0x00800000);
	static const __m256i mm256_mant_mask     = _mm256_set1_epi32(0x7f800000);
	static const __m256i mm256_inv_mant_mask = _mm256_set1_epi32(~0x7f800000);

	static const __m256  mm256_zero          = _mm256_setzero_ps();
	static const __m256  mm256_0p5           = _mm256_set1_ps(0.5F);
	static const __m256  mm256_one           = _mm256_set1_ps(1.0F);

	static const __m256  mm256_exp_hi        = _mm256_set1_ps(88.3762626647949F);
	static const __m256  mm256_exp_lo        = _mm256_set1_ps(-88.3762626647949F);
	static const __m256  mm256_log2ef        = _mm256_set1_ps(1.44269504088896341F);
	static const __m256  mm256_exp_c1        = _mm256_set1_ps(0.693359375F);
	static const __m256  mm256_exp_c2        = _mm256_set1_ps(-2.12194440e-4F);
	static const __m256  mm256_exp_p0        = _mm256_set1_ps(1.9875691500e-4F);
	static const __m256  mm256_exp_p1        = _mm256_set1_ps(1.3981999507e-3F);
	static const __m256  mm256_exp_p2        = _mm256_set1_ps(8.3334519073e-3F);
	static const __m256  mm256_exp_p3        = _mm256_set1_ps(4.1665795894e-2F);
	static const __m256  mm256_exp_p4        = _mm256_set1_ps(1.6666665459e-1F);
	static const __m256  mm256_exp_p5        = _mm256_set1_ps(5.0000001201e-1F);

	static const __m256  mm256_sqrthf        = _mm256_set1_ps(0.707106781186547524F);
	static const __m256  mm256_log_p0        = _mm256_set1_ps(7.0376836292e-2F);
	static const __m256  mm256_log_p1        = _mm256_set1_ps(-1.1514610310e-1F);
	static const __m256  mm256_log_p2        = _mm256_set1_ps(1.1676998740e-1F);
	static const __m256  mm256_log_p3        = _mm256_set1_ps(-1.2420140846e-1F);
	static const __m256  mm256_log_p4        = _mm256_set1_ps(1.4249322787e-1F);
	static const __m256  mm256_log_p5        = _mm256_set1_ps(-1.6668057665e-1F);
	static const __m256  mm256_log_p6        = _mm256_set1_ps(2.0000714765e-1F);
	static const __m256  mm256_log_p7        = _mm256_set1_ps(-2.4999993993e-1F);
	static const __m256  mm256_log_p8        = _mm256_set1_ps(3.3333331174e-1F);
	static const __m256  mm256_log_q1        = _mm256_set1_ps(-2.12194440e-4F);
	static const __m256  mm256_log_q2        = _mm256_set1_ps(0.693359375F);


	__m256 _mm256_exp_ps(__m256 x)
	{
		x = _mm256_min_ps(x, mm256_exp_hi);
		x = _mm256_max_ps(x, mm256_exp_lo);

		// express exp(x) as exp(g + n*log(2))
		__m256 fx = _mm256_mul_ps(x, mm256_log2ef);
		fx = _mm256_add_ps(fx, mm256_0p5);

		__m256 tmp = _mm256_floor_ps(fx);
		__m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
		mask = _mm256_and_ps(mask, mm256_one);
		fx = _mm256_sub_ps(tmp, mask);
		tmp = _mm256_mul_ps(fx, mm256_exp_c1);

		__m256 z = _mm256_mul_ps(fx, mm256_exp_c2);
		x = _mm256_sub_ps(x, tmp);
		x = _mm256_sub_ps(x, z);
		z = _mm256_mul_ps(x, x);

		__m256 y = mm256_exp_p0;
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_exp_p1);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_exp_p2);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_exp_p3);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_exp_p4);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_exp_p5);
		y = _mm256_mul_ps(y, z);
		y = _mm256_add_ps(y, x);
		y = _mm256_add_ps(y, mm256_one);

		__m256i imm0 = _mm256_cvttps_epi32(fx);
		imm0 = _mm256_add_epi32(imm0, mm256_0x7f);
		imm0 = _mm256_slli_epi32(imm0, 23);

		__m256 pow2n = _mm256_castsi256_ps(imm0);
		y = _mm256_mul_ps(y, pow2n);
		return y;
	}

	__m256 _mm256_log_ps(__m256 x)
	{
		__m256 invalid_mask = _mm256_cmp_ps(x, mm256_zero, _CMP_LE_OS);
		x = _mm256_max_ps(x, _mm256_castsi256_ps(mm256_min_norm_pos));

		__m256i imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
		imm0 = _mm256_sub_epi32(imm0, mm256_0x7f);
		__m256 e = _mm256_cvtepi32_ps(imm0);
		e = _mm256_add_ps(e, mm256_one);

		x = _mm256_and_ps(x, _mm256_castsi256_ps(mm256_inv_mant_mask));
		x = _mm256_or_ps(x, mm256_0p5);

		__m256 mask = _mm256_cmp_ps(x, mm256_sqrthf, _CMP_LT_OS);
		__m256 tmp = _mm256_and_ps(x, mask);
		x = _mm256_sub_ps(x, mm256_one);
		e = _mm256_sub_ps(e, _mm256_and_ps(mm256_one, mask));
		x = _mm256_add_ps(x, tmp);

		__m256 z = _mm256_mul_ps(x, x);
		__m256 y = mm256_log_p0;
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p1);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p2);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p3);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p4);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p5);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p6);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p7);
		y = _mm256_mul_ps(y, x);
		y = _mm256_add_ps(y, mm256_log_p8);
		y = _mm256_mul_ps(y, x);
		y = _mm256_mul_ps(y, z);

		tmp = _mm256_mul_ps(e, mm256_log_q1);
		y = _mm256_add_ps(y, tmp);
		tmp = _mm256_mul_ps(z, mm256_0p5);
		y = _mm256_sub_ps(y, tmp);
		tmp = _mm256_mul_ps(e, mm256_log_q2);
		x = _mm256_add_ps(x, y);
		x = _mm256_add_ps(x, tmp);
		x = _mm256_or_ps(x, invalid_mask);
		return x;
	}

} // namespace core

#endif
