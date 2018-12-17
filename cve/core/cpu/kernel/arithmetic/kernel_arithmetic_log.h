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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_LOG_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_LOG_H__

#include "kernel_arithmetic_math.h"

namespace core
{
	// Function template kernel_log

	template<cpu_inst_type inst>
	void kernel_log(size_t n, const float *a, float *b)
	{
		float flt_x, flt_e, flt_m;
		float flt_y, flt_p, flt_q;
		signed int *ptr_x, int_e, int_m;

		for (size_t i = 0; i < n; ++i)
		{
			flt_x = a[i] < flt_min ? flt_min : a[i];
			ptr_x = reinterpret_cast<signed int*>(flt_x);
			// keep the exponent part
			int_e = ((*ptr_x & flt_exp_mask) >> 23) - 127;
			flt_e = static_cast<float>(int_e);
			// keep the decimal part
			int_m = (*ptr_x & flt_mant_mask) | 0x3f800000;
			flt_m = *reinterpret_cast<float*>(&int_m);
			// if m >= sqrt(2) e = e + 1 and m = m / 2
			if (flt_m >= flt_sqrt2)
			{
				flt_e += flt_one;
				flt_m /= flt_two;
			}
			// y = (m - 1) / (m + 1)
			flt_y = (flt_m - flt_one) / (flt_m + flt_one);
			// x = y * y
			flt_x = flt_y * flt_y;
			// P(x) = p0 + p1 * x + p2 * x^2 + p3 * x^3
			flt_p = flt_log_p3;
			flt_p = flt_p * flt_x + flt_log_p2;
			flt_p = flt_p * flt_x + flt_log_p1;
			flt_p = flt_p * flt_x + flt_log_p0;
			// Q(x) = q0 + q1 * x + q2 * x^2 + q3 * x^3
			flt_q = flt_log_q3;
			flt_q = flt_q * flt_x + flt_log_q2;
			flt_q = flt_q * flt_x + flt_log_q1;
			flt_q = flt_q * flt_x + flt_log_q0;
			// y = y * P(x) / Q(x)
			flt_y *= flt_p / flt_q;
			// y += e * ln2;
			flt_y += flt_e * flt_ln2_hi;
			flt_y += flt_e * flt_ln2_lo;
			b[i] = flt_y;
		}
	};

	// Function template kernel_log

	template<cpu_inst_type inst>
	void kernel_log(size_t n, const double *a, double *b)
	{
		double dbl_x, dbl_e, dbl_m;
		double dbl_y, dbl_p, dbl_q;
		signed long long *ptr_x, lng_e, lng_m;

		for (size_t i = 0; i < n; ++i)
		{
			dbl_x = a[i] < dbl_min ? dbl_min : a[i];
			ptr_x = reinterpret_cast<signed long long*>(dbl_x);
			// keep the exponent part
			lng_e = ((*ptr_x & dbl_exp_mask) >> 52) - 1023;
			dbl_e = static_cast<double>(lng_e);
			// keep the decimal part
			lng_m = (*ptr_x & dbl_mant_mask) | 0x3ff0000000000000;
			dbl_m = *reinterpret_cast<double*>(&lng_m);
			// if m >= sqrt(2) e = e + 1 and m = m / 2
			if (dbl_m >= dbl_sqrt2)
			{
				dbl_e += dbl_one;
				dbl_m /= dbl_two;
			}
			// y = (m - 1) / (m + 1)
			dbl_y = (dbl_m - dbl_one) / (dbl_m + dbl_one);
			// x = y * y
			dbl_x = dbl_y * dbl_y;
			// P(x) = p0 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4
			dbl_p = dbl_log_p4;
			dbl_p = dbl_p * dbl_x + dbl_log_p3;
			dbl_p = dbl_p * dbl_x + dbl_log_p2;
			dbl_p = dbl_p * dbl_x + dbl_log_p1;
			dbl_p = dbl_p * dbl_x + dbl_log_p0;
			// Q(x) = q0 + q1 * x + q2 * x^2 + q3 * x^3 + q4 * x^4
			dbl_q = dbl_log_q4;
			dbl_q = dbl_q * dbl_x + dbl_log_q3;
			dbl_q = dbl_q * dbl_x + dbl_log_q2;
			dbl_q = dbl_q * dbl_x + dbl_log_q1;
			dbl_q = dbl_q * dbl_x + dbl_log_q0;
			// y = y * P(x) / Q(x)
			dbl_y *= dbl_p / dbl_q;
			// y += e * ln2;
			dbl_y += dbl_e * dbl_ln2_hi;
			dbl_y += dbl_e * dbl_ln2_lo;
			b[i] = dbl_y;
		}
	};

	template<>
	void kernel_log<cpu_sse2>(size_t n, const float *a, float *b)
	{
		constexpr size_t block = 4;
		const size_t aligned = n & ~(block - 1);
		__m128 xmm_x, xmm_e, xmm_m;
		__m128i xmm_i;
		float flt_y, flt_p, flt_q;
		signed int *ptr_x, int_e, int_m;

		for (size_t i = 0; i < aligned; i += block)
		{
			// load data from memory
			xmm_x = _mm_loadu_ps(a + i);
			// x = max(x, log_min);
			xmm_x = _mm_max_ps(xmm_x, xmm_logf_min);
			// keep the exponent part
			xmm_i = _mm_srli_epi32(_mm_castps_si128(xmm_x), 23);
			xmm_i = _mm_sub_epi32(xmm_i, xmm_flt_base);
			xmm_e = _mm_cvtepi32_ps(xmm_i);
			// keep the decimal part
			xmm_m = _mm_and_ps(xmm_x, _mm_castsi128_ps(xmm_flt_mant));
			xmm_m = _mm_or_ps(xmm_m, _mm_castsi128_ps(xmm_flt_exp));
			// if m >= sqrt(2) e = e + 1 and m = m / 2

			xmm_mask = _mm_cmp_pd(xmm_m, xmm_t, _CMP_GE_OS);
			xmm_r = _mm_sub_pd(xmm_r, _mm_and_pd(xmm_mask, xmm_oned));

			
			if (flt_m > flt_sqrt2)
			{
				flt_e += flt_one;
				flt_m /= flt_two;
			}
			// y = (m - 1) / (m + 1)
			flt_y = (flt_m - flt_one) / (flt_m + flt_one);
			// x = y * y
			flt_x = flt_y * flt_y;
			// P(x) = p0 + p1 * x + p2 * x^2 + p3 * x^3
			flt_p = flt_log_p3;
			flt_p = flt_p * flt_x + flt_log_p2;
			flt_p = flt_p * flt_x + flt_log_p1;
			flt_p = flt_p * flt_x + flt_log_p0;
			// Q(x) = q0 + q1 * x + q2 * x^2 + q3 * x^3
			flt_q = flt_log_q3;
			flt_q = flt_q * flt_x + flt_log_q2;
			flt_q = flt_q * flt_x + flt_log_q1;
			flt_q = flt_q * flt_x + flt_log_q0;
			// y = y * P(x) / Q(x)
			flt_y *= flt_p / flt_q;
			// y += e * ln2;
			flt_y += flt_e * flt_ln2_hi;
			flt_y += flt_e * flt_ln2_lo;
			b[i] = flt_y;
		}
	}

	template<>
	struct kernel_log<float, cpu_sse2>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128 xmm_a, xmm_e, xmm_m;
			__m128 xmm_b, xmm_neg, xmm_t;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// neg = a <= 0;
				xmm_neg = _mm_cmp_ps(xmm_a, xmm_zerof, _CMP_LE_OS);
				// a = max(a, log_min);
				xmm_a = _mm_max_ps(xmm_a, xmm_logf_min);
				// keep the exponent part
				xmm_i = _mm_srli_epi32(_mm_castps_si128(xmm_a), 23);
				xmm_i = _mm_sub_epi32(xmm_i, xmm_flt_base);
				xmm_e = _mm_cvtepi32_ps(xmm_i);
				// keep the decimal part
				xmm_m = _mm_and_ps(xmm_a, _mm_castsi128_ps(xmm_flt_mant));
				xmm_m = _mm_or_ps(xmm_m, _mm_castsi128_ps(xmm_flt_exp));
				// t = (m - 1) / (m + 1);
				xmm_t = _mm_div_ps(_mm_sub_ps(xmm_m, xmm_onef), _mm_add_ps(xmm_m, xmm_onef));
				// a = t * t;
				xmm_a = _mm_mul_ps(xmm_t, xmm_t);
				// Maclaurin expansion of ln((t+1)/(t-1)):
				// b = 2t(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
				xmm_b = xmm_logf_p7;
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p6);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p5);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p4);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p3);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p2);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_logf_p1);
				xmm_b = _mm_mul_ps(xmm_b, xmm_t);
				xmm_b = _mm_mul_ps(xmm_b, xmm_twof);
				// b += e * ln2_hi;
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_e, xmm_ln2f_hi), xmm_b);
				// b += e * ln2f_lo;
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_e, xmm_ln2f_lo), xmm_b);
				// negative number returns to NAN 
				xmm_b = _mm_or_ps(xmm_b, xmm_neg);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = log(a[i]);
		}
	};

	template<>
	struct kernel_log<float, cpu_sse2 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128 xmm_a, xmm_e, xmm_m;
			__m128 xmm_b, xmm_neg, xmm_t;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// neg = a <= 0;
				xmm_neg = _mm_cmp_ps(xmm_a, xmm_zerof, _CMP_LE_OS);
				// a = max(a, log_min);
				xmm_a = _mm_max_ps(xmm_a, xmm_logf_min);
				// keep the exponent part
				xmm_i = _mm_srli_epi32(_mm_castps_si128(xmm_a), 23);
				xmm_i = _mm_sub_epi32(xmm_i, xmm_flt_base);
				xmm_e = _mm_cvtepi32_ps(xmm_i);
				// keep the decimal part
				xmm_m = _mm_and_ps(xmm_a, _mm_castsi128_ps(xmm_flt_mant));
				xmm_m = _mm_or_ps(xmm_m, _mm_castsi128_ps(xmm_flt_exp));
				// t = (m - 1) / (m + 1);
				xmm_t = _mm_div_ps(_mm_sub_ps(xmm_m, xmm_onef), _mm_add_ps(xmm_m, xmm_onef));
				// a = t * t;
				xmm_a = _mm_mul_ps(xmm_t, xmm_t);
				// Maclaurin expansion of ln((t+1)/(t-1)):
				// b = 2t(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
				xmm_b = xmm_logf_p7;
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p6);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p5);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p4);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p3);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p2);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_logf_p1);
				xmm_b = _mm_mul_ps(xmm_b, xmm_t);
				xmm_b = _mm_mul_ps(xmm_b, xmm_twof);
				// b += e * ln2_hi;
				xmm_b = _mm_fmadd_ps(xmm_e, xmm_ln2f_hi, xmm_b);
				// b += e * ln2f_lo;
				xmm_b = _mm_fmadd_ps(xmm_e, xmm_ln2f_lo, xmm_b);
				// negative number returns to NAN 
				xmm_b = _mm_or_ps(xmm_b, xmm_neg);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = log(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse41>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128d xmm_a, xmm_t, xmm_r;
			__m128d xmm_b, xmm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// a = max(a, exp_min);
				xmm_a = _mm_max_pd(xmm_a, xmm_expd_min);
				// a = min(a, exp_max);
				xmm_a = _mm_min_pd(xmm_a, xmm_expd_max);
				// t = a * log2e;
				xmm_t = _mm_mul_pd(xmm_a, xmm_log2ed);
				// r = round(t);
				xmm_r = _mm_round_pd(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmp_pd(xmm_r, xmm_t, _CMP_GT_OS);
				xmm_r = _mm_sub_pd(xmm_r, _mm_and_pd(xmm_mask, xmm_oned));
				// a -= r * ln2_hi;
				xmm_a = _mm_sub_pd(xmm_a, _mm_mul_pd(xmm_r, xmm_ln2d_hi));
				// a -= r * ln2_lo;
				xmm_a = _mm_sub_pd(xmm_a, _mm_mul_pd(xmm_r, xmm_ln2d_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				//   + a^8/8! + a^9/9! + a^10/10! + a^11/11! + + a^12/12! + + a^13/13!
				xmm_b = xmm_expd_p13;
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p12);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p11);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p10);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p9);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p8);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p7);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p6);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p5);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p4);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p3);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p2);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_expd_p1);
				xmm_b = _mm_add_pd(_mm_mul_pd(xmm_b, xmm_a), xmm_oned);
				// i = 2^r
				xmm_i = _mm_cvtepi32_epi64(_mm_cvttpd_epi32(xmm_r));
				xmm_i = _mm_add_epi64(xmm_i, xmm_dbl_base);
				xmm_i = _mm_slli_epi64(xmm_i, 52);
				// b *= (double) i;
				xmm_b = _mm_mul_pd(xmm_b, _mm_castsi128_pd(xmm_i));
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse41 | cpu_fma>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128d xmm_a, xmm_t, xmm_r;
			__m128d xmm_b, xmm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// a = max(a, exp_min);
				xmm_a = _mm_max_pd(xmm_a, xmm_expd_min);
				// a = min(a, exp_max);
				xmm_a = _mm_min_pd(xmm_a, xmm_expd_max);
				// t = a * log2e;
				xmm_t = _mm_mul_pd(xmm_a, xmm_log2ed);
				// r = round(t);
				xmm_r = _mm_round_pd(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmp_pd(xmm_r, xmm_t, _CMP_GT_OS);
				xmm_r = _mm_sub_pd(xmm_r, _mm_and_pd(xmm_mask, xmm_oned));
				// a -= r * ln2_hi;
				xmm_a = _mm_sub_pd(xmm_a, _mm_mul_pd(xmm_r, xmm_ln2d_hi));
				// a -= r * ln2_lo;
				xmm_a = _mm_sub_pd(xmm_a, _mm_mul_pd(xmm_r, xmm_ln2d_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				//   + a^8/8! + a^9/9! + a^10/10! + a^11/11! + + a^12/12! + + a^13/13!
				xmm_b = xmm_expd_p13;
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p12);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p11);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p10);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p9);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p8);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p7);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p6);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p5);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p4);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p3);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p2);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_expd_p1);
				xmm_b = _mm_fmadd_pd(xmm_b, xmm_a, xmm_oned);
				// i = 2^r
				xmm_i = _mm_cvtepi32_epi64(_mm_cvttpd_epi32(xmm_r));
				xmm_i = _mm_add_epi64(xmm_i, xmm_dbl_base);
				xmm_i = _mm_slli_epi64(xmm_i, 52);
				// b *= (double) i;
				xmm_b = _mm_mul_pd(xmm_b, _mm_castsi128_pd(xmm_i));
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_log<float, cpu_avx2>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256 ymm_a, ymm_e, ymm_m;
			__m256 ymm_b, ymm_neg, ymm_t;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// neg = a <= 0;
				ymm_neg = _mm256_cmp_ps(ymm_a, ymm_zerof, _CMP_LE_OS);
				// a = max(a, log_min);
				ymm_a = _mm256_max_ps(ymm_a, ymm_logf_min);
				// keep the exponent part
				ymm_i = _mm256_srli_epi32(_mm256_castps_si256(ymm_a), 23);
				ymm_i = _mm256_sub_epi32(ymm_i, ymm_flt_base);
				ymm_e = _mm256_cvtepi32_ps(ymm_i);
				// keep the decimal part
				ymm_m = _mm256_and_ps(ymm_a, _mm256_castsi256_ps(ymm_flt_mant));
				ymm_m = _mm256_or_ps(ymm_m, _mm256_castsi256_ps(ymm_flt_exp));
				// t = (m - 1) / (m + 1);
				ymm_t = _mm256_div_ps(_mm256_sub_ps(ymm_m, ymm_onef), _mm256_add_ps(ymm_m, ymm_onef));
				// a = t * t;
				ymm_a = _mm256_mul_ps(ymm_t, ymm_t);
				// Maclaurin expansion of ln((t+1)/(t-1)):
				// b = 2t(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
				ymm_b = ymm_logf_p7;
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p6);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p5);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p4);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p3);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p2);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_logf_p1);
				ymm_b = _mm256_mul_ps(ymm_b, ymm_t);
				ymm_b = _mm256_mul_ps(ymm_b, ymm_twof);
				// b += e * ln2_hi;
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_e, ymm_ln2f_hi), ymm_b);
				// b += e * ln2f_lo;
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_e, ymm_ln2f_lo), ymm_b);
				// negative number returns to NAN 
				ymm_b = _mm256_or_ps(ymm_b, ymm_neg);
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = log(a[i]);
		}
	};

	template<>
	struct kernel_log<float, cpu_avx2 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256 ymm_a, ymm_e, ymm_m;
			__m256 ymm_b, ymm_neg, ymm_t;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// neg = a <= 0;
				ymm_neg = _mm256_cmp_ps(ymm_a, ymm_zerof, _CMP_LE_OS);
				// a = max(a, log_min);
				ymm_a = _mm256_max_ps(ymm_a, ymm_logf_min);
				// keep the exponent part
				ymm_i = _mm256_srli_epi32(_mm256_castps_si256(ymm_a), 23);
				ymm_i = _mm256_sub_epi32(ymm_i, ymm_flt_base);
				ymm_e = _mm256_cvtepi32_ps(ymm_i);
				// keep the decimal part
				ymm_m = _mm256_and_ps(ymm_a, _mm256_castsi256_ps(ymm_flt_mant));
				ymm_m = _mm256_or_ps(ymm_m, _mm256_castsi256_ps(ymm_flt_exp));
				// t = (m - 1) / (m + 1);
				ymm_t = _mm256_div_ps(_mm256_sub_ps(ymm_m, ymm_onef), _mm256_add_ps(ymm_m, ymm_onef));
				// a = t * t;
				ymm_a = _mm256_mul_ps(ymm_t, ymm_t);
				// Maclaurin expansion of ln((t+1)/(t-1)):
				// b = 2t(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 + t^10/11 + t^12/13)
				ymm_b = ymm_logf_p7;
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p6);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p5);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p4);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p3);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p2);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_logf_p1);
				ymm_b = _mm256_mul_ps(ymm_b, ymm_t);
				ymm_b = _mm256_mul_ps(ymm_b, ymm_twof);
				// b += e * ln2_hi;
				ymm_b = _mm256_fmadd_ps(ymm_e, ymm_ln2f_hi, ymm_b);
				// b += e * ln2f_lo;
				ymm_b = _mm256_fmadd_ps(ymm_e, ymm_ln2f_lo, ymm_b);
				// negative number returns to NAN 
				ymm_b = _mm256_or_ps(ymm_b, ymm_neg);
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = log(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_avx2>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256d ymm_a, ymm_t, ymm_r;
			__m256d ymm_b, ymm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_pd(a + i);
				// a = max(a, exp_min);
				ymm_a = _mm256_max_pd(ymm_a, ymm_expd_min);
				// a = min(a, exp_max);
				ymm_a = _mm256_min_pd(ymm_a, ymm_expd_max);
				// t = a * log2e;
				ymm_t = _mm256_mul_pd(ymm_a, ymm_log2ed);
				// r = round(t);
				ymm_r = _mm256_round_pd(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_pd(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_pd(ymm_r, _mm256_and_pd(ymm_mask, ymm_oned));
				// a -= r * ln2_hi;
				ymm_a = _mm256_sub_pd(ymm_a, _mm256_mul_pd(ymm_r, ymm_ln2d_hi));
				// a -= r * ln2_lo;
				ymm_a = _mm256_sub_pd(ymm_a, _mm256_mul_pd(ymm_r, ymm_ln2d_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				//   + a^8/8! + a^9/9! + a^10/10! + a^11/11! + + a^12/12! + + a^13/13!
				ymm_b = ymm_expd_p13;
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p12);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p11);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p10);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p9);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p8);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p7);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p6);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p5);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p4);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p3);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p2);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_expd_p1);
				ymm_b = _mm256_add_pd(_mm256_mul_pd(ymm_b, ymm_a), ymm_oned);
				// i = 2^r;
				ymm_i = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(ymm_r));
				ymm_i = _mm256_add_epi64(ymm_i, ymm_dbl_base);
				ymm_i = _mm256_slli_epi64(ymm_i, 52);
				// b *= (double) i;
				ymm_b = _mm256_mul_pd(ymm_b, _mm256_castsi256_pd(ymm_i));
				// store data into memory
				_mm256_storeu_pd(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_avx2 | cpu_fma>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256d ymm_a, ymm_t, ymm_r;
			__m256d ymm_b, ymm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_pd(a + i);
				// a = max(a, exp_min);
				ymm_a = _mm256_max_pd(ymm_a, ymm_expd_min);
				// a = min(a, exp_max);
				ymm_a = _mm256_min_pd(ymm_a, ymm_expd_max);
				// t = a * log2e;
				ymm_t = _mm256_mul_pd(ymm_a, ymm_log2ed);
				// r = round(t);
				ymm_r = _mm256_round_pd(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_pd(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_pd(ymm_r, _mm256_and_pd(ymm_mask, ymm_oned));
				// a -= r * ln2_hi;
				ymm_a = _mm256_sub_pd(ymm_a, _mm256_mul_pd(ymm_r, ymm_ln2d_hi));
				// a -= r * ln2_lo;
				ymm_a = _mm256_sub_pd(ymm_a, _mm256_mul_pd(ymm_r, ymm_ln2d_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				//   + a^8/8! + a^9/9! + a^10/10! + a^11/11! + + a^12/12! + + a^13/13!
				ymm_b = ymm_expd_p13;
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p12);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p11);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p10);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p9);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p8);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p7);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p6);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p5);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p4);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p3);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p2);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_expd_p1);
				ymm_b = _mm256_fmadd_pd(ymm_b, ymm_a, ymm_oned);
				// i = 2^r;
				ymm_i = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(ymm_r));
				ymm_i = _mm256_add_epi64(ymm_i, ymm_dbl_base);
				ymm_i = _mm256_slli_epi64(ymm_i, 52);
				// b *= (double) i;
				ymm_b = _mm256_mul_pd(ymm_b, _mm256_castsi256_pd(ymm_i));
				// store data into memory
				_mm256_storeu_pd(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

} // namespace core

#endif
