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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_EXP_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_EXP_H__

#include "kernel_arithmetic_math.h"

namespace core
{
	// Class template kernel_exp

	template<class T, cpu_inst_type inst>
	struct kernel_exp
	{
		void operator()(size_t n, const T *a, T *b) const
		{
			constexpr size_t block = 8;

			while (n >= block)
			{
				b[0] = exp(a[0]);
				b[1] = exp(a[1]);
				b[2] = exp(a[2]);
				b[3] = exp(a[3]);
				b[4] = exp(a[4]);
				b[5] = exp(a[5]);
				b[6] = exp(a[6]);
				b[7] = exp(a[7]);
				a += block;
				b += block;
				n -= block;
			}
			for (size_t i = 0; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse41>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128 xmm_a, xmm_t, xmm_r;
			__m128 xmm_b, xmm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// a = max(a, exp_min);
				xmm_a = _mm_max_ps(xmm_a, xmm_expf_min);
				// a = min(a, exp_max);
				xmm_a = _mm_min_ps(xmm_a, xmm_expf_max);
				// t = a * log2e;
				xmm_t = _mm_mul_ps(xmm_a, xmm_log2ef);
				// r = round(t);
				xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmp_ps(xmm_r, xmm_t, _CMP_GT_OS);
				xmm_r = _mm_sub_ps(xmm_r, _mm_and_ps(xmm_mask, xmm_onef));
				// a -= r * ln2_hi;
				xmm_a = _mm_sub_ps(xmm_a, _mm_mul_ps(xmm_r, xmm_ln2f_hi));
				// a -= r * ln2_lo;
				xmm_a = _mm_sub_ps(xmm_a, _mm_mul_ps(xmm_r, xmm_ln2f_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				xmm_b = xmm_expf_p7;
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p6);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p5);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p4);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p3);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p2);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_expf_p1);
				xmm_b = _mm_add_ps(_mm_mul_ps(xmm_b, xmm_a), xmm_onef);
				// i = 2^r;
				xmm_i = _mm_cvttps_epi32(xmm_r);
				xmm_i = _mm_add_epi32(xmm_i, xmm_flt_base);
				xmm_i = _mm_slli_epi32(xmm_i, 23);
				// b *= (float) i;
				xmm_b = _mm_mul_ps(xmm_b, _mm_castsi128_ps(xmm_i));
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse41 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128i xmm_i;
			__m128 xmm_a, xmm_t, xmm_r;
			__m128 xmm_b, xmm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// a = max(a, exp_min);
				xmm_a = _mm_max_ps(xmm_a, xmm_expf_min);
				// a = min(a, exp_max);
				xmm_a = _mm_min_ps(xmm_a, xmm_expf_max);
				// t = a * log2e;
				xmm_t = _mm_mul_ps(xmm_a, xmm_log2ef);
				// r = round(t);
				xmm_r = _mm_round_ps(xmm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				xmm_mask = _mm_cmp_ps(xmm_r, xmm_t, _CMP_GT_OS);
				xmm_r = _mm_sub_ps(xmm_r, _mm_and_ps(xmm_mask, xmm_onef));
				// a -= r * ln2_hi;
				xmm_a = _mm_sub_ps(xmm_a, _mm_mul_ps(xmm_r, xmm_ln2f_hi));
				// a -= r * ln2_lo;
				xmm_a = _mm_sub_ps(xmm_a, _mm_mul_ps(xmm_r, xmm_ln2f_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				xmm_b = xmm_expf_p7;
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p6);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p5);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p4);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p3);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p2);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_expf_p1);
				xmm_b = _mm_fmadd_ps(xmm_b, xmm_a, xmm_onef);
				// i = 2^r;
				xmm_i = _mm_cvttps_epi32(xmm_r);
				xmm_i = _mm_add_epi32(xmm_i, xmm_flt_base);
				xmm_i = _mm_slli_epi32(xmm_i, 23);
				// b *= (float) i;
				xmm_b = _mm_mul_ps(xmm_b, _mm_castsi128_ps(xmm_i));
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
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
	struct kernel_exp<float, cpu_avx2>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256 ymm_a, ymm_t, ymm_r;
			__m256 ymm_b, ymm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// a = max(a, exp_min);
				ymm_a = _mm256_max_ps(ymm_a, ymm_expf_min);
				// a = min(a, exp_max);
				ymm_a = _mm256_min_ps(ymm_a, ymm_expf_max);
				// t = a * log2e;
				ymm_t = _mm256_mul_ps(ymm_a, ymm_log2ef);
				// r = round(t);
				ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_ps(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_ps(ymm_r, _mm256_and_ps(ymm_mask, ymm_onef));
				// a -= r * ln2_hi;
				ymm_a = _mm256_sub_ps(ymm_a, _mm256_mul_ps(ymm_r, ymm_ln2f_hi));
				// a -= r * ln2_lo;
				ymm_a = _mm256_sub_ps(ymm_a, _mm256_mul_ps(ymm_r, ymm_ln2f_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				ymm_b = ymm_expf_p7;
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p6);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p5);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p4);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p3);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p2);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_expf_p1);
				ymm_b = _mm256_add_ps(_mm256_mul_ps(ymm_b, ymm_a), ymm_onef);
				// i = 2^r;
				ymm_i = _mm256_cvttps_epi32(ymm_r);
				ymm_i = _mm256_add_epi32(ymm_i, ymm_flt_base);
				ymm_i = _mm256_slli_epi32(ymm_i, 23);
				// b *= (float) i;
				ymm_b = _mm256_mul_ps(ymm_b, _mm256_castsi256_ps(ymm_i));
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_avx2 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256i ymm_i;
			__m256 ymm_a, ymm_t, ymm_r;
			__m256 ymm_b, ymm_mask;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// a = max(a, exp_min);
				ymm_a = _mm256_max_ps(ymm_a, ymm_expf_min);
				// a = min(a, exp_max);
				ymm_a = _mm256_min_ps(ymm_a, ymm_expf_max);
				// t = a * log2e;
				ymm_t = _mm256_mul_ps(ymm_a, ymm_log2ef);
				// r = round(t);
				ymm_r = _mm256_round_ps(ymm_t, _MM_FROUND_NINT);
				// if (r > t) r -= 1;
				ymm_mask = _mm256_cmp_ps(ymm_r, ymm_t, _CMP_GT_OS);
				ymm_r = _mm256_sub_ps(ymm_r, _mm256_and_ps(ymm_mask, ymm_onef));
				// a -= r * ln2_hi;
				ymm_a = _mm256_sub_ps(ymm_a, _mm256_mul_ps(ymm_r, ymm_ln2f_hi));
				// a -= r * ln2_lo;
				ymm_a = _mm256_sub_ps(ymm_a, _mm256_mul_ps(ymm_r, ymm_ln2f_lo));
				// Maclaurin expansion of e^x:
				// b = 1 + a + a^2/2! + a^3/3! + a^4/4! + a^5/5! + a^6/6! + a^7/7!
				ymm_b = ymm_expf_p7;
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p6);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p5);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p4);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p3);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p2);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_expf_p1);
				ymm_b = _mm256_fmadd_ps(ymm_b, ymm_a, ymm_onef);
				// i = 2^r;
				ymm_i = _mm256_cvttps_epi32(ymm_r);
				ymm_i = _mm256_add_epi32(ymm_i, ymm_flt_base);
				ymm_i = _mm256_slli_epi32(ymm_i, 23);
				// b *= (float) i;
				ymm_b = _mm256_mul_ps(ymm_b, _mm256_castsi256_ps(ymm_i));
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp(a[i]);
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
