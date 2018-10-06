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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEMMT_DOUBLE_H__
#define __CORE_CPU_KERNEL_BLOCK_GEMMT_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gemmt_double
	
	template<cpu_inst_type inst>
	struct block_gemmt_double
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t /*aligned_p*/, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double val_a0, val_a1, val_a2, val_a3;
			double val_b0, val_b1, val_b2, val_b3;
			double val_t00, val_t01, val_t02, val_t03;
			double val_t10, val_t11, val_t12, val_t13;
			double val_t20, val_t21, val_t22, val_t23;
			double val_t30, val_t31, val_t32, val_t33;

			val_t00 = val_t01 = val_t02 = val_t03 = 0;
			val_t10 = val_t11 = val_t12 = val_t13 = 0;
			val_t20 = val_t21 = val_t22 = val_t23 = 0;
			val_t30 = val_t31 = val_t32 = val_t33 = 0;
			for (size_t k = 0; k < p; ++k)
			{
				val_a0 = ptr_a0[k];
				val_a1 = ptr_a1[k];
				val_a2 = ptr_a2[k];
				val_a3 = ptr_a3[k];
				val_b0 = ptr_b0[k];
				val_b1 = ptr_b1[k];
				val_b2 = ptr_b2[k];
				val_b3 = ptr_b3[k];
				val_t00 += val_a0 * val_b0;
				val_t01 += val_a0 * val_b1;
				val_t02 += val_a0 * val_b2;
				val_t03 += val_a0 * val_b3;
				val_t10 += val_a1 * val_b0;
				val_t11 += val_a1 * val_b1;
				val_t12 += val_a1 * val_b2;
				val_t13 += val_a1 * val_b3;
				val_t20 += val_a2 * val_b0;
				val_t21 += val_a2 * val_b1;
				val_t22 += val_a2 * val_b2;
				val_t23 += val_a2 * val_b3;
				val_t30 += val_a3 * val_b0;
				val_t31 += val_a3 * val_b1;
				val_t32 += val_a3 * val_b2;
				val_t33 += val_a3 * val_b3;
			}
			ptr_c0[0] += val_t00;
			ptr_c0[1] += val_t01;
			ptr_c0[2] += val_t02;
			ptr_c0[3] += val_t03;
			ptr_c1[0] += val_t10;
			ptr_c1[1] += val_t11;
			ptr_c1[2] += val_t12;
			ptr_c1[3] += val_t13;
			ptr_c2[0] += val_t20;
			ptr_c2[1] += val_t21;
			ptr_c2[2] += val_t22;
			ptr_c2[3] += val_t23;
			ptr_c3[0] += val_t30;
			ptr_c3[1] += val_t31;
			ptr_c3[2] += val_t32;
			ptr_c3[3] += val_t33;
		}
	};

	template<>
	struct block_gemmt_double<cpu_sse3>
	{
		// C(2x2) += A(2xp) * B(2xp)^T
		void operator()(size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t00, xmm_t01;
			__m128d xmm_t10, xmm_t11;
			__m128d xmm_c0 = _mm_loadu_pd(ptr_c0);
			__m128d xmm_c1 = _mm_loadu_pd(ptr_c1);

			if (aligned_p > 0)
			{
				xmm_t00 = xmm_t01 = _mm_setzero_pd();
				xmm_t10 = xmm_t11 = _mm_setzero_pd();
				for (size_t k = 0; k < aligned_p; k += 2)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
					xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
					xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
					xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
					// return the weighted sum
					xmm_t00 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_t00);
					xmm_t01 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b1), xmm_t01);
					xmm_t10 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b0), xmm_t10);
					xmm_t11 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b1), xmm_t11);
				}
				// return the horizontal sum
				xmm_t00 = _mm_hadd_pd(xmm_t00, xmm_t01);
				xmm_t10 = _mm_hadd_pd(xmm_t10, xmm_t11);
				xmm_c0 = _mm_add_pd(xmm_c0, xmm_t00);
				xmm_c1 = _mm_add_pd(xmm_c1, xmm_t10);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_pd(ptr_a0[k]);
					xmm_a1 = _mm_set1_pd(ptr_a1[k]);
					xmm_b0 = _mm_set_pd(ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_c0);
					xmm_c1 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b0), xmm_c1);
				}
			}
			// store data into memory
			_mm_storeu_pd(ptr_c0, xmm_c0);
			_mm_storeu_pd(ptr_c1, xmm_c1);
		}
	};

	template<>
	struct block_gemmt_double<cpu_sse3 | cpu_fma>
	{
		// C(2x2) += A(2xp) * B(2xp)^T
		void operator()(size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t00, xmm_t01;
			__m128d xmm_t10, xmm_t11;
			__m128d xmm_c0 = _mm_loadu_pd(ptr_c0);
			__m128d xmm_c1 = _mm_loadu_pd(ptr_c1);

			if (aligned_p > 0)
			{
				xmm_t00 = xmm_t01 = _mm_setzero_pd();
				xmm_t10 = xmm_t11 = _mm_setzero_pd();
				for (size_t k = 0; k < aligned_p; k += 2)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
					xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
					xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
					xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
					// return the weighted sum
					xmm_t00 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_t00);
					xmm_t01 = _mm_fmadd_pd(xmm_a0, xmm_b1, xmm_t01);
					xmm_t10 = _mm_fmadd_pd(xmm_a1, xmm_b0, xmm_t10);
					xmm_t11 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_t11);
				}
				// return the horizontal sum
				xmm_t00 = _mm_hadd_pd(xmm_t00, xmm_t01);
				xmm_t10 = _mm_hadd_pd(xmm_t10, xmm_t11);
				xmm_c0 = _mm_add_pd(xmm_c0, xmm_t00);
				xmm_c1 = _mm_add_pd(xmm_c1, xmm_t10);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_pd(ptr_a0[k]);
					xmm_a1 = _mm_set1_pd(ptr_a1[k]);
					xmm_b0 = _mm_set_pd(ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_c0);
					xmm_c1 = _mm_fmadd_pd(xmm_a1, xmm_b0, xmm_c1);
				}
			}
			// store data into memory
			_mm_storeu_pd(ptr_c0, xmm_c0);
			_mm_storeu_pd(ptr_c1, xmm_c1);
		}
	};

	template<>
	struct block_gemmt_double<cpu_avx>
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t00, ymm_t01, ymm_t02, ymm_t03;
			__m256d ymm_t10, ymm_t11, ymm_t12, ymm_t13;
			__m256d ymm_t20, ymm_t21, ymm_t22, ymm_t23;
			__m256d ymm_t30, ymm_t31, ymm_t32, ymm_t33;
			__m256d ymm_c0 = _mm256_loadu_pd(ptr_c0);
			__m256d ymm_c1 = _mm256_loadu_pd(ptr_c1);
			__m256d ymm_c2 = _mm256_loadu_pd(ptr_c2);
			__m256d ymm_c3 = _mm256_loadu_pd(ptr_c3);

			if (aligned_p > 0)
			{
				ymm_t00 = ymm_t01 = ymm_t02 = ymm_t03 = _mm256_setzero_pd();
				ymm_t10 = ymm_t11 = ymm_t12 = ymm_t13 = _mm256_setzero_pd();
				ymm_t20 = ymm_t21 = ymm_t22 = ymm_t23 = _mm256_setzero_pd();
				ymm_t30 = ymm_t31 = ymm_t32 = ymm_t33 = _mm256_setzero_pd();
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
					ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
					ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
					ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
					ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
					// return the weighted sum
					ymm_t00 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_t00);
					ymm_t01 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b1), ymm_t01);
					ymm_t02 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b2), ymm_t02);
					ymm_t03 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b3), ymm_t03);
					ymm_t10 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b0), ymm_t10);
					ymm_t11 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b1), ymm_t11);
					ymm_t12 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b2), ymm_t12);
					ymm_t13 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b3), ymm_t13);
					ymm_t20 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b0), ymm_t20);
					ymm_t21 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b1), ymm_t21);
					ymm_t22 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b2), ymm_t22);
					ymm_t23 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b3), ymm_t23);
					ymm_t30 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b0), ymm_t30);
					ymm_t31 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b1), ymm_t31);
					ymm_t32 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b2), ymm_t32);
					ymm_t33 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b3), ymm_t33);
				}
				// return the horizontal sum
				ymm_t00 = _mm256_hadd_pd(ymm_t00, ymm_t01);
				ymm_t10 = _mm256_hadd_pd(ymm_t10, ymm_t11);
				ymm_t20 = _mm256_hadd_pd(ymm_t20, ymm_t21);
				ymm_t30 = _mm256_hadd_pd(ymm_t30, ymm_t31);
				ymm_t02 = _mm256_hadd_pd(ymm_t02, ymm_t03);
				ymm_t12 = _mm256_hadd_pd(ymm_t12, ymm_t13);
				ymm_t22 = _mm256_hadd_pd(ymm_t22, ymm_t23);
				ymm_t32 = _mm256_hadd_pd(ymm_t32, ymm_t33);
				ymm_t01 = _mm256_permute2f128_pd(ymm_t00, ymm_t02, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t11 = _mm256_permute2f128_pd(ymm_t10, ymm_t12, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t21 = _mm256_permute2f128_pd(ymm_t20, ymm_t22, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t31 = _mm256_permute2f128_pd(ymm_t30, ymm_t32, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t03 = _mm256_permute2f128_pd(ymm_t00, ymm_t02, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t13 = _mm256_permute2f128_pd(ymm_t10, ymm_t12, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t23 = _mm256_permute2f128_pd(ymm_t20, ymm_t22, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t33 = _mm256_permute2f128_pd(ymm_t30, ymm_t32, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t00 = _mm256_add_pd(ymm_t01, ymm_t03);
				ymm_t10 = _mm256_add_pd(ymm_t11, ymm_t13);
				ymm_t20 = _mm256_add_pd(ymm_t21, ymm_t23);
				ymm_t30 = _mm256_add_pd(ymm_t31, ymm_t33);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t00);
				ymm_c1 = _mm256_add_pd(ymm_c1, ymm_t10);
				ymm_c2 = _mm256_add_pd(ymm_c2, ymm_t20);
				ymm_c3 = _mm256_add_pd(ymm_c3, ymm_t30);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_pd(ptr_a0[k]);
					ymm_a1 = _mm256_set1_pd(ptr_a1[k]);
					ymm_a2 = _mm256_set1_pd(ptr_a2[k]);
					ymm_a3 = _mm256_set1_pd(ptr_a3[k]);
					ymm_b0 = _mm256_set_pd(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_c0);
					ymm_c1 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b0), ymm_c1);
					ymm_c2 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b0), ymm_c2);
					ymm_c3 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b0), ymm_c3);
				}
			}
			// store data into memory
			_mm256_storeu_pd(ptr_c0, ymm_c0);
			_mm256_storeu_pd(ptr_c1, ymm_c1);
			_mm256_storeu_pd(ptr_c2, ymm_c2);
			_mm256_storeu_pd(ptr_c3, ymm_c3);
		}
	};

	template<>
	struct block_gemmt_double<cpu_avx | cpu_fma>
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t00, ymm_t01, ymm_t02, ymm_t03;
			__m256d ymm_t10, ymm_t11, ymm_t12, ymm_t13;
			__m256d ymm_t20, ymm_t21, ymm_t22, ymm_t23;
			__m256d ymm_t30, ymm_t31, ymm_t32, ymm_t33;
			__m256d ymm_c0 = _mm256_loadu_pd(ptr_c0);
			__m256d ymm_c1 = _mm256_loadu_pd(ptr_c1);
			__m256d ymm_c2 = _mm256_loadu_pd(ptr_c2);
			__m256d ymm_c3 = _mm256_loadu_pd(ptr_c3);

			if (aligned_p > 0)
			{
				ymm_t00 = ymm_t01 = ymm_t02 = ymm_t03 = _mm256_setzero_pd();
				ymm_t10 = ymm_t11 = ymm_t12 = ymm_t13 = _mm256_setzero_pd();
				ymm_t20 = ymm_t21 = ymm_t22 = ymm_t23 = _mm256_setzero_pd();
				ymm_t30 = ymm_t31 = ymm_t32 = ymm_t33 = _mm256_setzero_pd();
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
					ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
					ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
					ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
					ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
					// return the weighted sum
					ymm_t00 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_t00);
					ymm_t01 = _mm256_fmadd_pd(ymm_a0, ymm_b1, ymm_t01);
					ymm_t02 = _mm256_fmadd_pd(ymm_a0, ymm_b2, ymm_t02);
					ymm_t03 = _mm256_fmadd_pd(ymm_a0, ymm_b3, ymm_t03);
					ymm_t10 = _mm256_fmadd_pd(ymm_a1, ymm_b0, ymm_t10);
					ymm_t11 = _mm256_fmadd_pd(ymm_a1, ymm_b1, ymm_t11);
					ymm_t12 = _mm256_fmadd_pd(ymm_a1, ymm_b2, ymm_t12);
					ymm_t13 = _mm256_fmadd_pd(ymm_a1, ymm_b3, ymm_t13);
					ymm_t20 = _mm256_fmadd_pd(ymm_a2, ymm_b0, ymm_t20);
					ymm_t21 = _mm256_fmadd_pd(ymm_a2, ymm_b1, ymm_t21);
					ymm_t22 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_t22);
					ymm_t23 = _mm256_fmadd_pd(ymm_a2, ymm_b3, ymm_t23);
					ymm_t30 = _mm256_fmadd_pd(ymm_a3, ymm_b0, ymm_t30);
					ymm_t31 = _mm256_fmadd_pd(ymm_a3, ymm_b1, ymm_t31);
					ymm_t32 = _mm256_fmadd_pd(ymm_a3, ymm_b2, ymm_t32);
					ymm_t33 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_t33);
				}
				// return the horizontal sum
				ymm_t00 = _mm256_hadd_pd(ymm_t00, ymm_t01);
				ymm_t10 = _mm256_hadd_pd(ymm_t10, ymm_t11);
				ymm_t20 = _mm256_hadd_pd(ymm_t20, ymm_t21);
				ymm_t30 = _mm256_hadd_pd(ymm_t30, ymm_t31);
				ymm_t02 = _mm256_hadd_pd(ymm_t02, ymm_t03);
				ymm_t12 = _mm256_hadd_pd(ymm_t12, ymm_t13);
				ymm_t22 = _mm256_hadd_pd(ymm_t22, ymm_t23);
				ymm_t32 = _mm256_hadd_pd(ymm_t32, ymm_t33);
				ymm_t01 = _mm256_permute2f128_pd(ymm_t00, ymm_t02, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t11 = _mm256_permute2f128_pd(ymm_t10, ymm_t12, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t21 = _mm256_permute2f128_pd(ymm_t20, ymm_t22, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t31 = _mm256_permute2f128_pd(ymm_t30, ymm_t32, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t03 = _mm256_permute2f128_pd(ymm_t00, ymm_t02, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t13 = _mm256_permute2f128_pd(ymm_t10, ymm_t12, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t23 = _mm256_permute2f128_pd(ymm_t20, ymm_t22, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t33 = _mm256_permute2f128_pd(ymm_t30, ymm_t32, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t00 = _mm256_add_pd(ymm_t01, ymm_t03);
				ymm_t10 = _mm256_add_pd(ymm_t11, ymm_t13);
				ymm_t20 = _mm256_add_pd(ymm_t21, ymm_t23);
				ymm_t30 = _mm256_add_pd(ymm_t31, ymm_t33);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t00);
				ymm_c1 = _mm256_add_pd(ymm_c1, ymm_t10);
				ymm_c2 = _mm256_add_pd(ymm_c2, ymm_t20);
				ymm_c3 = _mm256_add_pd(ymm_c3, ymm_t30);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_pd(ptr_a0[k]);
					ymm_a1 = _mm256_set1_pd(ptr_a1[k]);
					ymm_a2 = _mm256_set1_pd(ptr_a2[k]);
					ymm_a3 = _mm256_set1_pd(ptr_a3[k]);
					ymm_b0 = _mm256_set_pd(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_c0);
					ymm_c1 = _mm256_fmadd_pd(ymm_a1, ymm_b0, ymm_c1);
					ymm_c2 = _mm256_fmadd_pd(ymm_a2, ymm_b0, ymm_c2);
					ymm_c3 = _mm256_fmadd_pd(ymm_a3, ymm_b0, ymm_c3);
				}
			}
			// store data into memory
			_mm256_storeu_pd(ptr_c0, ymm_c0);
			_mm256_storeu_pd(ptr_c1, ymm_c1);
			_mm256_storeu_pd(ptr_c2, ymm_c2);
			_mm256_storeu_pd(ptr_c3, ymm_c3);
		}
	};

} // namespace core

#endif
