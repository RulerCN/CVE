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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEMM_FLOAT_H__
#define __CORE_CPU_KERNEL_BLOCK_GEMM_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gemmt_float
	
	template<cpu_inst_type inst>
	struct block_gemmt_float
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			float val_a0, val_a1, val_a2, val_a3;
			float val_t0, val_t1, val_t2, val_t3;

			for (size_t i = 0; i < 4; ++i)
			{
				val_a0 = a[0];
				val_a1 = a[1];
				val_a2 = a[2];
				val_a3 = a[3];
				for (size_t j = 0; j < n; ++j)
				{
					val_t0 = val_a0 * ptr_b0[j];
					val_t1 = val_a1 * ptr_b1[j];
					val_t2 = val_a2 * ptr_b2[j];
					val_t3 = val_a3 * ptr_b3[j];
					val_t0 = (val_t0 + val_t1) + (val_t2 + val_t3);
					c[j] += val_t0;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_gemmt_float<cpu_sse3>
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const __m128 zero = _mm_setzero_ps();
			__m128 xmm_t00 = zero, xmm_t01 = zero, xmm_t02 = zero, xmm_t03 = zero;
			__m128 xmm_t10 = zero, xmm_t11 = zero, xmm_t12 = zero, xmm_t13 = zero;
			__m128 xmm_t20 = zero, xmm_t21 = zero, xmm_t22 = zero, xmm_t23 = zero;
			__m128 xmm_t30 = zero, xmm_t31 = zero, xmm_t32 = zero, xmm_t33 = zero;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0 = _mm_loadu_ps(ptr_c0);
			__m128 xmm_c1 = _mm_loadu_ps(ptr_c1);
			__m128 xmm_c2 = _mm_loadu_ps(ptr_c2);
			__m128 xmm_c3 = _mm_loadu_ps(ptr_c3);

			if (aligned_p > 0)
			{
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
					xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
					xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
					xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
					xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
					// return the weighted sum
					xmm_t00 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_t00);
					xmm_t01 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b1), xmm_t01);
					xmm_t02 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b2), xmm_t02);
					xmm_t03 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b3), xmm_t03);
					xmm_t10 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b0), xmm_t10);
					xmm_t11 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b1), xmm_t11);
					xmm_t12 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b2), xmm_t12);
					xmm_t13 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b3), xmm_t13);
					xmm_t20 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b0), xmm_t20);
					xmm_t21 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b1), xmm_t21);
					xmm_t22 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b2), xmm_t22);
					xmm_t23 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b3), xmm_t23);
					xmm_t30 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b0), xmm_t30);
					xmm_t31 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b1), xmm_t31);
					xmm_t32 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b2), xmm_t32);
					xmm_t33 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b3), xmm_t33);
				}
				// return the horizontal sum
				xmm_t00 = _mm_hadd_ps(xmm_t00, xmm_t01);
				xmm_t10 = _mm_hadd_ps(xmm_t10, xmm_t11);
				xmm_t20 = _mm_hadd_ps(xmm_t20, xmm_t21);
				xmm_t30 = _mm_hadd_ps(xmm_t30, xmm_t31);
				xmm_t02 = _mm_hadd_ps(xmm_t02, xmm_t03);
				xmm_t12 = _mm_hadd_ps(xmm_t12, xmm_t13);
				xmm_t22 = _mm_hadd_ps(xmm_t22, xmm_t23);
				xmm_t32 = _mm_hadd_ps(xmm_t32, xmm_t33);
				xmm_t00 = _mm_hadd_ps(xmm_t00, xmm_t02);
				xmm_t10 = _mm_hadd_ps(xmm_t10, xmm_t12);
				xmm_t20 = _mm_hadd_ps(xmm_t20, xmm_t22);
				xmm_t30 = _mm_hadd_ps(xmm_t30, xmm_t32);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_t00);
				xmm_c1 = _mm_add_ps(xmm_c1, xmm_t10);
				xmm_c2 = _mm_add_ps(xmm_c2, xmm_t20);
				xmm_c3 = _mm_add_ps(xmm_c3, xmm_t30);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_ps(ptr_a0[k]);
					xmm_a1 = _mm_set1_ps(ptr_a1[k]);
					xmm_a2 = _mm_set1_ps(ptr_a2[k]);
					xmm_a3 = _mm_set1_ps(ptr_a3[k]);
					xmm_b0 = _mm_set_ps(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_c0);
					xmm_c1 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b0), xmm_c1);
					xmm_c2 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b0), xmm_c2);
					xmm_c3 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b0), xmm_c3);
				}
			}
			// store data into memory
			_mm_storeu_ps(ptr_c0, xmm_c0);
			_mm_storeu_ps(ptr_c1, xmm_c1);
			_mm_storeu_ps(ptr_c2, xmm_c2);
			_mm_storeu_ps(ptr_c3, xmm_c3);
		}
	};

	template<>
	struct block_gemmt_float<cpu_sse3 | cpu_fma>
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const __m128 zero = _mm_setzero_ps();
			__m128 xmm_t00 = zero, xmm_t01 = zero, xmm_t02 = zero, xmm_t03 = zero;
			__m128 xmm_t10 = zero, xmm_t11 = zero, xmm_t12 = zero, xmm_t13 = zero;
			__m128 xmm_t20 = zero, xmm_t21 = zero, xmm_t22 = zero, xmm_t23 = zero;
			__m128 xmm_t30 = zero, xmm_t31 = zero, xmm_t32 = zero, xmm_t33 = zero;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0 = _mm_loadu_ps(ptr_c0);
			__m128 xmm_c1 = _mm_loadu_ps(ptr_c1);
			__m128 xmm_c2 = _mm_loadu_ps(ptr_c2);
			__m128 xmm_c3 = _mm_loadu_ps(ptr_c3);

			if (aligned_p > 0)
			{
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
					xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
					xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
					xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
					xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
					xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
					xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
					xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
					// return the weighted sum
					xmm_t00 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_t00);
					xmm_t01 = _mm_fmadd_ps(xmm_a0, xmm_b1, xmm_t01);
					xmm_t02 = _mm_fmadd_ps(xmm_a0, xmm_b2, xmm_t02);
					xmm_t03 = _mm_fmadd_ps(xmm_a0, xmm_b3, xmm_t03);
					xmm_t10 = _mm_fmadd_ps(xmm_a1, xmm_b0, xmm_t10);
					xmm_t11 = _mm_fmadd_ps(xmm_a1, xmm_b1, xmm_t11);
					xmm_t12 = _mm_fmadd_ps(xmm_a1, xmm_b2, xmm_t12);
					xmm_t13 = _mm_fmadd_ps(xmm_a1, xmm_b3, xmm_t13);
					xmm_t20 = _mm_fmadd_ps(xmm_a2, xmm_b0, xmm_t20);
					xmm_t21 = _mm_fmadd_ps(xmm_a2, xmm_b1, xmm_t21);
					xmm_t22 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_t22);
					xmm_t23 = _mm_fmadd_ps(xmm_a2, xmm_b3, xmm_t23);
					xmm_t30 = _mm_fmadd_ps(xmm_a3, xmm_b0, xmm_t30);
					xmm_t31 = _mm_fmadd_ps(xmm_a3, xmm_b1, xmm_t31);
					xmm_t32 = _mm_fmadd_ps(xmm_a3, xmm_b2, xmm_t32);
					xmm_t33 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_t33);
				}
				// return the horizontal sum
				xmm_t00 = _mm_hadd_ps(xmm_t00, xmm_t01);
				xmm_t10 = _mm_hadd_ps(xmm_t10, xmm_t11);
				xmm_t20 = _mm_hadd_ps(xmm_t20, xmm_t21);
				xmm_t30 = _mm_hadd_ps(xmm_t30, xmm_t31);
				xmm_t02 = _mm_hadd_ps(xmm_t02, xmm_t03);
				xmm_t12 = _mm_hadd_ps(xmm_t12, xmm_t13);
				xmm_t22 = _mm_hadd_ps(xmm_t22, xmm_t23);
				xmm_t32 = _mm_hadd_ps(xmm_t32, xmm_t33);
				xmm_t00 = _mm_hadd_ps(xmm_t00, xmm_t02);
				xmm_t10 = _mm_hadd_ps(xmm_t10, xmm_t12);
				xmm_t20 = _mm_hadd_ps(xmm_t20, xmm_t22);
				xmm_t30 = _mm_hadd_ps(xmm_t30, xmm_t32);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_t00);
				xmm_c1 = _mm_add_ps(xmm_c1, xmm_t10);
				xmm_c2 = _mm_add_ps(xmm_c2, xmm_t20);
				xmm_c3 = _mm_add_ps(xmm_c3, xmm_t30);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_ps(ptr_a0[k]);
					xmm_a1 = _mm_set1_ps(ptr_a1[k]);
					xmm_a2 = _mm_set1_ps(ptr_a2[k]);
					xmm_a3 = _mm_set1_ps(ptr_a3[k]);
					xmm_b0 = _mm_set_ps(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_c0);
					xmm_c1 = _mm_fmadd_ps(xmm_a1, xmm_b0, xmm_c1);
					xmm_c2 = _mm_fmadd_ps(xmm_a2, xmm_b0, xmm_c2);
					xmm_c3 = _mm_fmadd_ps(xmm_a3, xmm_b0, xmm_c3);
				}
			}
			// store data into memory
			_mm_storeu_ps(ptr_c0, xmm_c0);
			_mm_storeu_ps(ptr_c1, xmm_c1);
			_mm_storeu_ps(ptr_c2, xmm_c2);
			_mm_storeu_ps(ptr_c3, xmm_c3);
		}
	};
/*
	template<>
	struct block_gemmt_float<cpu_avx>
	{
		// C(8x8) += A(8xp) * B(8xp)^T
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			float *ptr_c4 = ptr_c3 + rsc;
			float *ptr_c5 = ptr_c4 + rsc;
			float *ptr_c6 = ptr_c5 + rsc;
			float *ptr_c7 = ptr_c6 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			const __m256 zero = _mm256_setzero_ps();
			__m256 ymm_t00 = zero, ymm_t01 = zero, ymm_t02 = zero, ymm_t03 = zero, ymm_t04 = zero, ymm_t05 = zero, ymm_t06 = zero, ymm_t07 = zero;
			__m256 ymm_t10 = zero, ymm_t11 = zero, ymm_t12 = zero, ymm_t13 = zero, ymm_t14 = zero, ymm_t15 = zero, ymm_t16 = zero, ymm_t17 = zero;
			__m256 ymm_t20 = zero, ymm_t21 = zero, ymm_t22 = zero, ymm_t23 = zero, ymm_t24 = zero, ymm_t25 = zero, ymm_t26 = zero, ymm_t27 = zero;
			__m256 ymm_t30 = zero, ymm_t31 = zero, ymm_t32 = zero, ymm_t33 = zero, ymm_t34 = zero, ymm_t35 = zero, ymm_t36 = zero, ymm_t37 = zero;
			__m256 ymm_t40 = zero, ymm_t41 = zero, ymm_t42 = zero, ymm_t43 = zero, ymm_t44 = zero, ymm_t45 = zero, ymm_t46 = zero, ymm_t47 = zero;
			__m256 ymm_t50 = zero, ymm_t51 = zero, ymm_t52 = zero, ymm_t53 = zero, ymm_t54 = zero, ymm_t55 = zero, ymm_t56 = zero, ymm_t57 = zero;
			__m256 ymm_t60 = zero, ymm_t61 = zero, ymm_t62 = zero, ymm_t63 = zero, ymm_t64 = zero, ymm_t65 = zero, ymm_t66 = zero, ymm_t67 = zero;
			__m256 ymm_t70 = zero, ymm_t71 = zero, ymm_t72 = zero, ymm_t73 = zero, ymm_t74 = zero, ymm_t75 = zero, ymm_t76 = zero, ymm_t77 = zero;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b0 + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b1 + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b2 + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b3 + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b4 + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b5 + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b6 + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b7 + j);
						ymm_c0 = _mm256_loadu_ps(c + j);
						// return the weighted sum
						ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_t1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_t2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_t3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_t4 = _mm256_mul_ps(ymm_a4, ymm_b4);
						ymm_t5 = _mm256_mul_ps(ymm_a5, ymm_b5);
						ymm_t6 = _mm256_mul_ps(ymm_a6, ymm_b6);
						ymm_t7 = _mm256_mul_ps(ymm_a7, ymm_b7);
						ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t1);
						ymm_t2 = _mm256_add_ps(ymm_t2, ymm_t3);
						ymm_t4 = _mm256_add_ps(ymm_t4, ymm_t5);
						ymm_t6 = _mm256_add_ps(ymm_t6, ymm_t7);
						ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t2);
						ymm_t4 = _mm256_add_ps(ymm_t4, ymm_t6);
						ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t4);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t0);
						// store data into memory
						_mm256_storeu_ps(c + j, ymm_c0);
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(ptr_a[0]);
				ymm_a1 = _mm256_loadu_ps(ptr_a[1]);
				ymm_a2 = _mm256_loadu_ps(ptr_a[2]);
				ymm_a3 = _mm256_loadu_ps(ptr_a[3]);
				ymm_a4 = _mm256_loadu_ps(ptr_a[4]);
				ymm_a5 = _mm256_loadu_ps(ptr_a[5]);
				ymm_a6 = _mm256_loadu_ps(ptr_a[6]);
				ymm_a7 = _mm256_loadu_ps(ptr_a[7]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j], ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_t1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_t2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_t3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_t4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_t5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_t6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_t7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_ps(ymm_t2, ymm_t3);
					ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t5);
					ymm_t6 = _mm256_hadd_ps(ymm_t6, ymm_t7);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t2);
					ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t6);
					ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t5 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_ps(ymm_t1, ymm_t5);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_t0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_t0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_t0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_t0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_t0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_t0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_t0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_t0)[7];
				}
			}
		}
	};

	template<>
	struct block_gemmt_float<cpu_avx | cpu_fma>
	{
		// C(8xn) += A(8x8) * B(8xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c[8];
			const float *ptr_a[8];
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256 ymm_c0;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					ymm_a0 = _mm256_set1_ps(a[0]);
					ymm_a1 = _mm256_set1_ps(a[1]);
					ymm_a2 = _mm256_set1_ps(a[2]);
					ymm_a3 = _mm256_set1_ps(a[3]);
					ymm_a4 = _mm256_set1_ps(a[4]);
					ymm_a5 = _mm256_set1_ps(a[5]);
					ymm_a6 = _mm256_set1_ps(a[6]);
					ymm_a7 = _mm256_set1_ps(a[7]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b0 + j);
						ymm_b1 = _mm256_loadu_ps(ptr_b1 + j);
						ymm_b2 = _mm256_loadu_ps(ptr_b2 + j);
						ymm_b3 = _mm256_loadu_ps(ptr_b3 + j);
						ymm_b4 = _mm256_loadu_ps(ptr_b4 + j);
						ymm_b5 = _mm256_loadu_ps(ptr_b5 + j);
						ymm_b6 = _mm256_loadu_ps(ptr_b6 + j);
						ymm_b7 = _mm256_loadu_ps(ptr_b7 + j);
						ymm_c0 = _mm256_loadu_ps(c + j);
						// return the weighted sum
						ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						ymm_t1 = _mm256_mul_ps(ymm_a1, ymm_b1);
						ymm_t2 = _mm256_mul_ps(ymm_a2, ymm_b2);
						ymm_t3 = _mm256_mul_ps(ymm_a3, ymm_b3);
						ymm_t0 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_t0);
						ymm_t1 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_t1);
						ymm_t2 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_t2);
						ymm_t3 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_t3);
						ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t1);
						ymm_t2 = _mm256_add_ps(ymm_t2, ymm_t3);
						ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t2);
						ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t0);
						// store data into memory
						_mm256_storeu_ps(c + j, ymm_c0);
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(ptr_a[0]);
				ymm_a1 = _mm256_loadu_ps(ptr_a[1]);
				ymm_a2 = _mm256_loadu_ps(ptr_a[2]);
				ymm_a3 = _mm256_loadu_ps(ptr_a[3]);
				ymm_a4 = _mm256_loadu_ps(ptr_a[4]);
				ymm_a5 = _mm256_loadu_ps(ptr_a[5]);
				ymm_a6 = _mm256_loadu_ps(ptr_a[6]);
				ymm_a7 = _mm256_loadu_ps(ptr_a[7]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j], ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_t1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_t2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_t3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_t4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_t5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_t6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_t7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_ps(ymm_t2, ymm_t3);
					ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t5);
					ymm_t6 = _mm256_hadd_ps(ymm_t6, ymm_t7);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t2);
					ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t6);
					ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t5 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_ps(ymm_t1, ymm_t5);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_t0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_t0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_t0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_t0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_t0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_t0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_t0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_t0)[7];
				}
			}
		}
	};
*/
} // namespace core

#endif
