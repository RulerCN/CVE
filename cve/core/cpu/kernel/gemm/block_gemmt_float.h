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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEMMT_FLOAT_H__
#define __CORE_CPU_KERNEL_BLOCK_GEMMT_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gemmt_float
	
	template<cpu_inst_type inst>
	struct block_gemmt_float
	{
		// C(4x4) += A(4xp) * B(4xp)^T
		void operator()(size_t /*aligned_p*/, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
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
			float val_a0, val_a1, val_a2, val_a3;
			float val_b0, val_b1, val_b2, val_b3;
			float val_t00, val_t01, val_t02, val_t03;
			float val_t10, val_t11, val_t12, val_t13;
			float val_t20, val_t21, val_t22, val_t23;
			float val_t30, val_t31, val_t32, val_t33;

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
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t00, xmm_t01, xmm_t02, xmm_t03;
			__m128 xmm_t10, xmm_t11, xmm_t12, xmm_t13;
			__m128 xmm_t20, xmm_t21, xmm_t22, xmm_t23;
			__m128 xmm_t30, xmm_t31, xmm_t32, xmm_t33;
			__m128 xmm_c0 = _mm_loadu_ps(ptr_c0);
			__m128 xmm_c1 = _mm_loadu_ps(ptr_c1);
			__m128 xmm_c2 = _mm_loadu_ps(ptr_c2);
			__m128 xmm_c3 = _mm_loadu_ps(ptr_c3);

			if (aligned_p > 0)
			{
				xmm_t00 = xmm_t01 = xmm_t02 = xmm_t03 = _mm_setzero_ps();
				xmm_t10 = xmm_t11 = xmm_t12 = xmm_t13 = _mm_setzero_ps();
				xmm_t20 = xmm_t21 = xmm_t22 = xmm_t23 = _mm_setzero_ps();
				xmm_t30 = xmm_t31 = xmm_t32 = xmm_t33 = _mm_setzero_ps();
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
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t00, xmm_t01, xmm_t02, xmm_t03;
			__m128 xmm_t10, xmm_t11, xmm_t12, xmm_t13;
			__m128 xmm_t20, xmm_t21, xmm_t22, xmm_t23;
			__m128 xmm_t30, xmm_t31, xmm_t32, xmm_t33;
			__m128 xmm_c0 = _mm_loadu_ps(ptr_c0);
			__m128 xmm_c1 = _mm_loadu_ps(ptr_c1);
			__m128 xmm_c2 = _mm_loadu_ps(ptr_c2);
			__m128 xmm_c3 = _mm_loadu_ps(ptr_c3);

			if (aligned_p > 0)
			{
				xmm_t00 = xmm_t01 = xmm_t02 = xmm_t03 = _mm_setzero_ps();
				xmm_t10 = xmm_t11 = xmm_t12 = xmm_t13 = _mm_setzero_ps();
				xmm_t20 = xmm_t21 = xmm_t22 = xmm_t23 = _mm_setzero_ps();
				xmm_t30 = xmm_t31 = xmm_t32 = xmm_t33 = _mm_setzero_ps();
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

	template<>
	struct block_gemmt_float<cpu_avx>
	{
		// C(8x8) += A(8xp) * B(8xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
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
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_t00, ymm_t01, ymm_t02, ymm_t03, ymm_t04, ymm_t05, ymm_t06, ymm_t07;
			__m256 ymm_t10, ymm_t11, ymm_t12, ymm_t13, ymm_t14, ymm_t15, ymm_t16, ymm_t17;
			__m256 ymm_t20, ymm_t21, ymm_t22, ymm_t23, ymm_t24, ymm_t25, ymm_t26, ymm_t27;
			__m256 ymm_t30, ymm_t31, ymm_t32, ymm_t33, ymm_t34, ymm_t35, ymm_t36, ymm_t37;
			__m256 ymm_t40, ymm_t41, ymm_t42, ymm_t43, ymm_t44, ymm_t45, ymm_t46, ymm_t47;
			__m256 ymm_t50, ymm_t51, ymm_t52, ymm_t53, ymm_t54, ymm_t55, ymm_t56, ymm_t57;
			__m256 ymm_t60, ymm_t61, ymm_t62, ymm_t63, ymm_t64, ymm_t65, ymm_t66, ymm_t67;
			__m256 ymm_t70, ymm_t71, ymm_t72, ymm_t73, ymm_t74, ymm_t75, ymm_t76, ymm_t77;
			__m256 ymm_c0 = _mm256_loadu_ps(ptr_c0);
			__m256 ymm_c1 = _mm256_loadu_ps(ptr_c1);
			__m256 ymm_c2 = _mm256_loadu_ps(ptr_c2);
			__m256 ymm_c3 = _mm256_loadu_ps(ptr_c3);
			__m256 ymm_c4 = _mm256_loadu_ps(ptr_c4);
			__m256 ymm_c5 = _mm256_loadu_ps(ptr_c5);
			__m256 ymm_c6 = _mm256_loadu_ps(ptr_c6);
			__m256 ymm_c7 = _mm256_loadu_ps(ptr_c7);

			if (aligned_p > 0)
			{
				ymm_t00 = ymm_t01 = ymm_t02 = ymm_t03 = ymm_t04 = ymm_t05 = ymm_t06 = ymm_t07 = _mm256_setzero_ps();
				ymm_t10 = ymm_t11 = ymm_t12 = ymm_t13 = ymm_t14 = ymm_t15 = ymm_t16 = ymm_t17 = _mm256_setzero_ps();
				ymm_t20 = ymm_t21 = ymm_t22 = ymm_t23 = ymm_t24 = ymm_t25 = ymm_t26 = ymm_t27 = _mm256_setzero_ps();
				ymm_t30 = ymm_t31 = ymm_t32 = ymm_t33 = ymm_t34 = ymm_t35 = ymm_t36 = ymm_t37 = _mm256_setzero_ps();
				ymm_t40 = ymm_t41 = ymm_t42 = ymm_t43 = ymm_t44 = ymm_t45 = ymm_t46 = ymm_t47 = _mm256_setzero_ps();
				ymm_t50 = ymm_t51 = ymm_t52 = ymm_t53 = ymm_t54 = ymm_t55 = ymm_t56 = ymm_t57 = _mm256_setzero_ps();
				ymm_t60 = ymm_t61 = ymm_t62 = ymm_t63 = ymm_t64 = ymm_t65 = ymm_t66 = ymm_t67 = _mm256_setzero_ps();
				ymm_t70 = ymm_t71 = ymm_t72 = ymm_t73 = ymm_t74 = ymm_t75 = ymm_t76 = ymm_t77 = _mm256_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 8)
				{
					ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
					ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
					ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
					ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
					ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
					ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
					ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
					ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
					ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
					ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
					ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
					ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
					ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
					// return the weighted sum
					ymm_t00 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_t00);
					ymm_t01 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b1), ymm_t01);
					ymm_t02 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b2), ymm_t02);
					ymm_t03 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b3), ymm_t03);
					ymm_t04 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b4), ymm_t04);
					ymm_t05 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b5), ymm_t05);
					ymm_t06 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b6), ymm_t06);
					ymm_t07 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b7), ymm_t07);
					ymm_t10 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b0), ymm_t10);
					ymm_t11 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b1), ymm_t11);
					ymm_t12 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b2), ymm_t12);
					ymm_t13 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b3), ymm_t13);
					ymm_t14 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b4), ymm_t14);
					ymm_t15 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b5), ymm_t15);
					ymm_t16 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b6), ymm_t16);
					ymm_t17 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b7), ymm_t17);
					ymm_t20 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b0), ymm_t20);
					ymm_t21 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b1), ymm_t21);
					ymm_t22 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b2), ymm_t22);
					ymm_t23 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b3), ymm_t23);
					ymm_t24 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b4), ymm_t24);
					ymm_t25 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b5), ymm_t25);
					ymm_t26 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b6), ymm_t26);
					ymm_t27 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b7), ymm_t27);
					ymm_t30 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b0), ymm_t30);
					ymm_t31 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b1), ymm_t31);
					ymm_t32 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b2), ymm_t32);
					ymm_t33 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b3), ymm_t33);
					ymm_t34 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b4), ymm_t34);
					ymm_t35 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b5), ymm_t35);
					ymm_t36 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b6), ymm_t36);
					ymm_t37 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b7), ymm_t37);
					ymm_t40 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b0), ymm_t40);
					ymm_t41 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b1), ymm_t41);
					ymm_t42 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b2), ymm_t42);
					ymm_t43 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b3), ymm_t43);
					ymm_t44 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b4), ymm_t44);
					ymm_t45 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b5), ymm_t45);
					ymm_t46 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b6), ymm_t46);
					ymm_t47 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b7), ymm_t47);
					ymm_t50 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b0), ymm_t50);
					ymm_t51 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b1), ymm_t51);
					ymm_t52 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b2), ymm_t52);
					ymm_t53 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b3), ymm_t53);
					ymm_t54 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b4), ymm_t54);
					ymm_t55 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b5), ymm_t55);
					ymm_t56 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b6), ymm_t56);
					ymm_t57 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b7), ymm_t57);
					ymm_t60 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b0), ymm_t60);
					ymm_t61 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b1), ymm_t61);
					ymm_t62 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b2), ymm_t62);
					ymm_t63 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b3), ymm_t63);
					ymm_t64 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b4), ymm_t64);
					ymm_t65 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b5), ymm_t65);
					ymm_t66 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b6), ymm_t66);
					ymm_t67 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b7), ymm_t67);
					ymm_t70 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b0), ymm_t70);
					ymm_t71 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b1), ymm_t71);
					ymm_t72 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b2), ymm_t72);
					ymm_t73 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b3), ymm_t73);
					ymm_t74 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b4), ymm_t74);
					ymm_t75 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b5), ymm_t75);
					ymm_t76 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b6), ymm_t76);
					ymm_t77 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b7), ymm_t77);
				}
				// return the horizontal sum
				ymm_t00 = _mm256_hadd_ps(ymm_t00, ymm_t01);
				ymm_t10 = _mm256_hadd_ps(ymm_t10, ymm_t11);
				ymm_t20 = _mm256_hadd_ps(ymm_t20, ymm_t21);
				ymm_t30 = _mm256_hadd_ps(ymm_t30, ymm_t31);
				ymm_t40 = _mm256_hadd_ps(ymm_t40, ymm_t41);
				ymm_t50 = _mm256_hadd_ps(ymm_t50, ymm_t51);
				ymm_t60 = _mm256_hadd_ps(ymm_t60, ymm_t61);
				ymm_t70 = _mm256_hadd_ps(ymm_t70, ymm_t71);
				ymm_t02 = _mm256_hadd_ps(ymm_t02, ymm_t03);
				ymm_t12 = _mm256_hadd_ps(ymm_t12, ymm_t13);
				ymm_t22 = _mm256_hadd_ps(ymm_t22, ymm_t23);
				ymm_t32 = _mm256_hadd_ps(ymm_t32, ymm_t33);
				ymm_t42 = _mm256_hadd_ps(ymm_t42, ymm_t43);
				ymm_t52 = _mm256_hadd_ps(ymm_t52, ymm_t53);
				ymm_t62 = _mm256_hadd_ps(ymm_t62, ymm_t63);
				ymm_t72 = _mm256_hadd_ps(ymm_t72, ymm_t73);
				ymm_t04 = _mm256_hadd_ps(ymm_t04, ymm_t05);
				ymm_t14 = _mm256_hadd_ps(ymm_t14, ymm_t15);
				ymm_t24 = _mm256_hadd_ps(ymm_t24, ymm_t25);
				ymm_t34 = _mm256_hadd_ps(ymm_t34, ymm_t35);
				ymm_t44 = _mm256_hadd_ps(ymm_t44, ymm_t45);
				ymm_t54 = _mm256_hadd_ps(ymm_t54, ymm_t55);
				ymm_t64 = _mm256_hadd_ps(ymm_t64, ymm_t65);
				ymm_t74 = _mm256_hadd_ps(ymm_t74, ymm_t75);
				ymm_t06 = _mm256_hadd_ps(ymm_t06, ymm_t07);
				ymm_t16 = _mm256_hadd_ps(ymm_t16, ymm_t17);
				ymm_t26 = _mm256_hadd_ps(ymm_t26, ymm_t27);
				ymm_t36 = _mm256_hadd_ps(ymm_t36, ymm_t37);
				ymm_t46 = _mm256_hadd_ps(ymm_t46, ymm_t47);
				ymm_t56 = _mm256_hadd_ps(ymm_t56, ymm_t57);
				ymm_t66 = _mm256_hadd_ps(ymm_t66, ymm_t67);
				ymm_t76 = _mm256_hadd_ps(ymm_t76, ymm_t77);
				ymm_t00 = _mm256_hadd_ps(ymm_t00, ymm_t02);
				ymm_t10 = _mm256_hadd_ps(ymm_t10, ymm_t12);
				ymm_t20 = _mm256_hadd_ps(ymm_t20, ymm_t22);
				ymm_t30 = _mm256_hadd_ps(ymm_t30, ymm_t32);
				ymm_t40 = _mm256_hadd_ps(ymm_t40, ymm_t42);
				ymm_t50 = _mm256_hadd_ps(ymm_t50, ymm_t52);
				ymm_t60 = _mm256_hadd_ps(ymm_t60, ymm_t62);
				ymm_t70 = _mm256_hadd_ps(ymm_t70, ymm_t72);
				ymm_t04 = _mm256_hadd_ps(ymm_t04, ymm_t06);
				ymm_t14 = _mm256_hadd_ps(ymm_t14, ymm_t16);
				ymm_t24 = _mm256_hadd_ps(ymm_t24, ymm_t26);
				ymm_t34 = _mm256_hadd_ps(ymm_t34, ymm_t36);
				ymm_t44 = _mm256_hadd_ps(ymm_t44, ymm_t46);
				ymm_t54 = _mm256_hadd_ps(ymm_t54, ymm_t56);
				ymm_t64 = _mm256_hadd_ps(ymm_t64, ymm_t66);
				ymm_t74 = _mm256_hadd_ps(ymm_t74, ymm_t76);
				ymm_t01 = _mm256_permute2f128_ps(ymm_t00, ymm_t04, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t11 = _mm256_permute2f128_ps(ymm_t10, ymm_t14, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t21 = _mm256_permute2f128_ps(ymm_t20, ymm_t24, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t31 = _mm256_permute2f128_ps(ymm_t30, ymm_t34, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t41 = _mm256_permute2f128_ps(ymm_t40, ymm_t44, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t51 = _mm256_permute2f128_ps(ymm_t50, ymm_t54, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t61 = _mm256_permute2f128_ps(ymm_t60, ymm_t64, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t71 = _mm256_permute2f128_ps(ymm_t70, ymm_t74, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t05 = _mm256_permute2f128_ps(ymm_t00, ymm_t04, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t15 = _mm256_permute2f128_ps(ymm_t10, ymm_t14, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t25 = _mm256_permute2f128_ps(ymm_t20, ymm_t24, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t35 = _mm256_permute2f128_ps(ymm_t30, ymm_t34, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t45 = _mm256_permute2f128_ps(ymm_t40, ymm_t44, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t55 = _mm256_permute2f128_ps(ymm_t50, ymm_t54, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t65 = _mm256_permute2f128_ps(ymm_t60, ymm_t64, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t75 = _mm256_permute2f128_ps(ymm_t70, ymm_t74, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t00 = _mm256_add_ps(ymm_t01, ymm_t05);
				ymm_t10 = _mm256_add_ps(ymm_t11, ymm_t15);
				ymm_t20 = _mm256_add_ps(ymm_t21, ymm_t25);
				ymm_t30 = _mm256_add_ps(ymm_t31, ymm_t35);
				ymm_t40 = _mm256_add_ps(ymm_t41, ymm_t45);
				ymm_t50 = _mm256_add_ps(ymm_t51, ymm_t55);
				ymm_t60 = _mm256_add_ps(ymm_t61, ymm_t65);
				ymm_t70 = _mm256_add_ps(ymm_t71, ymm_t75);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t00);
				ymm_c1 = _mm256_add_ps(ymm_c1, ymm_t10);
				ymm_c2 = _mm256_add_ps(ymm_c2, ymm_t20);
				ymm_c3 = _mm256_add_ps(ymm_c3, ymm_t30);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_t40);
				ymm_c5 = _mm256_add_ps(ymm_c5, ymm_t50);
				ymm_c6 = _mm256_add_ps(ymm_c6, ymm_t60);
				ymm_c7 = _mm256_add_ps(ymm_c7, ymm_t70);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_ps(ptr_a0[k]);
					ymm_a1 = _mm256_set1_ps(ptr_a1[k]);
					ymm_a2 = _mm256_set1_ps(ptr_a2[k]);
					ymm_a3 = _mm256_set1_ps(ptr_a3[k]);
					ymm_a4 = _mm256_set1_ps(ptr_a4[k]);
					ymm_a5 = _mm256_set1_ps(ptr_a5[k]);
					ymm_a6 = _mm256_set1_ps(ptr_a6[k]);
					ymm_a7 = _mm256_set1_ps(ptr_a7[k]);
					ymm_b0 = _mm256_set_ps(ptr_b7[k], ptr_b6[k], ptr_b5[k], ptr_b4[k], ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_c0);
					ymm_c1 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b0), ymm_c1);
					ymm_c2 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b0), ymm_c2);
					ymm_c3 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b0), ymm_c3);
					ymm_c4 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b0), ymm_c4);
					ymm_c5 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b0), ymm_c5);
					ymm_c6 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b0), ymm_c6);
					ymm_c7 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b0), ymm_c7);
				}
			}
			// store data into memory
			_mm256_storeu_ps(ptr_c0, ymm_c0);
			_mm256_storeu_ps(ptr_c1, ymm_c1);
			_mm256_storeu_ps(ptr_c2, ymm_c2);
			_mm256_storeu_ps(ptr_c3, ymm_c3);
			_mm256_storeu_ps(ptr_c4, ymm_c4);
			_mm256_storeu_ps(ptr_c5, ymm_c5);
			_mm256_storeu_ps(ptr_c6, ymm_c6);
			_mm256_storeu_ps(ptr_c7, ymm_c7);
		}
	};

	template<>
	struct block_gemmt_float<cpu_avx | cpu_fma>
	{
		// C(8x8) += A(8xp) * B(8xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
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
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_t00, ymm_t01, ymm_t02, ymm_t03, ymm_t04, ymm_t05, ymm_t06, ymm_t07;
			__m256 ymm_t10, ymm_t11, ymm_t12, ymm_t13, ymm_t14, ymm_t15, ymm_t16, ymm_t17;
			__m256 ymm_t20, ymm_t21, ymm_t22, ymm_t23, ymm_t24, ymm_t25, ymm_t26, ymm_t27;
			__m256 ymm_t30, ymm_t31, ymm_t32, ymm_t33, ymm_t34, ymm_t35, ymm_t36, ymm_t37;
			__m256 ymm_t40, ymm_t41, ymm_t42, ymm_t43, ymm_t44, ymm_t45, ymm_t46, ymm_t47;
			__m256 ymm_t50, ymm_t51, ymm_t52, ymm_t53, ymm_t54, ymm_t55, ymm_t56, ymm_t57;
			__m256 ymm_t60, ymm_t61, ymm_t62, ymm_t63, ymm_t64, ymm_t65, ymm_t66, ymm_t67;
			__m256 ymm_t70, ymm_t71, ymm_t72, ymm_t73, ymm_t74, ymm_t75, ymm_t76, ymm_t77;
			__m256 ymm_c0 = _mm256_loadu_ps(ptr_c0);
			__m256 ymm_c1 = _mm256_loadu_ps(ptr_c1);
			__m256 ymm_c2 = _mm256_loadu_ps(ptr_c2);
			__m256 ymm_c3 = _mm256_loadu_ps(ptr_c3);
			__m256 ymm_c4 = _mm256_loadu_ps(ptr_c4);
			__m256 ymm_c5 = _mm256_loadu_ps(ptr_c5);
			__m256 ymm_c6 = _mm256_loadu_ps(ptr_c6);
			__m256 ymm_c7 = _mm256_loadu_ps(ptr_c7);

			if (aligned_p > 0)
			{
				ymm_t00 = ymm_t01 = ymm_t02 = ymm_t03 = ymm_t04 = ymm_t05 = ymm_t06 = ymm_t07 = _mm256_setzero_ps();
				ymm_t10 = ymm_t11 = ymm_t12 = ymm_t13 = ymm_t14 = ymm_t15 = ymm_t16 = ymm_t17 = _mm256_setzero_ps();
				ymm_t20 = ymm_t21 = ymm_t22 = ymm_t23 = ymm_t24 = ymm_t25 = ymm_t26 = ymm_t27 = _mm256_setzero_ps();
				ymm_t30 = ymm_t31 = ymm_t32 = ymm_t33 = ymm_t34 = ymm_t35 = ymm_t36 = ymm_t37 = _mm256_setzero_ps();
				ymm_t40 = ymm_t41 = ymm_t42 = ymm_t43 = ymm_t44 = ymm_t45 = ymm_t46 = ymm_t47 = _mm256_setzero_ps();
				ymm_t50 = ymm_t51 = ymm_t52 = ymm_t53 = ymm_t54 = ymm_t55 = ymm_t56 = ymm_t57 = _mm256_setzero_ps();
				ymm_t60 = ymm_t61 = ymm_t62 = ymm_t63 = ymm_t64 = ymm_t65 = ymm_t66 = ymm_t67 = _mm256_setzero_ps();
				ymm_t70 = ymm_t71 = ymm_t72 = ymm_t73 = ymm_t74 = ymm_t75 = ymm_t76 = ymm_t77 = _mm256_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 8)
				{
					ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
					ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
					ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
					ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
					ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
					ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
					ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
					ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
					ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
					ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
					ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
					ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
					ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
					// return the weighted sum
					ymm_t00 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_t00);
					ymm_t01 = _mm256_fmadd_ps(ymm_a0, ymm_b1, ymm_t01);
					ymm_t02 = _mm256_fmadd_ps(ymm_a0, ymm_b2, ymm_t02);
					ymm_t03 = _mm256_fmadd_ps(ymm_a0, ymm_b3, ymm_t03);
					ymm_t04 = _mm256_fmadd_ps(ymm_a0, ymm_b4, ymm_t04);
					ymm_t05 = _mm256_fmadd_ps(ymm_a0, ymm_b5, ymm_t05);
					ymm_t06 = _mm256_fmadd_ps(ymm_a0, ymm_b6, ymm_t06);
					ymm_t07 = _mm256_fmadd_ps(ymm_a0, ymm_b7, ymm_t07);
					ymm_t10 = _mm256_fmadd_ps(ymm_a1, ymm_b0, ymm_t10);
					ymm_t11 = _mm256_fmadd_ps(ymm_a1, ymm_b1, ymm_t11);
					ymm_t12 = _mm256_fmadd_ps(ymm_a1, ymm_b2, ymm_t12);
					ymm_t13 = _mm256_fmadd_ps(ymm_a1, ymm_b3, ymm_t13);
					ymm_t14 = _mm256_fmadd_ps(ymm_a1, ymm_b4, ymm_t14);
					ymm_t15 = _mm256_fmadd_ps(ymm_a1, ymm_b5, ymm_t15);
					ymm_t16 = _mm256_fmadd_ps(ymm_a1, ymm_b6, ymm_t16);
					ymm_t17 = _mm256_fmadd_ps(ymm_a1, ymm_b7, ymm_t17);
					ymm_t20 = _mm256_fmadd_ps(ymm_a2, ymm_b0, ymm_t20);
					ymm_t21 = _mm256_fmadd_ps(ymm_a2, ymm_b1, ymm_t21);
					ymm_t22 = _mm256_fmadd_ps(ymm_a2, ymm_b2, ymm_t22);
					ymm_t23 = _mm256_fmadd_ps(ymm_a2, ymm_b3, ymm_t23);
					ymm_t24 = _mm256_fmadd_ps(ymm_a2, ymm_b4, ymm_t24);
					ymm_t25 = _mm256_fmadd_ps(ymm_a2, ymm_b5, ymm_t25);
					ymm_t26 = _mm256_fmadd_ps(ymm_a2, ymm_b6, ymm_t26);
					ymm_t27 = _mm256_fmadd_ps(ymm_a2, ymm_b7, ymm_t27);
					ymm_t30 = _mm256_fmadd_ps(ymm_a3, ymm_b0, ymm_t30);
					ymm_t31 = _mm256_fmadd_ps(ymm_a3, ymm_b1, ymm_t31);
					ymm_t32 = _mm256_fmadd_ps(ymm_a3, ymm_b2, ymm_t32);
					ymm_t33 = _mm256_fmadd_ps(ymm_a3, ymm_b3, ymm_t33);
					ymm_t34 = _mm256_fmadd_ps(ymm_a3, ymm_b4, ymm_t34);
					ymm_t35 = _mm256_fmadd_ps(ymm_a3, ymm_b5, ymm_t35);
					ymm_t36 = _mm256_fmadd_ps(ymm_a3, ymm_b6, ymm_t36);
					ymm_t37 = _mm256_fmadd_ps(ymm_a3, ymm_b7, ymm_t37);
					ymm_t40 = _mm256_fmadd_ps(ymm_a4, ymm_b0, ymm_t40);
					ymm_t41 = _mm256_fmadd_ps(ymm_a4, ymm_b1, ymm_t41);
					ymm_t42 = _mm256_fmadd_ps(ymm_a4, ymm_b2, ymm_t42);
					ymm_t43 = _mm256_fmadd_ps(ymm_a4, ymm_b3, ymm_t43);
					ymm_t44 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_t44);
					ymm_t45 = _mm256_fmadd_ps(ymm_a4, ymm_b5, ymm_t45);
					ymm_t46 = _mm256_fmadd_ps(ymm_a4, ymm_b6, ymm_t46);
					ymm_t47 = _mm256_fmadd_ps(ymm_a4, ymm_b7, ymm_t47);
					ymm_t50 = _mm256_fmadd_ps(ymm_a5, ymm_b0, ymm_t50);
					ymm_t51 = _mm256_fmadd_ps(ymm_a5, ymm_b1, ymm_t51);
					ymm_t52 = _mm256_fmadd_ps(ymm_a5, ymm_b2, ymm_t52);
					ymm_t53 = _mm256_fmadd_ps(ymm_a5, ymm_b3, ymm_t53);
					ymm_t54 = _mm256_fmadd_ps(ymm_a5, ymm_b4, ymm_t54);
					ymm_t55 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_t55);
					ymm_t56 = _mm256_fmadd_ps(ymm_a5, ymm_b6, ymm_t56);
					ymm_t57 = _mm256_fmadd_ps(ymm_a5, ymm_b7, ymm_t57);
					ymm_t60 = _mm256_fmadd_ps(ymm_a6, ymm_b0, ymm_t60);
					ymm_t61 = _mm256_fmadd_ps(ymm_a6, ymm_b1, ymm_t61);
					ymm_t62 = _mm256_fmadd_ps(ymm_a6, ymm_b2, ymm_t62);
					ymm_t63 = _mm256_fmadd_ps(ymm_a6, ymm_b3, ymm_t63);
					ymm_t64 = _mm256_fmadd_ps(ymm_a6, ymm_b4, ymm_t64);
					ymm_t65 = _mm256_fmadd_ps(ymm_a6, ymm_b5, ymm_t65);
					ymm_t66 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_t66);
					ymm_t67 = _mm256_fmadd_ps(ymm_a6, ymm_b7, ymm_t67);
					ymm_t70 = _mm256_fmadd_ps(ymm_a7, ymm_b0, ymm_t70);
					ymm_t71 = _mm256_fmadd_ps(ymm_a7, ymm_b1, ymm_t71);
					ymm_t72 = _mm256_fmadd_ps(ymm_a7, ymm_b2, ymm_t72);
					ymm_t73 = _mm256_fmadd_ps(ymm_a7, ymm_b3, ymm_t73);
					ymm_t74 = _mm256_fmadd_ps(ymm_a7, ymm_b4, ymm_t74);
					ymm_t75 = _mm256_fmadd_ps(ymm_a7, ymm_b5, ymm_t75);
					ymm_t76 = _mm256_fmadd_ps(ymm_a7, ymm_b6, ymm_t76);
					ymm_t77 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_t77);
				}
				// return the horizontal sum
				ymm_t00 = _mm256_hadd_ps(ymm_t00, ymm_t01);
				ymm_t10 = _mm256_hadd_ps(ymm_t10, ymm_t11);
				ymm_t20 = _mm256_hadd_ps(ymm_t20, ymm_t21);
				ymm_t30 = _mm256_hadd_ps(ymm_t30, ymm_t31);
				ymm_t40 = _mm256_hadd_ps(ymm_t40, ymm_t41);
				ymm_t50 = _mm256_hadd_ps(ymm_t50, ymm_t51);
				ymm_t60 = _mm256_hadd_ps(ymm_t60, ymm_t61);
				ymm_t70 = _mm256_hadd_ps(ymm_t70, ymm_t71);
				ymm_t02 = _mm256_hadd_ps(ymm_t02, ymm_t03);
				ymm_t12 = _mm256_hadd_ps(ymm_t12, ymm_t13);
				ymm_t22 = _mm256_hadd_ps(ymm_t22, ymm_t23);
				ymm_t32 = _mm256_hadd_ps(ymm_t32, ymm_t33);
				ymm_t42 = _mm256_hadd_ps(ymm_t42, ymm_t43);
				ymm_t52 = _mm256_hadd_ps(ymm_t52, ymm_t53);
				ymm_t62 = _mm256_hadd_ps(ymm_t62, ymm_t63);
				ymm_t72 = _mm256_hadd_ps(ymm_t72, ymm_t73);
				ymm_t04 = _mm256_hadd_ps(ymm_t04, ymm_t05);
				ymm_t14 = _mm256_hadd_ps(ymm_t14, ymm_t15);
				ymm_t24 = _mm256_hadd_ps(ymm_t24, ymm_t25);
				ymm_t34 = _mm256_hadd_ps(ymm_t34, ymm_t35);
				ymm_t44 = _mm256_hadd_ps(ymm_t44, ymm_t45);
				ymm_t54 = _mm256_hadd_ps(ymm_t54, ymm_t55);
				ymm_t64 = _mm256_hadd_ps(ymm_t64, ymm_t65);
				ymm_t74 = _mm256_hadd_ps(ymm_t74, ymm_t75);
				ymm_t06 = _mm256_hadd_ps(ymm_t06, ymm_t07);
				ymm_t16 = _mm256_hadd_ps(ymm_t16, ymm_t17);
				ymm_t26 = _mm256_hadd_ps(ymm_t26, ymm_t27);
				ymm_t36 = _mm256_hadd_ps(ymm_t36, ymm_t37);
				ymm_t46 = _mm256_hadd_ps(ymm_t46, ymm_t47);
				ymm_t56 = _mm256_hadd_ps(ymm_t56, ymm_t57);
				ymm_t66 = _mm256_hadd_ps(ymm_t66, ymm_t67);
				ymm_t76 = _mm256_hadd_ps(ymm_t76, ymm_t77);
				ymm_t00 = _mm256_hadd_ps(ymm_t00, ymm_t02);
				ymm_t10 = _mm256_hadd_ps(ymm_t10, ymm_t12);
				ymm_t20 = _mm256_hadd_ps(ymm_t20, ymm_t22);
				ymm_t30 = _mm256_hadd_ps(ymm_t30, ymm_t32);
				ymm_t40 = _mm256_hadd_ps(ymm_t40, ymm_t42);
				ymm_t50 = _mm256_hadd_ps(ymm_t50, ymm_t52);
				ymm_t60 = _mm256_hadd_ps(ymm_t60, ymm_t62);
				ymm_t70 = _mm256_hadd_ps(ymm_t70, ymm_t72);
				ymm_t04 = _mm256_hadd_ps(ymm_t04, ymm_t06);
				ymm_t14 = _mm256_hadd_ps(ymm_t14, ymm_t16);
				ymm_t24 = _mm256_hadd_ps(ymm_t24, ymm_t26);
				ymm_t34 = _mm256_hadd_ps(ymm_t34, ymm_t36);
				ymm_t44 = _mm256_hadd_ps(ymm_t44, ymm_t46);
				ymm_t54 = _mm256_hadd_ps(ymm_t54, ymm_t56);
				ymm_t64 = _mm256_hadd_ps(ymm_t64, ymm_t66);
				ymm_t74 = _mm256_hadd_ps(ymm_t74, ymm_t76);
				ymm_t01 = _mm256_permute2f128_ps(ymm_t00, ymm_t04, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t11 = _mm256_permute2f128_ps(ymm_t10, ymm_t14, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t21 = _mm256_permute2f128_ps(ymm_t20, ymm_t24, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t31 = _mm256_permute2f128_ps(ymm_t30, ymm_t34, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t41 = _mm256_permute2f128_ps(ymm_t40, ymm_t44, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t51 = _mm256_permute2f128_ps(ymm_t50, ymm_t54, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t61 = _mm256_permute2f128_ps(ymm_t60, ymm_t64, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t71 = _mm256_permute2f128_ps(ymm_t70, ymm_t74, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t05 = _mm256_permute2f128_ps(ymm_t00, ymm_t04, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t15 = _mm256_permute2f128_ps(ymm_t10, ymm_t14, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t25 = _mm256_permute2f128_ps(ymm_t20, ymm_t24, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t35 = _mm256_permute2f128_ps(ymm_t30, ymm_t34, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t45 = _mm256_permute2f128_ps(ymm_t40, ymm_t44, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t55 = _mm256_permute2f128_ps(ymm_t50, ymm_t54, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t65 = _mm256_permute2f128_ps(ymm_t60, ymm_t64, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t75 = _mm256_permute2f128_ps(ymm_t70, ymm_t74, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t00 = _mm256_add_ps(ymm_t01, ymm_t05);
				ymm_t10 = _mm256_add_ps(ymm_t11, ymm_t15);
				ymm_t20 = _mm256_add_ps(ymm_t21, ymm_t25);
				ymm_t30 = _mm256_add_ps(ymm_t31, ymm_t35);
				ymm_t40 = _mm256_add_ps(ymm_t41, ymm_t45);
				ymm_t50 = _mm256_add_ps(ymm_t51, ymm_t55);
				ymm_t60 = _mm256_add_ps(ymm_t61, ymm_t65);
				ymm_t70 = _mm256_add_ps(ymm_t71, ymm_t75);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t00);
				ymm_c1 = _mm256_add_ps(ymm_c1, ymm_t10);
				ymm_c2 = _mm256_add_ps(ymm_c2, ymm_t20);
				ymm_c3 = _mm256_add_ps(ymm_c3, ymm_t30);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_t40);
				ymm_c5 = _mm256_add_ps(ymm_c5, ymm_t50);
				ymm_c6 = _mm256_add_ps(ymm_c6, ymm_t60);
				ymm_c7 = _mm256_add_ps(ymm_c7, ymm_t70);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_ps(ptr_a0[k]);
					ymm_a1 = _mm256_set1_ps(ptr_a1[k]);
					ymm_a2 = _mm256_set1_ps(ptr_a2[k]);
					ymm_a3 = _mm256_set1_ps(ptr_a3[k]);
					ymm_a4 = _mm256_set1_ps(ptr_a4[k]);
					ymm_a5 = _mm256_set1_ps(ptr_a5[k]);
					ymm_a6 = _mm256_set1_ps(ptr_a6[k]);
					ymm_a7 = _mm256_set1_ps(ptr_a7[k]);
					ymm_b0 = _mm256_set_ps(ptr_b7[k], ptr_b6[k], ptr_b5[k], ptr_b4[k], ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_c0);
					ymm_c1 = _mm256_fmadd_ps(ymm_a1, ymm_b0, ymm_c1);
					ymm_c2 = _mm256_fmadd_ps(ymm_a2, ymm_b0, ymm_c2);
					ymm_c3 = _mm256_fmadd_ps(ymm_a3, ymm_b0, ymm_c3);
					ymm_c4 = _mm256_fmadd_ps(ymm_a4, ymm_b0, ymm_c4);
					ymm_c5 = _mm256_fmadd_ps(ymm_a5, ymm_b0, ymm_c5);
					ymm_c6 = _mm256_fmadd_ps(ymm_a6, ymm_b0, ymm_c6);
					ymm_c7 = _mm256_fmadd_ps(ymm_a7, ymm_b0, ymm_c7);
				}
			}
			// store data into memory
			_mm256_storeu_ps(ptr_c0, ymm_c0);
			_mm256_storeu_ps(ptr_c1, ymm_c1);
			_mm256_storeu_ps(ptr_c2, ymm_c2);
			_mm256_storeu_ps(ptr_c3, ymm_c3);
			_mm256_storeu_ps(ptr_c4, ymm_c4);
			_mm256_storeu_ps(ptr_c5, ymm_c5);
			_mm256_storeu_ps(ptr_c6, ymm_c6);
			_mm256_storeu_ps(ptr_c7, ymm_c7);
		}
	};

} // namespace core

#endif
