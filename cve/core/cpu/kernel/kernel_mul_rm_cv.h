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

#ifndef __CORE_CPU_KERNEL_MUL_RM_CV_H__
#define __CORE_CPU_KERNEL_MUL_RM_CV_H__

#include "../cpu.h"

namespace core
{
	// Class template common_mul_rm_cv
	template<class T>
	struct common_mul_rm_cv
	{
		// C(mx1) += A(mxp) * B(1xp)^T
		void operator()(size_t m, size_t p, const T *a, size_t rsa, const T *b, T *c, size_t rsc) const
		{
			T val_c;
			for (size_t i = 0; i < m; ++i)
			{
				val_c = 0;
				for (size_t k = 0; k < p; ++k)
					val_c += a[k] * b[k];
				*c += val_c;
				a += rsa;
				c += rsc;
			}
		}
	};

	// Class template block_mul_rm_cv
	template<class T, cpu_inst_type inst>
	struct block_mul_rm_cv
	{
		// C(4x1) += A(4xp) * B(1xp)^T
		void operator()(size_t p, const T *a, size_t rsa, const T *b, T *c, size_t rsc) const
		{
			const T *ptr_a0 = a;
			const T *ptr_a1 = ptr_a0 + rsa;
			const T *ptr_a2 = ptr_a1 + rsa;
			const T *ptr_a3 = ptr_a2 + rsa;
			float *ptr_c0 = c;
			float *ptr_c1 = ptr_c0 + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			T val_b0, val_b1, val_b2, val_b3;
			T val_c0 = 0, val_c1 = 0, val_c2 = 0, val_c3 = 0;

			for (size_t k = 0; k < p; k += 4)
			{
				val_b0 = b[0];
				val_b1 = b[1];
				val_b2 = b[2];
				val_b3 = b[3];
				val_c0 += ptr_a0[0] * val_b0;
				val_c1 += ptr_a1[0] * val_b0;
				val_c2 += ptr_a2[0] * val_b0;
				val_c3 += ptr_a3[0] * val_b0;
				val_c0 += ptr_a0[1] * val_b1;
				val_c1 += ptr_a1[1] * val_b1;
				val_c2 += ptr_a2[1] * val_b1;
				val_c3 += ptr_a3[1] * val_b1;
				val_c0 += ptr_a0[2] * val_b2;
				val_c1 += ptr_a1[2] * val_b2;
				val_c2 += ptr_a2[2] * val_b2;
				val_c3 += ptr_a3[2] * val_b2;
				val_c0 += ptr_a0[3] * val_b3;
				val_c1 += ptr_a1[3] * val_b3;
				val_c2 += ptr_a2[3] * val_b3;
				val_c3 += ptr_a3[3] * val_b3;
				ptr_a0 += 4;
				ptr_a1 += 4;
				ptr_a2 += 4;
				ptr_a3 += 4;
				b += 4;
			}
			*ptr_c0 += val_c0;
			*ptr_c1 += val_c1;
			*ptr_c2 += val_c2;
			*ptr_c3 += val_c3;
		}
	};

	template<>
	struct block_mul_rm_cv<float, cpu_sse3>
	{
		// C(4x1) += A(4xp) * B(1xp)^T
		void operator()(size_t p, const float *a, size_t rsa, const float *b, float *c, size_t rsc) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			float *ptr_c0 = c;
			float *ptr_c1 = ptr_c0 + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b;
			__m128 xmm_c0 = _mm_setzero_ps();
			__m128 xmm_c1 = _mm_setzero_ps();
			__m128 xmm_c2 = _mm_setzero_ps();
			__m128 xmm_c3 = _mm_setzero_ps();

			for (size_t k = 0; k < p; k += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
				xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
				xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
				xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
				xmm_b = _mm_loadu_ps(b + k);
				// return the weighted sum
				xmm_c0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b), xmm_c0);
				xmm_c1 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b), xmm_c1);
				xmm_c2 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b), xmm_c2);
				xmm_c3 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b), xmm_c3);
			}
			// return the horizontal sum
			xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_ps(xmm_c1, xmm_c1);
			xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c2);
			xmm_c3 = _mm_hadd_ps(xmm_c3, xmm_c3);
			xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_ps(xmm_c1, xmm_c1);
			xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c2);
			xmm_c3 = _mm_hadd_ps(xmm_c3, xmm_c3);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<float*>(&xmm_c0);
			*ptr_c1 += *reinterpret_cast<float*>(&xmm_c1);
			*ptr_c2 += *reinterpret_cast<float*>(&xmm_c2);
			*ptr_c3 += *reinterpret_cast<float*>(&xmm_c3);
		}
	};

	template<>
	struct block_mul_rm_cv<float, cpu_sse3 | cpu_fma>
	{
		// C(4x1) += A(4xp) * B(1xp)^T
		void operator()(size_t p, const float *a, size_t rsa, const float *b, float *c, size_t rsc) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			float *ptr_c0 = c;
			float *ptr_c1 = ptr_c0 + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b;
			__m128 xmm_c0 = _mm_setzero_ps();
			__m128 xmm_c1 = _mm_setzero_ps();
			__m128 xmm_c2 = _mm_setzero_ps();
			__m128 xmm_c3 = _mm_setzero_ps();

			for (size_t k = 0; k < p; k += 4)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
				xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
				xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
				xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
				xmm_b = _mm_loadu_ps(b + k);
				// return the weighted sum
				xmm_c0 = _mm_fmadd_ps(xmm_a0, xmm_b, xmm_c0);
				xmm_c1 = _mm_fmadd_ps(xmm_a1, xmm_b, xmm_c1);
				xmm_c2 = _mm_fmadd_ps(xmm_a2, xmm_b, xmm_c2);
				xmm_c3 = _mm_fmadd_ps(xmm_a3, xmm_b, xmm_c3);
			}
			// return the horizontal sum
			xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_ps(xmm_c1, xmm_c1);
			xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c2);
			xmm_c3 = _mm_hadd_ps(xmm_c3, xmm_c3);
			xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_ps(xmm_c1, xmm_c1);
			xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c2);
			xmm_c3 = _mm_hadd_ps(xmm_c3, xmm_c3);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<float*>(&xmm_c0);
			*ptr_c1 += *reinterpret_cast<float*>(&xmm_c1);
			*ptr_c2 += *reinterpret_cast<float*>(&xmm_c2);
			*ptr_c3 += *reinterpret_cast<float*>(&xmm_c3);
		}
	};

	template<>
	struct block_mul_rm_cv<double, cpu_sse3>
	{
		// C(2x1) += A(2xp) * B(1xp)^T
		void operator()(size_t p, const double *a, size_t rsa, const double *b, double *c, size_t rsc) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			double *ptr_c0 = c;
			double *ptr_c1 = ptr_c0 + rsc;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b;
			__m128d xmm_c0 = _mm_setzero_pd();
			__m128d xmm_c1 = _mm_setzero_pd();

			for (size_t k = 0; k < p; k += 2)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
				xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
				xmm_b = _mm_loadu_pd(b + k);
				// return the weighted sum
				xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b), xmm_c0);
				xmm_c1 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b), xmm_c1);
			}
			// return the horizontal sum
			xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_pd(xmm_c1, xmm_c1);
			xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_pd(xmm_c1, xmm_c1);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<double*>(&xmm_c0);
			*ptr_c1 += *reinterpret_cast<double*>(&xmm_c1);
		}
	};

	template<>
	struct block_mul_rm_cv<double, cpu_sse3 | cpu_fma>
	{
		// C(2x1) += A(2xp) * B(1xp)^T
		void operator()(size_t p, const double *a, size_t rsa, const double *b, double *c, size_t rsc) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			double *ptr_c0 = c;
			double *ptr_c1 = ptr_c0 + rsc;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b;
			__m128d xmm_c0 = _mm_setzero_pd();
			__m128d xmm_c1 = _mm_setzero_pd();

			for (size_t k = 0; k < p; k += 2)
			{
				// load data from memory
				xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
				xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
				xmm_b = _mm_loadu_pd(b + k);
				// return the weighted sum
				xmm_c0 = _mm_fmadd_pd(xmm_a0, xmm_b, xmm_c0);
				xmm_c1 = _mm_fmadd_pd(xmm_a1, xmm_b, xmm_c1);
			}
			// return the horizontal sum
			xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_pd(xmm_c1, xmm_c1);
			xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c0);
			xmm_c1 = _mm_hadd_pd(xmm_c1, xmm_c1);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<double*>(&xmm_c0);
			*ptr_c1 += *reinterpret_cast<double*>(&xmm_c1);
		}
	};

	template<>
	struct block_mul_rm_cv<float, cpu_avx>
	{
		// C(4x1) += A(4xp) * B(1xp)^T
		void operator()(size_t p, const float *a, size_t rsa, const float *b, float *c, size_t rsc) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			float *ptr_c0 = c;
			float *ptr_c1 = ptr_c0 + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			float *ptr_c4 = ptr_c3 + rsc;
			float *ptr_c5 = ptr_c4 + rsc;
			float *ptr_c6 = ptr_c5 + rsc;
			float *ptr_c7 = ptr_c6 + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b;
			__m256 ymm_c0 = _mm256_setzero_ps();
			__m256 ymm_c1 = _mm256_setzero_ps();
			__m256 ymm_c2 = _mm256_setzero_ps();
			__m256 ymm_c3 = _mm256_setzero_ps();
			__m256 ymm_c4 = _mm256_setzero_ps();
			__m256 ymm_c5 = _mm256_setzero_ps();
			__m256 ymm_c6 = _mm256_setzero_ps();
			__m256 ymm_c7 = _mm256_setzero_ps();

			for (size_t k = 0; k < p; k += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
				ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
				ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
				ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
				ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
				ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
				ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
				ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
				ymm_b = _mm256_loadu_ps(b + k);
				// return the weighted sum
				ymm_c0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b), ymm_c0);
				ymm_c1 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b), ymm_c1);
				ymm_c2 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b), ymm_c2);
				ymm_c3 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b), ymm_c3);
				ymm_c4 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b), ymm_c4);
				ymm_c5 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b), ymm_c5);
				ymm_c6 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b), ymm_c6);
				ymm_c7 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b), ymm_c7);
			}
			// return the horizontal sum
			ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_ps(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_ps(ymm_c3, ymm_c3);
			ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c4);
			ymm_c5 = _mm256_hadd_ps(ymm_c5, ymm_c5);
			ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c6);
			ymm_c7 = _mm256_hadd_ps(ymm_c7, ymm_c7);
			ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_ps(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_ps(ymm_c3, ymm_c3);
			ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c4);
			ymm_c5 = _mm256_hadd_ps(ymm_c5, ymm_c5);
			ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c6);
			ymm_c7 = _mm256_hadd_ps(ymm_c7, ymm_c7);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<float*>(&ymm_c0);
			*ptr_c1 += *reinterpret_cast<float*>(&ymm_c1);
			*ptr_c2 += *reinterpret_cast<float*>(&ymm_c2);
			*ptr_c3 += *reinterpret_cast<float*>(&ymm_c3);
			*ptr_c4 += *reinterpret_cast<float*>(&ymm_c4);
			*ptr_c5 += *reinterpret_cast<float*>(&ymm_c5);
			*ptr_c6 += *reinterpret_cast<float*>(&ymm_c6);
			*ptr_c7 += *reinterpret_cast<float*>(&ymm_c7);
		}
	};

	template<>
	struct block_mul_rm_cv<float, cpu_avx | cpu_fma>
	{
		// C(4x1) += A(4xp) * B(1xp)^T
		void operator()(size_t p, const float *a, size_t rsa, const float *b, float *c, size_t rsc) const
		{
			const float *ptr_a0 = a;
			const float *ptr_a1 = ptr_a0 + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			float *ptr_c0 = c;
			float *ptr_c1 = ptr_c0 + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			float *ptr_c4 = ptr_c3 + rsc;
			float *ptr_c5 = ptr_c4 + rsc;
			float *ptr_c6 = ptr_c5 + rsc;
			float *ptr_c7 = ptr_c6 + rsc;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b;
			__m256 ymm_c0 = _mm256_setzero_ps();
			__m256 ymm_c1 = _mm256_setzero_ps();
			__m256 ymm_c2 = _mm256_setzero_ps();
			__m256 ymm_c3 = _mm256_setzero_ps();
			__m256 ymm_c4 = _mm256_setzero_ps();
			__m256 ymm_c5 = _mm256_setzero_ps();
			__m256 ymm_c6 = _mm256_setzero_ps();
			__m256 ymm_c7 = _mm256_setzero_ps();

			for (size_t k = 0; k < p; k += 8)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
				ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
				ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
				ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
				ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
				ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
				ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
				ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
				ymm_b = _mm256_loadu_ps(b + k);
				// return the weighted sum
				ymm_c0 = _mm256_fmadd_ps(ymm_a0, ymm_b, ymm_c0);
				ymm_c1 = _mm256_fmadd_ps(ymm_a1, ymm_b, ymm_c1);
				ymm_c2 = _mm256_fmadd_ps(ymm_a2, ymm_b, ymm_c2);
				ymm_c3 = _mm256_fmadd_ps(ymm_a3, ymm_b, ymm_c3);
				ymm_c4 = _mm256_fmadd_ps(ymm_a4, ymm_b, ymm_c4);
				ymm_c5 = _mm256_fmadd_ps(ymm_a5, ymm_b, ymm_c5);
				ymm_c6 = _mm256_fmadd_ps(ymm_a6, ymm_b, ymm_c6);
				ymm_c7 = _mm256_fmadd_ps(ymm_a7, ymm_b, ymm_c7);
			}
			// return the horizontal sum
			ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_ps(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_ps(ymm_c3, ymm_c3);
			ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c4);
			ymm_c5 = _mm256_hadd_ps(ymm_c5, ymm_c5);
			ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c6);
			ymm_c7 = _mm256_hadd_ps(ymm_c7, ymm_c7);
			ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_ps(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_ps(ymm_c3, ymm_c3);
			ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c4);
			ymm_c5 = _mm256_hadd_ps(ymm_c5, ymm_c5);
			ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c6);
			ymm_c7 = _mm256_hadd_ps(ymm_c7, ymm_c7);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<float*>(&ymm_c0);
			*ptr_c1 += *reinterpret_cast<float*>(&ymm_c1);
			*ptr_c2 += *reinterpret_cast<float*>(&ymm_c2);
			*ptr_c3 += *reinterpret_cast<float*>(&ymm_c3);
			*ptr_c4 += *reinterpret_cast<float*>(&ymm_c4);
			*ptr_c5 += *reinterpret_cast<float*>(&ymm_c5);
			*ptr_c6 += *reinterpret_cast<float*>(&ymm_c6);
			*ptr_c7 += *reinterpret_cast<float*>(&ymm_c7);
		}
	};

	template<>
	struct block_mul_rm_cv<double, cpu_avx>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, size_t rsa, const double *b, double *c, size_t rsc) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			double *ptr_c0 = c;
			double *ptr_c1 = ptr_c0 + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b;
			__m256d ymm_c0 = _mm256_setzero_pd();
			__m256d ymm_c1 = _mm256_setzero_pd();
			__m256d ymm_c2 = _mm256_setzero_pd();
			__m256d ymm_c3 = _mm256_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
				ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
				ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
				ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
				ymm_b = _mm256_loadu_pd(b + k);
				// return the weighted sum
				ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b), ymm_c0);
				ymm_c1 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b), ymm_c1);
				ymm_c2 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b), ymm_c2);
				ymm_c3 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b), ymm_c3);
			}
			// return the horizontal sum
			ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_pd(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_pd(ymm_c3, ymm_c3);
			ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_pd(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_pd(ymm_c3, ymm_c3);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<double*>(&ymm_c0);
			*ptr_c1 += *reinterpret_cast<double*>(&ymm_c1);
			*ptr_c2 += *reinterpret_cast<double*>(&ymm_c2);
			*ptr_c3 += *reinterpret_cast<double*>(&ymm_c3);
		}
	};

	template<>
	struct block_mul_rm_cv<double, cpu_avx | cpu_fma>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, size_t rsa, const double *b, double *c, size_t rsc) const
		{
			const double *ptr_a0 = a;
			const double *ptr_a1 = ptr_a0 + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			double *ptr_c0 = c;
			double *ptr_c1 = ptr_c0 + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b;
			__m256d ymm_c0 = _mm256_setzero_pd();
			__m256d ymm_c1 = _mm256_setzero_pd();
			__m256d ymm_c2 = _mm256_setzero_pd();
			__m256d ymm_c3 = _mm256_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				// load data from memory
				ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
				ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
				ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
				ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
				ymm_b = _mm256_loadu_pd(b + k);
				// return the weighted sum
				ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b, ymm_c0);
				ymm_c1 = _mm256_fmadd_pd(ymm_a1, ymm_b, ymm_c1);
				ymm_c2 = _mm256_fmadd_pd(ymm_a2, ymm_b, ymm_c2);
				ymm_c3 = _mm256_fmadd_pd(ymm_a3, ymm_b, ymm_c3);
			}
			// return the horizontal sum
			ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_pd(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_pd(ymm_c3, ymm_c3);
			ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
			ymm_c1 = _mm256_hadd_pd(ymm_c1, ymm_c1);
			ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c2);
			ymm_c3 = _mm256_hadd_pd(ymm_c3, ymm_c3);
			// store data into memory
			*ptr_c0 += *reinterpret_cast<double*>(&ymm_c0);
			*ptr_c1 += *reinterpret_cast<double*>(&ymm_c1);
			*ptr_c2 += *reinterpret_cast<double*>(&ymm_c2);
			*ptr_c3 += *reinterpret_cast<double*>(&ymm_c3);
		}
	};

	// Class template kernel_mul_rm_cv
	template<class T, size_t block_m, size_t block_p, cpu_inst_type inst>
	struct kernel_mul_rm_cv
	{
		// C(mx1) += A(mxp) * B(1xp)^T
		void operator()(size_t m, size_t p, const T *a, size_t rsa, const T *b, T *c) const
		{
			const size_t block_rsa = block_m * rsa;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const struct common_mul_rm_cv<T> functor;
			const struct block_mul_rm_cv<T, inst> special_functor;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_p > 0)
					special_functor(aligned_p, a, rsa, b, c + i, 1);
				if (surplus_p > 0)
					functor(block_m, surplus_p, a + aligned_p, rsa, b + aligned_p, c + i, 1);
				a += block_rsa;
			}
			if (surplus_m > 0)
				functor(surplus_m, p, a, rsa, b, c + aligned_m, 1);
		}
	};
	// TODO: The multiplication of the row-major order matrix and the column vector

} // namespace core

#endif
