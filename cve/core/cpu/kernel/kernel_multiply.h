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

#ifndef __CORE_CPU_KERNEL_MULTIPLY_H__
#define __CORE_CPU_KERNEL_MULTIPLY_H__

#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Class template common_multiply
	template<class T>
	struct common_multiply
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t p, const T *a, const T *b, size_t rsb, T *c) const
		{
			T val_c;
			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				for (size_t k = 0; k < p; ++k)
					val_c += a[k] * b[k];
				c[j] += val_c;
				b += rsb;
			}
		}
	};

	// Class template block_multiply
	template<class T, inst_type inst>
	struct block_multiply
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const T *a, const T *b, size_t rsb, T *c) const
		{
			const T *ptr_b0 = b;
			const T *ptr_b1 = ptr_b0 + rsb;
			const T *ptr_b2 = ptr_b1 + rsb;
			const T *ptr_b3 = ptr_b2 + rsb;
			T val_a;
			T val_c0 = 0, val_c1 = 0, val_c2 = 0, val_c3 = 0;

			for (size_t k = 0; k < p;)
			{
				// k
				val_a = a[k];
				val_c0 += val_a * ptr_b0[k];
				val_c1 += val_a * ptr_b1[k];
				val_c2 += val_a * ptr_b2[k];
				val_c3 += val_a * ptr_b3[k];
				++k;
				// k + 1
				val_a = a[k];
				val_c0 += val_a * ptr_b0[k];
				val_c1 += val_a * ptr_b1[k];
				val_c2 += val_a * ptr_b2[k];
				val_c3 += val_a * ptr_b3[k];
				++k;
				// k + 2
				val_a = a[k];
				val_c0 += val_a * ptr_b0[k];
				val_c1 += val_a * ptr_b1[k];
				val_c2 += val_a * ptr_b2[k];
				val_c3 += val_a * ptr_b3[k];
				++k;
				// k + 3
				val_a = a[k];
				val_c0 += val_a * ptr_b0[k];
				val_c1 += val_a * ptr_b1[k];
				val_c2 += val_a * ptr_b2[k];
				val_c3 += val_a * ptr_b3[k];
				++k;
			}
			c[0] += val_c0;
			c[1] += val_c1;
			c[2] += val_c2;
			c[3] += val_c3;
		}
	};

	template<>
	struct block_multiply<float, inst_sse>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0 = _mm_setzero_ps();
			__m128 xmm_c1 = _mm_setzero_ps();
			__m128 xmm_c2 = _mm_setzero_ps();
			__m128 xmm_c3 = _mm_setzero_ps();

			for (size_t k = 0; k < p; k += 4)
			{
				xmm_a = _mm_loadu_ps(a + k);
				xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
				xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
				xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
				xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
				xmm_c0 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b0), xmm_c0);
				xmm_c1 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b1), xmm_c1);
				xmm_c2 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b2), xmm_c2);
				xmm_c3 = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b3), xmm_c3);
			}
			// return the horizontal sum
			xmm_b0 = _mm_shuffle_ps(xmm_c0, xmm_c1, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_b1 = _mm_shuffle_ps(xmm_c2, xmm_c3, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_b2 = _mm_shuffle_ps(xmm_c0, xmm_c1, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_b3 = _mm_shuffle_ps(xmm_c2, xmm_c3, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_b0 = _mm_add_ps(xmm_b0, xmm_b2);
			xmm_b1 = _mm_add_ps(xmm_b1, xmm_b3);
			xmm_c0 = _mm_shuffle_ps(xmm_b0, xmm_b1, _MM_SHUFFLE(2, 0, 2, 0));
			xmm_c1 = _mm_shuffle_ps(xmm_b0, xmm_b1, _MM_SHUFFLE(3, 1, 3, 1));
			xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
			// store data into memory
			_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
		}
	};

	template<>
	struct block_multiply<float, inst_sse | inst_fma>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0 = _mm_setzero_ps();
			__m128 xmm_c1 = _mm_setzero_ps();
			__m128 xmm_c2 = _mm_setzero_ps();
			__m128 xmm_c3 = _mm_setzero_ps();

			for (size_t k = 0; k < p; k += 4)
			{
				xmm_a = _mm_loadu_ps(a + k);
				xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
				xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
				xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
				xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
				xmm_c0 = _mm_fmadd_ps(xmm_a, xmm_b0, xmm_c0);
				xmm_c1 = _mm_fmadd_ps(xmm_a, xmm_b1, xmm_c1);
				xmm_c2 = _mm_fmadd_ps(xmm_a, xmm_b2, xmm_c2);
				xmm_c3 = _mm_fmadd_ps(xmm_a, xmm_b3, xmm_c3);
			}
			// return the horizontal sum
			xmm_b0 = _mm_shuffle_ps(xmm_c0, xmm_c1, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_b1 = _mm_shuffle_ps(xmm_c2, xmm_c3, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_b2 = _mm_shuffle_ps(xmm_c0, xmm_c1, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_b3 = _mm_shuffle_ps(xmm_c2, xmm_c3, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_b0 = _mm_add_ps(xmm_b0, xmm_b2);
			xmm_b1 = _mm_add_ps(xmm_b1, xmm_b3);
			xmm_c0 = _mm_shuffle_ps(xmm_b0, xmm_b1, _MM_SHUFFLE(2, 0, 2, 0));
			xmm_c1 = _mm_shuffle_ps(xmm_b0, xmm_b1, _MM_SHUFFLE(3, 1, 3, 1));
			xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
			// store data into memory
			_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
		}
	};

	template<>
	struct block_multiply<double, inst_sse2>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			__m128d xmm_a;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0 = _mm_setzero_pd();
			__m128d xmm_c1 = _mm_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				xmm_a = _mm_loadu_pd(a + k);
				xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
				xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
				xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b0), xmm_c0);
				xmm_c1 = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b1), xmm_c1);
			}
			// return the horizontal sum
			xmm_b0 = _mm_shuffle_pd(xmm_c0, xmm_c1, _MM_SHUFFLE(0, 0, 0, 0));
			xmm_b1 = _mm_shuffle_pd(xmm_c0, xmm_c1, _MM_SHUFFLE(0, 0, 3, 3));
			xmm_c0 = _mm_add_pd(xmm_b0, xmm_b1);
			// store data into memory
			_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
		}
	};

	template<>
	struct block_multiply<double, inst_sse2 | inst_fma>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			__m128d xmm_a;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0 = _mm_setzero_pd();
			__m128d xmm_c1 = _mm_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				xmm_a = _mm_loadu_pd(a + k);
				xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
				xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
				xmm_c0 = _mm_fmadd_pd(xmm_a, xmm_b0, xmm_c0);
				xmm_c1 = _mm_fmadd_pd(xmm_a, xmm_b1, xmm_c1);
			}
			// return the horizontal sum
			xmm_b0 = _mm_shuffle_pd(xmm_c0, xmm_c1, _MM_SHUFFLE(0, 0, 0, 0));
			xmm_b1 = _mm_shuffle_pd(xmm_c0, xmm_c1, _MM_SHUFFLE(0, 0, 3, 3));
			xmm_c0 = _mm_add_pd(xmm_b0, xmm_b1);
			// store data into memory
			_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
		}
	};

	template<>
	struct block_multiply<float, inst_avx>
	{
		// C(1x8) += A(1xp) * B(8xp)^T
		void operator()(size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
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
				ymm_a = _mm256_loadu_ps(a + k);
				ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
				ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
				ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
				ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
				ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
				ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
				ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
				ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
				ymm_c0 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b0), ymm_c0);
				ymm_c1 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b1), ymm_c1);
				ymm_c2 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b2), ymm_c2);
				ymm_c3 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b3), ymm_c3);
				ymm_c4 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b4), ymm_c4);
				ymm_c5 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b5), ymm_c5);
				ymm_c6 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b6), ymm_c6);
				ymm_c7 = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b7), ymm_c7);
			}
			// return the horizontal sum
			ymm_b0 = _mm256_shuffle_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b1 = _mm256_shuffle_ps(ymm_c2, ymm_c3, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b2 = _mm256_shuffle_ps(ymm_c4, ymm_c5, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b3 = _mm256_shuffle_ps(ymm_c6, ymm_c7, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b4 = _mm256_shuffle_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b5 = _mm256_shuffle_ps(ymm_c2, ymm_c3, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b6 = _mm256_shuffle_ps(ymm_c4, ymm_c5, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b7 = _mm256_shuffle_ps(ymm_c6, ymm_c7, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b0 = _mm256_add_ps(ymm_b0, ymm_b4);
			ymm_b1 = _mm256_add_ps(ymm_b1, ymm_b5);
			ymm_b2 = _mm256_add_ps(ymm_b2, ymm_b6);
			ymm_b3 = _mm256_add_ps(ymm_b3, ymm_b7);
			ymm_c0 = _mm256_shuffle_ps(ymm_b0, ymm_b1, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_c1 = _mm256_shuffle_ps(ymm_b2, ymm_b3, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_c2 = _mm256_shuffle_ps(ymm_b0, ymm_b1, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_c3 = _mm256_shuffle_ps(ymm_b2, ymm_b3, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
			ymm_c1 = _mm256_add_ps(ymm_c1, ymm_c3);
			ymm_b0 = _mm256_permute2f128_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_b1 = _mm256_permute2f128_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_c0 = _mm256_add_ps(ymm_b0, ymm_b1);
			// store data into memory
			_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
		}
	};

	template<>
	struct block_multiply<float, inst_avx | inst_fma>
	{
		// C(1x8) += A(1xp) * B(8xp)^T
		void operator()(size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
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
				ymm_a = _mm256_loadu_ps(a + k);
				ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
				ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
				ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
				ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
				ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
				ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
				ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
				ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
				ymm_c0 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c0);
				ymm_c1 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c1);
				ymm_c2 = _mm256_fmadd_ps(ymm_a, ymm_b2, ymm_c2);
				ymm_c3 = _mm256_fmadd_ps(ymm_a, ymm_b3, ymm_c3);
				ymm_c4 = _mm256_fmadd_ps(ymm_a, ymm_b4, ymm_c4);
				ymm_c5 = _mm256_fmadd_ps(ymm_a, ymm_b5, ymm_c5);
				ymm_c6 = _mm256_fmadd_ps(ymm_a, ymm_b6, ymm_c6);
				ymm_c7 = _mm256_fmadd_ps(ymm_a, ymm_b7, ymm_c7);
			}
			// return the horizontal sum
			ymm_b0 = _mm256_shuffle_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b1 = _mm256_shuffle_ps(ymm_c2, ymm_c3, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b2 = _mm256_shuffle_ps(ymm_c4, ymm_c5, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b3 = _mm256_shuffle_ps(ymm_c6, ymm_c7, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_b4 = _mm256_shuffle_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b5 = _mm256_shuffle_ps(ymm_c2, ymm_c3, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b6 = _mm256_shuffle_ps(ymm_c4, ymm_c5, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b7 = _mm256_shuffle_ps(ymm_c6, ymm_c7, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_b0 = _mm256_add_ps(ymm_b0, ymm_b4);
			ymm_b1 = _mm256_add_ps(ymm_b1, ymm_b5);
			ymm_b2 = _mm256_add_ps(ymm_b2, ymm_b6);
			ymm_b3 = _mm256_add_ps(ymm_b3, ymm_b7);
			ymm_c0 = _mm256_shuffle_ps(ymm_b0, ymm_b1, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_c1 = _mm256_shuffle_ps(ymm_b2, ymm_b3, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_c2 = _mm256_shuffle_ps(ymm_b0, ymm_b1, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_c3 = _mm256_shuffle_ps(ymm_b2, ymm_b3, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
			ymm_c1 = _mm256_add_ps(ymm_c1, ymm_c3);
			ymm_b0 = _mm256_permute2f128_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_b1 = _mm256_permute2f128_ps(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_c0 = _mm256_add_ps(ymm_b0, ymm_b1);
			// store data into memory
			_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
		}
	};

	template<>
	struct block_multiply<double, inst_avx>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0 = _mm256_setzero_pd();
			__m256d ymm_c1 = _mm256_setzero_pd();
			__m256d ymm_c2 = _mm256_setzero_pd();
			__m256d ymm_c3 = _mm256_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				ymm_a = _mm256_loadu_pd(a + k);
				ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
				ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
				ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
				ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
				ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b0), ymm_c0);
				ymm_c1 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b1), ymm_c1);
				ymm_c2 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b2), ymm_c2);
				ymm_c3 = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b3), ymm_c3);
			}
			// return the horizontal sum
			ymm_b0 = _mm256_shuffle_pd(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_b1 = _mm256_shuffle_pd(ymm_c2, ymm_c3, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_b2 = _mm256_shuffle_pd(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_b3 = _mm256_shuffle_pd(ymm_c2, ymm_c3, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_b0 = _mm256_add_pd(ymm_b0, ymm_b2);
			ymm_b1 = _mm256_add_pd(ymm_b1, ymm_b3);
			ymm_c0 = _mm256_permute2f128_pd(ymm_b0, ymm_b1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_c1 = _mm256_permute2f128_pd(ymm_b0, ymm_b1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
			// store data into memory
			_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
		}
	};

	template<>
	struct block_multiply<double, inst_avx | inst_fma>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t p, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0 = _mm256_setzero_pd();
			__m256d ymm_c1 = _mm256_setzero_pd();
			__m256d ymm_c2 = _mm256_setzero_pd();
			__m256d ymm_c3 = _mm256_setzero_pd();

			for (size_t k = 0; k < p; k += 4)
			{
				ymm_a = _mm256_loadu_pd(a + k);
				ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
				ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
				ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
				ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
				ymm_c0 = _mm256_fmadd_pd(ymm_a, ymm_b0, ymm_c0);
				ymm_c1 = _mm256_fmadd_pd(ymm_a, ymm_b1, ymm_c1);
				ymm_c2 = _mm256_fmadd_pd(ymm_a, ymm_b2, ymm_c2);
				ymm_c3 = _mm256_fmadd_pd(ymm_a, ymm_b3, ymm_c3);
			}
			// return the horizontal sum
			ymm_b0 = _mm256_shuffle_pd(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_b1 = _mm256_shuffle_pd(ymm_c2, ymm_c3, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_b2 = _mm256_shuffle_pd(ymm_c0, ymm_c1, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_b3 = _mm256_shuffle_pd(ymm_c2, ymm_c3, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_b0 = _mm256_add_pd(ymm_b0, ymm_b2);
			ymm_b1 = _mm256_add_pd(ymm_b1, ymm_b3);
			ymm_c0 = _mm256_permute2f128_pd(ymm_b0, ymm_b1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_c1 = _mm256_permute2f128_pd(ymm_b0, ymm_b1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
			// store data into memory
			_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
		}
	};

	// Class template kernel_multiply
	template<class T, size_t block_n, size_t block_p, inst_type inst>
	struct kernel_multiply
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t p, const T *a, const T *b, size_t rsb, T *c) const
		{
			const size_t block_rsb = block_n * rsb;
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t surplus_n = n - aligned_n;
			const size_t surplus_p = p - aligned_p;
			const struct common_multiply<T> functor;
			const struct block_multiply<T, inst> special_functor;

			for (int j = 0; j < aligned_n; j += block_n)
			{
				if (aligned_p > 0)
					special_functor(aligned_p, a, b, rsb, c);
				if (surplus_p > 0)
					functor(block_n, surplus_p, a, b, rsb, c);
				b += block_rsb;
			}
			if (surplus_n > 0)
				functor(surplus_n, p, a, b, rsb, c + aligned_n);
		}

		// C(mxn) += A(mxp) * B(nxp)^T
		void operator()(size_t m, size_t n, size_t p, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			const size_t block_rsb = block_n * rsb;
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t surplus_n = n - aligned_n;
			const size_t surplus_p = p - aligned_p;
			const struct common_multiply<T> functor;
			const struct block_multiply<T, inst> special_functor;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (int j = 0; j < aligned_n; j += block_n)
				{
					if (aligned_p > 0)
						special_functor(aligned_p, a, ptr_b, rsb, c + j);
					if (surplus_p > 0)
						functor(block_n, surplus_p, a + aligned_p, ptr_b + aligned_p, rsb, c + j);
					ptr_b += block_rsb;
				}
				if (surplus_n > 0)
					functor(surplus_n, p, a, ptr_b, rsb, c + aligned_n);
				a += rsa;
				c += rsc;
			}
		}
	};

} // namespace core

#endif
