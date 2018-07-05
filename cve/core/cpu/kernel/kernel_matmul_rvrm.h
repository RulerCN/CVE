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

#ifndef __CORE_CPU_KERNEL_MATMUL_RVRM_H__
#define __CORE_CPU_KERNEL_MATMUL_RVRM_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template common_matmul_rvrm
	template<class T>
	struct common_matmul_rvrm
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t n, const T *a, const T *b, size_t rsb, T *c) const
		{
			T val_a;
			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				for (size_t j = 0; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

	// Class template block_matmul_rvrm
	template<class T, cpu_inst_type inst>
	struct block_matmul_rvrm
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t n, const T *a, const T *b, size_t rsb, T *c) const
		{
			const T *ptr_b0 = b;
			const T *ptr_b1 = b + rsb;
			const T *ptr_b2 = ptr_b1 + rsb;
			const T *ptr_b3 = ptr_b2 + rsb;
			T val_a0 = a[0];
			T val_a1 = a[1];
			T val_a2 = a[2];
			T val_a3 = a[3];
			T val_c0, val_c1, val_c2, val_c3;

			for (size_t j = 0; j < n; j += 4)
			{
				val_c0 = val_a0 * ptr_b0[0];
				val_c1 = val_a0 * ptr_b0[1];
				val_c2 = val_a0 * ptr_b0[2];
				val_c3 = val_a0 * ptr_b0[3];
				val_c0 += val_a1 * ptr_b1[0];
				val_c1 += val_a1 * ptr_b1[1];
				val_c2 += val_a1 * ptr_b1[2];
				val_c3 += val_a1 * ptr_b1[3];
				val_c0 += val_a2 * ptr_b2[0];
				val_c1 += val_a2 * ptr_b2[1];
				val_c2 += val_a2 * ptr_b2[2];
				val_c3 += val_a2 * ptr_b2[3];
				val_c0 += val_a3 * ptr_b3[0];
				val_c1 += val_a3 * ptr_b3[1];
				val_c2 += val_a3 * ptr_b3[2];
				val_c3 += val_a3 * ptr_b3[3];
				c[0] += val_c0;
				c[1] += val_c1;
				c[2] += val_c2;
				c[3] += val_c3;
				ptr_b0 += 4;
				ptr_b1 += 4;
				ptr_b2 += 4;
				ptr_b3 += 4;
				c += 4;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<float, cpu_sse>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0 = _mm_set1_ps(a[0]);
			__m128 xmm_a1 = _mm_set1_ps(a[1]);
			__m128 xmm_a2 = _mm_set1_ps(a[2]);
			__m128 xmm_a3 = _mm_set1_ps(a[3]);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(ptr_b0 + j);
				xmm_b1 = _mm_loadu_ps(ptr_b1 + j);
				xmm_b2 = _mm_loadu_ps(ptr_b2 + j);
				xmm_b3 = _mm_loadu_ps(ptr_b3 + j);
				// return the weighted sum
				xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
				xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
				xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b2);
				xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b3);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
				xmm_c2 = _mm_add_ps(xmm_c2, xmm_c3);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_c2);
				// store data into memory
				_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
				c += 4;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<float, cpu_sse | cpu_fma>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0 = _mm_set1_ps(a[0]);
			__m128 xmm_a1 = _mm_set1_ps(a[1]);
			__m128 xmm_a2 = _mm_set1_ps(a[2]);
			__m128 xmm_a3 = _mm_set1_ps(a[3]);
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(ptr_b0 + j);
				xmm_b1 = _mm_loadu_ps(ptr_b1 + j);
				xmm_b2 = _mm_loadu_ps(ptr_b2 + j);
				xmm_b3 = _mm_loadu_ps(ptr_b3 + j);
				// return the weighted sum
				xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
				xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
				xmm_c0 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_c0);
				xmm_c1 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_c1);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
				// store data into memory
				_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), xmm_c0));
				c += 4;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<double, cpu_sse2>
	{
		// C(1xn) += A(1x2) * B(2xn)
		void operator()(size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0 = _mm_set1_pd(a[0]);
			__m128d xmm_a1 = _mm_set1_pd(a[1]);
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			for (size_t j = 0; j < n; j += 2)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
				xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
				// return the weighted sum
				xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
				xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
				xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
				// store data into memory
				_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
				c += 2;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<double, cpu_sse2 | cpu_fma>
	{
		// C(1xn) += A(1x2) * B(2xn)
		void operator()(size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0 = _mm_set1_pd(a[0]);
			__m128d xmm_a1 = _mm_set1_pd(a[1]);
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0;

			for (size_t j = 0; j < n; j += 2)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
				xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
				// return the weighted sum
				xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
				xmm_c0 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_c0);
				// store data into memory
				_mm_storeu_pd(c, _mm_add_pd(_mm_loadu_pd(c), xmm_c0));
				c += 2;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<float, cpu_avx>
	{
		// C(1xn) += A(1x8) * B(8xn)
		void operator()(size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a0 = _mm256_set1_ps(a[0]);
			__m256 ymm_a1 = _mm256_set1_ps(a[1]);
			__m256 ymm_a2 = _mm256_set1_ps(a[2]);
			__m256 ymm_a3 = _mm256_set1_ps(a[3]);
			__m256 ymm_a4 = _mm256_set1_ps(a[4]);
			__m256 ymm_a5 = _mm256_set1_ps(a[5]);
			__m256 ymm_a6 = _mm256_set1_ps(a[6]);
			__m256 ymm_a7 = _mm256_set1_ps(a[7]);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			for (size_t j = 0; j < n; j += 8)
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
				// return the weighted sum
				ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
				ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b4);
				ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b5);
				ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b6);
				ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b7);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
				ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c5);
				ymm_c6 = _mm256_add_ps(ymm_c6, ymm_c7);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c6);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c4);
				// store data into memory
				_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
				c += 8;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<float, cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1x8) * B(8xn)
		void operator()(size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a0 = _mm256_set1_ps(a[0]);
			__m256 ymm_a1 = _mm256_set1_ps(a[1]);
			__m256 ymm_a2 = _mm256_set1_ps(a[2]);
			__m256 ymm_a3 = _mm256_set1_ps(a[3]);
			__m256 ymm_a4 = _mm256_set1_ps(a[4]);
			__m256 ymm_a5 = _mm256_set1_ps(a[5]);
			__m256 ymm_a6 = _mm256_set1_ps(a[6]);
			__m256 ymm_a7 = _mm256_set1_ps(a[7]);
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			for (size_t j = 0; j < n; j += 8)
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
				// return the weighted sum
				ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
				ymm_c0 = _mm256_fmadd_ps(ymm_a4, ymm_b4, ymm_c0);
				ymm_c1 = _mm256_fmadd_ps(ymm_a5, ymm_b5, ymm_c1);
				ymm_c2 = _mm256_fmadd_ps(ymm_a6, ymm_b6, ymm_c2);
				ymm_c3 = _mm256_fmadd_ps(ymm_a7, ymm_b7, ymm_c3);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
				ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
				// store data into memory
				_mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), ymm_c0));
				c += 8;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<double, cpu_avx>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0 = _mm256_set1_pd(a[0]);
			__m256d ymm_a1 = _mm256_set1_pd(a[1]);
			__m256d ymm_a2 = _mm256_set1_pd(a[2]);
			__m256d ymm_a3 = _mm256_set1_pd(a[3]);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(ptr_b0 + j);
				ymm_b1 = _mm256_loadu_pd(ptr_b1 + j);
				ymm_b2 = _mm256_loadu_pd(ptr_b2 + j);
				ymm_b3 = _mm256_loadu_pd(ptr_b3 + j);
				// return the weighted sum
				ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b3);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
				ymm_c2 = _mm256_add_pd(ymm_c2, ymm_c3);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c2);
				// store data into memory
				_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
				c += 4;
			}
		}
	};

	template<>
	struct block_matmul_rvrm<double, cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0 = _mm256_set1_pd(a[0]);
			__m256d ymm_a1 = _mm256_set1_pd(a[1]);
			__m256d ymm_a2 = _mm256_set1_pd(a[2]);
			__m256d ymm_a3 = _mm256_set1_pd(a[3]);
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1;

			for (size_t j = 0; j < n; j += 4)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(ptr_b0 + j);
				ymm_b1 = _mm256_loadu_pd(ptr_b1 + j);
				ymm_b2 = _mm256_loadu_pd(ptr_b2 + j);
				ymm_b3 = _mm256_loadu_pd(ptr_b3 + j);
				// return the weighted sum
				ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
				ymm_c0 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_c0);
				ymm_c1 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_c1);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
				// store data into memory
				_mm256_storeu_pd(c, _mm256_add_pd(_mm256_loadu_pd(c), ymm_c0));
				c += 4;
			}
		}
	};

	// Class template kernel_matmul_rvrm
	template<class T, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_matmul_rvrm
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t n, const T *a, const T *b, size_t rsb, T *c) const
		{
			const size_t block_rsb = block_p * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_n = n - aligned_n;
			const struct common_matmul_rvrm<T> functor;
			const struct block_matmul_rvrm<T, inst> special_functor;

			for (size_t k = 0; k < aligned_p; k += block_p)
			{
				if (aligned_n > 0)
					special_functor(aligned_n, a + k, b, rsb, c);
				if (surplus_n > 0)
					functor(block_p, surplus_n, a + k, b + aligned_n, rsb, c + aligned_n);
				b += block_rsb;
			}
			if (surplus_p > 0)
				functor(surplus_p, n, a + aligned_p, b, rsb, c);
		}
	};

} // namespace core

#endif
