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

#ifndef __CORE_CPU_KERNEL_MUL_RM_RM_H__
#define __CORE_CPU_KERNEL_MUL_RM_RM_H__

#include "kernel_mul_rv_rm.h"
#include "kernel_mul_rm_cv.h"

namespace core
{

	template<>
	struct block_mul_rv_rm<float, cpu_sse>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			float *ptr_c;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			for (size_t i = 0; i < 4; ++i)
			{
				ptr_c = c;
				xmm_a0 = _mm_set1_ps(a[0]);
				xmm_a1 = _mm_set1_ps(a[1]);
				xmm_a2 = _mm_set1_ps(a[2]);
				xmm_a3 = _mm_set1_ps(a[3]);

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
					_mm_storeu_ps(ptr_c, _mm_add_ps(_mm_loadu_ps(ptr_c), xmm_c0));
					ptr_c += 4;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<float, cpu_sse | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			float *ptr_c;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1;

			for (size_t i = 0; i < 4; ++i)
			{
				ptr_c = c;
				xmm_a0 = _mm_set1_ps(a[0]);
				xmm_a1 = _mm_set1_ps(a[1]);
				xmm_a2 = _mm_set1_ps(a[2]);
				xmm_a3 = _mm_set1_ps(a[3]);

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
					_mm_storeu_ps(ptr_c, _mm_add_ps(_mm_loadu_ps(ptr_c), xmm_c0));
					ptr_c += 4;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<double, cpu_sse2>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			double *ptr_c;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			for (size_t i = 0; i < 2; ++i)
			{
				ptr_c = c;
				xmm_a0 = _mm_set1_pd(a[0]);
				xmm_a1 = _mm_set1_pd(a[1]);

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
					_mm_storeu_pd(ptr_c, _mm_add_pd(_mm_loadu_pd(ptr_c), xmm_c0));
					ptr_c += 2;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<double, cpu_sse2 | cpu_fma>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			double *ptr_c;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			for (size_t i = 0; i < 2; ++i)
			{
				ptr_c = c;
				xmm_a0 = _mm_set1_pd(a[0]);
				xmm_a1 = _mm_set1_pd(a[1]);

				for (size_t j = 0; j < n; j += 2)
				{
					// load data from memory
					xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
					xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
					// return the weighted sum
					xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_c0 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_c0);
					// store data into memory
					_mm_storeu_pd(ptr_c, _mm_add_pd(_mm_loadu_pd(ptr_c), xmm_c0));
					ptr_c += 2;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<float, cpu_avx>
	{
		// C(8xn) += A(8x8) * B(8xn)
		void operator()(size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			float *ptr_c;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			for (size_t i = 0; i < 8; ++i)
			{
				ptr_c = c;
				ymm_a0 = _mm256_set1_ps(a[0]);
				ymm_a1 = _mm256_set1_ps(a[1]);
				ymm_a2 = _mm256_set1_ps(a[2]);
				ymm_a3 = _mm256_set1_ps(a[3]);
				ymm_a4 = _mm256_set1_ps(a[4]);
				ymm_a5 = _mm256_set1_ps(a[5]);
				ymm_a6 = _mm256_set1_ps(a[6]);
				ymm_a7 = _mm256_set1_ps(a[7]);

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
					_mm256_storeu_ps(ptr_c, _mm256_add_ps(_mm256_loadu_ps(ptr_c), ymm_c0));
					ptr_c += 8;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<float, cpu_avx | cpu_fma>
	{
		// C(8xn) += A(8x8) * B(8xn)
		void operator()(size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = ptr_b0 + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			float *ptr_c;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			for (size_t i = 0; i < 8; ++i)
			{
				ptr_c = c;
				ymm_a0 = _mm256_set1_ps(a[0]);
				ymm_a1 = _mm256_set1_ps(a[1]);
				ymm_a2 = _mm256_set1_ps(a[2]);
				ymm_a3 = _mm256_set1_ps(a[3]);
				ymm_a4 = _mm256_set1_ps(a[4]);
				ymm_a5 = _mm256_set1_ps(a[5]);
				ymm_a6 = _mm256_set1_ps(a[6]);
				ymm_a7 = _mm256_set1_ps(a[7]);

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
					_mm256_storeu_ps(ptr_c, _mm256_add_ps(_mm256_loadu_ps(ptr_c), ymm_c0));
					ptr_c += 8;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<double, cpu_avx>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double *ptr_c;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			for (size_t i = 0; i < 4; ++i)
			{
				ptr_c = c;
				ymm_a0 = _mm256_set1_pd(a[0]);
				ymm_a1 = _mm256_set1_pd(a[1]);
				ymm_a2 = _mm256_set1_pd(a[2]);
				ymm_a3 = _mm256_set1_pd(a[3]);

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
					_mm256_storeu_pd(ptr_c, _mm256_add_pd(_mm256_loadu_pd(ptr_c), ymm_c0));
					ptr_c += 4;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<double, cpu_avx>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double *ptr_c;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			for (size_t i = 0; i < 4; ++i)
			{
				ptr_c = c;
				ymm_a0 = _mm256_set1_pd(a[0]);
				ymm_a1 = _mm256_set1_pd(a[1]);
				ymm_a2 = _mm256_set1_pd(a[2]);
				ymm_a3 = _mm256_set1_pd(a[3]);

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
					_mm256_storeu_pd(ptr_c, _mm256_add_pd(_mm256_loadu_pd(ptr_c), ymm_c0));
					ptr_c += 4;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_mul_rv_rm<double, cpu_avx | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = ptr_b0 + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double *ptr_c;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1;

			for (size_t i = 0; i < 4; ++i)
			{
				ptr_c = c;
				ymm_a0 = _mm256_set1_pd(a[0]);
				ymm_a1 = _mm256_set1_pd(a[1]);
				ymm_a2 = _mm256_set1_pd(a[2]);
				ymm_a3 = _mm256_set1_pd(a[3]);

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
					_mm256_storeu_pd(ptr_c, _mm256_add_pd(_mm256_loadu_pd(ptr_c), ymm_c0));
					ptr_c += 4;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	// Class template kernel_matrix_multiply
	template<class T, size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_mul_rm_rm
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			const size_t block_rsb = block_p * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_n = n - aligned_n;
			const struct common_mul_rv_rm<T> functor;
			const struct block_mul_rv_rm<T, inst> mul_rv_rm;
			const struct block_mul_rm_cv<T, inst> mul_rm_cv;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					if (aligned_n > 0)
						mul_rv_rm(aligned_n, a + k, ptr_b, rsb, c);
					for (size_t j = aligned_n; j < n; ++j)
						mul_rm_cv(block_p, a + k, rsa, ptr_b + j, c + j, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					functor(surplus_p, n, a + aligned_p, ptr_b, rsb, c);
				a += rsa;
				c += rsc;
			}
		}

		void test(size_t m, size_t n, size_t p)
		{
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_n = n - aligned_n;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					for (size_t j = 0; j < aligned_n; j += block_n)
					{

					}
				}
			}
		}
	};

} // namespace core

#endif
