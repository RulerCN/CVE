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

#ifndef __CORE_CPU_KERNEL_MATMUL_RMRM00_H__
#define __CORE_CPU_KERNEL_MATMUL_RMRM00_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template block_matmul_rmrm00
	template<class T, cpu_inst_type inst>
	struct block_matmul_rmrm00
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b0 = b;
			const T *ptr_b1 = b + rsb;
			const T *ptr_b2 = ptr_b1 + rsb;
			const T *ptr_b3 = ptr_b2 + rsb;
			T val_a0, val_a1, val_a2, val_a3;
			T val_c0, val_c1, val_c2, val_c3;

			for (size_t i = 0; i < 4; ++i)
			{
				val_a0 = a[0];
				val_a1 = a[1];
				val_a2 = a[2];
				val_a3 = a[3];
				for (size_t j = 0; j < n; ++j)
				{
					val_c0 = val_a0 * ptr_b0[j];
					val_c1 = val_a1 * ptr_b1[j];
					val_c2 = val_a2 * ptr_b2[j];
					val_c3 = val_a3 * ptr_b3[j];
					val_c0 = (val_c0 + val_c1) + (val_c2 + val_c3);
					c[j] += val_c0;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<float, cpu_sse3>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c[4];
			const float *ptr_a[4];
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
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
						_mm_storeu_ps(c + j, _mm_add_ps(_mm_loadu_ps(c + j), xmm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(ptr_a[0]);
				xmm_a1 = _mm_loadu_ps(ptr_a[1]);
				xmm_a2 = _mm_loadu_ps(ptr_a[2]);
				xmm_a3 = _mm_loadu_ps(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b0);
					xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b0);
					xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b0);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c1);
					xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c3);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c2);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<float, cpu_sse3 | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c[4];
			const float *ptr_a[4];
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					xmm_a0 = _mm_set1_ps(a[0]);
					xmm_a1 = _mm_set1_ps(a[1]);
					xmm_a2 = _mm_set1_ps(a[2]);
					xmm_a3 = _mm_set1_ps(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
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
						_mm_storeu_ps(c + j, _mm_add_ps(_mm_loadu_ps(c + j), xmm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(ptr_a[0]);
				xmm_a1 = _mm_loadu_ps(ptr_a[1]);
				xmm_a2 = _mm_loadu_ps(ptr_a[2]);
				xmm_a3 = _mm_loadu_ps(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b0);
					xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b0);
					xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b0);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c1);
					xmm_c2 = _mm_hadd_ps(xmm_c2, xmm_c3);
					xmm_c0 = _mm_hadd_ps(xmm_c0, xmm_c2);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&xmm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&xmm_c0)[3];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<double, cpu_sse3>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c[2];
			const double *ptr_a[2];
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
						xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
						xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
						// store data into memory
						_mm_storeu_pd(c + j, _mm_add_pd(_mm_loadu_pd(c + j), xmm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(ptr_a[0]);
				xmm_a1 = _mm_loadu_pd(ptr_a[1]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b0);
					xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<double, cpu_sse3 | cpu_fma>
	{
		// C(2xn) += A(2x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c[2];
			const double *ptr_a[2];
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
						xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
						// return the weighted sum
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_c0 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_c0);
						// store data into memory
						_mm_storeu_pd(c + j, _mm_add_pd(_mm_loadu_pd(c + j), xmm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(ptr_a[0]);
				xmm_a1 = _mm_loadu_pd(ptr_a[1]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b0);
					xmm_c0 = _mm_hadd_pd(xmm_c0, xmm_c1);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&xmm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&xmm_c0)[1];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<float, cpu_avx>
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
						_mm256_storeu_ps(c + j, _mm256_add_ps(_mm256_loadu_ps(c + j), ymm_c0));
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
					ymm_b0 = _mm256_set_ps(
						ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j],
						ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c3);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c5);
					ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c7);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c2);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c6);
					ymm_c1 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_c5 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_ps(ymm_c1, ymm_c5);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<float, cpu_avx | cpu_fma>
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
						_mm256_storeu_ps(c + j, _mm256_add_ps(_mm256_loadu_ps(c + j), ymm_c0));
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
					ymm_b0 = _mm256_set_ps(
						ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j],
						ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b0);
					ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b0);
					ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b0);
					ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b0);
					ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b0);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_ps(ymm_c2, ymm_c3);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c5);
					ymm_c6 = _mm256_hadd_ps(ymm_c6, ymm_c7);
					ymm_c0 = _mm256_hadd_ps(ymm_c0, ymm_c2);
					ymm_c4 = _mm256_hadd_ps(ymm_c4, ymm_c6);
					ymm_c1 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_c5 = _mm256_permute2f128_ps(ymm_c0, ymm_c4, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_ps(ymm_c1, ymm_c5);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<float*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<float*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<float*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<float*>(&ymm_c0)[3];
					ptr_c[4][j] += reinterpret_cast<float*>(&ymm_c0)[4];
					ptr_c[5][j] += reinterpret_cast<float*>(&ymm_c0)[5];
					ptr_c[6][j] += reinterpret_cast<float*>(&ymm_c0)[6];
					ptr_c[7][j] += reinterpret_cast<float*>(&ymm_c0)[7];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<double, cpu_avx>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c[4];
			const double *ptr_a[4];
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
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
						_mm256_storeu_pd(c + j, _mm256_add_pd(_mm256_loadu_pd(c + j), ymm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(ptr_a[0]);
				ymm_a1 = _mm256_loadu_pd(ptr_a[1]);
				ymm_a2 = _mm256_loadu_pd(ptr_a[2]);
				ymm_a3 = _mm256_loadu_pd(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b0);
					ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c3);
					ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_c3 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_pd(ymm_c1, ymm_c3);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
				}
			}
		}
	};

	template<>
	struct block_matmul_rmrm00<double, cpu_avx | cpu_fma>
	{
		// C(4xn) += A(4x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c[4];
			const double *ptr_a[4];
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				for (size_t i = 0; i < 4; ++i)
				{
					ptr_a[i] = a;
					ptr_c[i] = c;
					ymm_a0 = _mm256_set1_pd(a[0]);
					ymm_a1 = _mm256_set1_pd(a[1]);
					ymm_a2 = _mm256_set1_pd(a[2]);
					ymm_a3 = _mm256_set1_pd(a[3]);
					for (size_t j = 0; j < aligned_n; j += 4)
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
						_mm256_storeu_pd(c + j, _mm256_add_pd(_mm256_loadu_pd(c + j), ymm_c0));
					}
					a += rsa;
					c += rsc;
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(ptr_a[0]);
				ymm_a1 = _mm256_loadu_pd(ptr_a[1]);
				ymm_a2 = _mm256_loadu_pd(ptr_a[2]);
				ymm_a3 = _mm256_loadu_pd(ptr_a[3]);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b0);
					ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b0);
					ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b0);
					ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c1);
					ymm_c2 = _mm256_hadd_pd(ymm_c2, ymm_c3);
					ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_c3 = _mm256_permute2f128_pd(ymm_c0, ymm_c2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_c0 = _mm256_add_pd(ymm_c1, ymm_c3);
					// store data into memory
					ptr_c[0][j] += reinterpret_cast<double*>(&ymm_c0)[0];
					ptr_c[1][j] += reinterpret_cast<double*>(&ymm_c0)[1];
					ptr_c[2][j] += reinterpret_cast<double*>(&ymm_c0)[2];
					ptr_c[3][j] += reinterpret_cast<double*>(&ymm_c0)[3];
				}
			}
		}
	};

} // namespace core

#endif
