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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEVM_FLOAT_H__
#define __CORE_CPU_KERNEL_BLOCK_GEVM_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gevm_float

	template<cpu_inst_type inst>
	struct block_gevm_float
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t /*aligned_n*/, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			float val_a0 = a[0];
			float val_a1 = a[1];
			float val_a2 = a[2];
			float val_a3 = a[3];
			float val_t0, val_t1, val_t2, val_t3;

			for (size_t j = 0; j < n; ++j)
			{
				val_t0 = val_a0 * ptr_b0[j];
				val_t1 = val_a1 * ptr_b1[j];
				val_t2 = val_a2 * ptr_b2[j];
				val_t3 = val_a3 * ptr_b3[j];
				val_t0 = (val_t0 + val_t1) + (val_t2 + val_t3);
				c[j] += val_t0;
			}
		}
	};

	template<>
	struct block_gevm_float<cpu_sse3>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128 xmm_c0;

			if (aligned_n > 0)
			{
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
					xmm_c0 = _mm_loadu_ps(c + j);
					// return the weighted sum
					xmm_t0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_t1 = _mm_mul_ps(xmm_a1, xmm_b1);
					xmm_t2 = _mm_mul_ps(xmm_a2, xmm_b2);
					xmm_t3 = _mm_mul_ps(xmm_a3, xmm_b3);
					xmm_t0 = _mm_add_ps(xmm_t0, xmm_t1);
					xmm_t2 = _mm_add_ps(xmm_t2, xmm_t3);
					xmm_t0 = _mm_add_ps(xmm_t0, xmm_t2);
					xmm_c0 = _mm_add_ps(xmm_c0, xmm_t0);
					// store data into memory
					_mm_storeu_ps(c + j, xmm_c0);
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_t0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t0);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t0);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_float<cpu_sse3 | cpu_fma>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t0, xmm_t1;
			__m128 xmm_c0;

			if (aligned_n > 0)
			{
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
					xmm_c0 = _mm_loadu_ps(c + j);
					// return the weighted sum
					xmm_t0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_t1 = _mm_mul_ps(xmm_a1, xmm_b1);
					xmm_t0 = _mm_fmadd_ps(xmm_a2, xmm_b2, xmm_t0);
					xmm_t1 = _mm_fmadd_ps(xmm_a3, xmm_b3, xmm_t1);
					xmm_t0 = _mm_add_ps(xmm_t0, xmm_t1);
					xmm_c0 = _mm_add_ps(xmm_c0, xmm_t0);
					// store data into memory
					_mm_storeu_ps(c + j, xmm_c0);
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_ps(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_ps(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_t0 = _mm_mul_ps(xmm_a0, xmm_b0);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t0);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t0);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_float<cpu_avx>
	{
		// C(1xn) += A(1x8) * B(8xn)
		void operator()(size_t aligned_n, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
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
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j], ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t0);
					ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t1);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&ymm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_float<cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1x8) * B(8xn)
		void operator()(size_t aligned_n, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
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
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256 ymm_c0;

			if (aligned_n > 0)
			{
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
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_ps(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_ps(ptr_b7[j], ptr_b6[j], ptr_b5[j], ptr_b4[j], ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_ps(ymm_a0, ymm_b0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t0);
					ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t0);
					ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_t0 = _mm256_add_ps(ymm_t0, ymm_t1);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&ymm_t0)[0];
				}
			}
		}
	};

} // namespace core

#endif
