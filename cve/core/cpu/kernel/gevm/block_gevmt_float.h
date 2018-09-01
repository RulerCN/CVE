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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEVMT_FLOAT_H__
#define __CORE_CPU_KERNEL_BLOCK_GEVMT_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gevmt_float
	
	template<cpu_inst_type inst>
	struct block_gevmt_float
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			float val_a0;
			float val_t0, val_t1, val_t2, val_t3;

			val_t0 = val_t1 = val_t2 = val_t3 = 0;
			for (size_t k = 0; k < p; ++k)
			{
				val_a0 = a[k];
				val_t0 += val_a0 * ptr_b0[k];
				val_t1 += val_a0 * ptr_b1[k];
				val_t2 += val_a0 * ptr_b2[k];
				val_t3 += val_a0 * ptr_b3[k];
			}
			c[0] += val_t0;
			c[1] += val_t1;
			c[2] += val_t2;
			c[3] += val_t3;
		}
	};

	template<>
	struct block_gevmt_float<cpu_sse3>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128 xmm_c0 = _mm_loadu_ps(c);

			if (aligned_p > 0)
			{
				xmm_t0 = xmm_t1 = xmm_t2 = xmm_t3 = _mm_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(a + k);
					xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
					xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
					xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
					xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
					// return the weighted sum
					xmm_t0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_t0);
					xmm_t1 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b1), xmm_t1);
					xmm_t2 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b2), xmm_t2);
					xmm_t3 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b3), xmm_t3);
				}
				// return the horizontal sum
				xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t1);
				xmm_t2 = _mm_hadd_ps(xmm_t2, xmm_t3);
				xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t2);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_t0);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_ps(a[k]);
					xmm_b0 = _mm_set_ps(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_c0);
				}
			}
			// store data into memory
			_mm_storeu_ps(c, xmm_c0);
		}
	};

	template<>
	struct block_gevmt_float<cpu_sse3 | cpu_fma>
	{
		// C(1x4) += A(1xp) * B(4xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			__m128 xmm_a0;
			__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;
			__m128 xmm_c0 = _mm_loadu_ps(c);

			if (aligned_p > 0)
			{
				xmm_t0 = xmm_t1 = xmm_t2 = xmm_t3 = _mm_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 4)
				{
					// load data from memory
					xmm_a0 = _mm_loadu_ps(a + k);
					xmm_b0 = _mm_loadu_ps(ptr_b0 + k);
					xmm_b1 = _mm_loadu_ps(ptr_b1 + k);
					xmm_b2 = _mm_loadu_ps(ptr_b2 + k);
					xmm_b3 = _mm_loadu_ps(ptr_b3 + k);
					// return the weighted sum
					xmm_t0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_t0);
					xmm_t1 = _mm_fmadd_ps(xmm_a0, xmm_b1, xmm_t1);
					xmm_t2 = _mm_fmadd_ps(xmm_a0, xmm_b2, xmm_t2);
					xmm_t3 = _mm_fmadd_ps(xmm_a0, xmm_b3, xmm_t3);
				}
				// return the horizontal sum
				xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t1);
				xmm_t2 = _mm_hadd_ps(xmm_t2, xmm_t3);
				xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t2);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_t0);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					xmm_a0 = _mm_set1_ps(a[k]);
					xmm_b0 = _mm_set_ps(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					xmm_c0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_c0);
				}
			}
			// store data into memory
			_mm_storeu_ps(c, xmm_c0);
		}
	};

	template<>
	struct block_gevmt_float<cpu_avx>
	{
		// C(1x8) += A(1xp) * B(8xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a0;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256 ymm_c0 = _mm256_loadu_ps(c);

			if (aligned_p > 0)
			{
				ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = ymm_t4 = ymm_t5 = ymm_t6 = ymm_t7 = _mm256_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 8)
				{
					ymm_a0 = _mm256_loadu_ps(a + k);
					ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
					ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
					ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
					ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
					ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
					// return the weighted sum
					ymm_t0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_t0);
					ymm_t1 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b1), ymm_t1);
					ymm_t2 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b2), ymm_t2);
					ymm_t3 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b3), ymm_t3);
					ymm_t4 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b4), ymm_t4);
					ymm_t5 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b5), ymm_t5);
					ymm_t6 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b6), ymm_t6);
					ymm_t7 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b7), ymm_t7);
				}
				// return the horizontal sum
				ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t1);
				ymm_t2 = _mm256_hadd_ps(ymm_t2, ymm_t3);
				ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t5);
				ymm_t6 = _mm256_hadd_ps(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t2);
				ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t6);
				ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t5 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t0 = _mm256_add_ps(ymm_t1, ymm_t5);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t0);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_ps(a[k]);
					ymm_b0 = _mm256_set_ps(ptr_b7[k], ptr_b6[k], ptr_b5[k], ptr_b4[k], ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_c0);
				}
			}
			// store data into memory
			_mm256_storeu_ps(c, ymm_c0);
		}
	};

	template<>
	struct block_gevmt_float<cpu_avx | cpu_fma>
	{
		// C(1x8) += A(1xp) * B(8xp)^T
		void operator()(size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			const float *ptr_b0 = b;
			const float *ptr_b1 = b + rsb;
			const float *ptr_b2 = ptr_b1 + rsb;
			const float *ptr_b3 = ptr_b2 + rsb;
			const float *ptr_b4 = ptr_b3 + rsb;
			const float *ptr_b5 = ptr_b4 + rsb;
			const float *ptr_b6 = ptr_b5 + rsb;
			const float *ptr_b7 = ptr_b6 + rsb;
			__m256 ymm_a0;
			__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;
			__m256 ymm_c0 = _mm256_loadu_ps(c);

			if (aligned_p > 0)
			{
				ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = ymm_t4 = ymm_t5 = ymm_t6 = ymm_t7 = _mm256_setzero_ps();
				for (size_t k = 0; k < aligned_p; k += 8)
				{
					ymm_a0 = _mm256_loadu_ps(a + k);
					ymm_b0 = _mm256_loadu_ps(ptr_b0 + k);
					ymm_b1 = _mm256_loadu_ps(ptr_b1 + k);
					ymm_b2 = _mm256_loadu_ps(ptr_b2 + k);
					ymm_b3 = _mm256_loadu_ps(ptr_b3 + k);
					ymm_b4 = _mm256_loadu_ps(ptr_b4 + k);
					ymm_b5 = _mm256_loadu_ps(ptr_b5 + k);
					ymm_b6 = _mm256_loadu_ps(ptr_b6 + k);
					ymm_b7 = _mm256_loadu_ps(ptr_b7 + k);
					// return the weighted sum
					ymm_t0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_t0);
					ymm_t1 = _mm256_fmadd_ps(ymm_a0, ymm_b1, ymm_t1);
					ymm_t2 = _mm256_fmadd_ps(ymm_a0, ymm_b2, ymm_t2);
					ymm_t3 = _mm256_fmadd_ps(ymm_a0, ymm_b3, ymm_t3);
					ymm_t4 = _mm256_fmadd_ps(ymm_a0, ymm_b4, ymm_t4);
					ymm_t5 = _mm256_fmadd_ps(ymm_a0, ymm_b5, ymm_t5);
					ymm_t6 = _mm256_fmadd_ps(ymm_a0, ymm_b6, ymm_t6);
					ymm_t7 = _mm256_fmadd_ps(ymm_a0, ymm_b7, ymm_t7);
				}
				// return the horizontal sum
				ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t1);
				ymm_t2 = _mm256_hadd_ps(ymm_t2, ymm_t3);
				ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t5);
				ymm_t6 = _mm256_hadd_ps(ymm_t6, ymm_t7);
				ymm_t0 = _mm256_hadd_ps(ymm_t0, ymm_t2);
				ymm_t4 = _mm256_hadd_ps(ymm_t4, ymm_t6);
				ymm_t1 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 2, 0, 0));
				ymm_t5 = _mm256_permute2f128_ps(ymm_t0, ymm_t4, _MM_SHUFFLE(0, 3, 0, 1));
				ymm_t0 = _mm256_add_ps(ymm_t1, ymm_t5);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_t0);
			}
			if (aligned_p < p)
			{
				for (size_t k = aligned_p; k < p; ++k)
				{
					// load data from memory
					ymm_a0 = _mm256_set1_ps(a[k]);
					ymm_b0 = _mm256_set_ps(ptr_b7[k], ptr_b6[k], ptr_b5[k], ptr_b4[k], ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
					// return the weighted sum
					ymm_c0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_c0);
				}
			}
			// store data into memory
			_mm256_storeu_ps(c, ymm_c0);
		}
	};

} // namespace core

#endif
