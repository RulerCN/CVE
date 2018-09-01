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

#ifndef __CORE_CPU_KERNEL_BLOCK_GEVM_DOUBLE_H__
#define __CORE_CPU_KERNEL_BLOCK_GEVM_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template block_gevm_double

	template<cpu_inst_type inst>
	struct block_gevm_double
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double val_a0 = a[0];
			double val_a1 = a[1];
			double val_a2 = a[2];
			double val_a3 = a[3];
			double val_t0, val_t1, val_t2, val_t3;

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
	struct block_gevm_double<cpu_sse3>
	{
		// C(1xn) += A(1x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_c0;

			if (aligned_n > 0)
			{
				xmm_a0 = _mm_set1_pd(a[0]);
				xmm_a1 = _mm_set1_pd(a[1]);
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
					xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
					xmm_c0 = _mm_loadu_pd(c + j);
					// return the weighted sum
					xmm_t0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_t1 = _mm_mul_pd(xmm_a1, xmm_b1);
					xmm_t0 = _mm_add_pd(xmm_t0, xmm_t1);
					xmm_c0 = _mm_add_pd(xmm_c0, xmm_t0);
					// store data into memory
					_mm_storeu_pd(c + j, xmm_c0);
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_t0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t0);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_double<cpu_sse3 | cpu_fma>
	{
		// C(1xn) += A(1x2) * B(2xn)
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t0;
			__m128d xmm_c0;

			if (aligned_n > 0)
			{
					xmm_a0 = _mm_set1_pd(a[0]);
					xmm_a1 = _mm_set1_pd(a[1]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b0 + j);
						xmm_b1 = _mm_loadu_pd(ptr_b1 + j);
						xmm_c0 = _mm_loadu_pd(c + j);
						// return the weighted sum
						xmm_t0 = _mm_mul_pd(xmm_a0, xmm_b0);
						xmm_t0 = _mm_fmadd_pd(xmm_a1, xmm_b1, xmm_t0);
						xmm_c0 = _mm_add_pd(xmm_c0, xmm_t0);
						// store data into memory
						_mm_storeu_pd(c + j, xmm_c0);
				}
			}
			if (aligned_n < n)
			{
				xmm_a0 = _mm_loadu_pd(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					xmm_b0 = _mm_set_pd(ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					xmm_t0 = _mm_mul_pd(xmm_a0, xmm_b0);
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t0);
					// store data into memory
					c[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_double<cpu_avx>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_c0;

			if (aligned_n > 0)
			{
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
						ymm_c0 = _mm256_loadu_pd(c + j);
						// return the weighted sum
						ymm_t0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						ymm_t1 = _mm256_mul_pd(ymm_a1, ymm_b1);
						ymm_t2 = _mm256_mul_pd(ymm_a2, ymm_b2);
						ymm_t3 = _mm256_mul_pd(ymm_a3, ymm_b3);
						ymm_t0 = _mm256_add_pd(ymm_t0, ymm_t1);
						ymm_t2 = _mm256_add_pd(ymm_t2, ymm_t3);
						ymm_t0 = _mm256_add_pd(ymm_t0, ymm_t2);
						ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t0);
						// store data into memory
						_mm256_storeu_pd(c + j, ymm_c0);
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_t0 = _mm256_hadd_pd(ymm_t0, ymm_t0);
					ymm_t1 = _mm256_permute2f128_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_t0 = _mm256_add_pd(ymm_t0, ymm_t1);
					// store data into memory
					c[j] += reinterpret_cast<double*>(&ymm_t0)[0];
				}
			}
		}
	};

	template<>
	struct block_gevm_double<cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1x4) * B(4xn)
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t0, ymm_t1;
			__m256d ymm_c0;

			if (aligned_n > 0)
			{
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
					ymm_c0 = _mm256_loadu_pd(c + j);
					// return the weighted sum
					ymm_t0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_t1 = _mm256_mul_pd(ymm_a1, ymm_b1);
					ymm_t0 = _mm256_fmadd_pd(ymm_a2, ymm_b2, ymm_t0);
					ymm_t1 = _mm256_fmadd_pd(ymm_a3, ymm_b3, ymm_t1);
					ymm_t0 = _mm256_add_pd(ymm_t0, ymm_t1);
					ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t0);
					// store data into memory
					_mm256_storeu_pd(c + j, ymm_c0);
				}
			}
			if (aligned_n < n)
			{
				ymm_a0 = _mm256_loadu_pd(a);
				for (size_t j = aligned_n; j < n; ++j)
				{
					// load data from memory
					ymm_b0 = _mm256_set_pd(ptr_b3[j], ptr_b2[j], ptr_b1[j], ptr_b0[j]);
					// return the horizontal weighted sum
					ymm_t0 = _mm256_mul_pd(ymm_a0, ymm_b0);
					ymm_t0 = _mm256_hadd_pd(ymm_t0, ymm_t0);
					ymm_t1 = _mm256_permute2f128_pd(ymm_t0, ymm_t0, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_t0 = _mm256_add_pd(ymm_t0, ymm_t1);
					// store data into memory
					c[j] += reinterpret_cast<double*>(&ymm_t0)[0];
				}
			}
		}
	};

} // namespace core

#endif
