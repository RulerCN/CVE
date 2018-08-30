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

#ifndef __CORE_CPU_KERNEL_ROWS_GEMMT_DOUBLE_H__
#define __CORE_CPU_KERNEL_ROWS_GEMMT_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_gemmt_double
	
	template<cpu_inst_type inst>
	struct rows_gemmt_double
	{
		// C(mx4) += A(mxp) * B(4xp)^T
		void operator()(size_t m, size_t, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			double val_a0;
			double val_t0, val_t1, val_t2, val_t3;

			for (size_t i = 0; i < m; ++i)
			{
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
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gemmt_double<cpu_sse3>
	{
		// C(mx2) += A(mxp) * B(2xp)^T
		void operator()(size_t m, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_c0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_c0 = _mm_loadu_pd(c);
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = _mm_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 2)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_pd(a + k);
						xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
						xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
						// return the weighted sum
						xmm_t0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_t0);
						xmm_t1 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b1), xmm_t1);
					}
					// return the horizontal sum
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t1);
					xmm_c0 = _mm_add_pd(xmm_c0, xmm_t0);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						xmm_a0 = _mm_set1_pd(a[k]);
						xmm_b0 = _mm_set_pd(ptr_b1[k], ptr_b0[k]);
						// return the weighted sum
						xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_c0);
					}
				}
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gemmt_double<cpu_sse3 | cpu_fma>
	{
		// C(mx2) += A(mxp) * B(2xp)^T
		void operator()(size_t m, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0, xmm_b1;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_c0;

			for (size_t i = 0; i < m; ++i)
			{
				xmm_c0 = _mm_loadu_pd(c);
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = _mm_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 2)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_pd(a + k);
						xmm_b0 = _mm_loadu_pd(ptr_b0 + k);
						xmm_b1 = _mm_loadu_pd(ptr_b1 + k);
						// return the weighted sum
						xmm_t0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_t0);
						xmm_t1 = _mm_fmadd_pd(xmm_a0, xmm_b1, xmm_t1);
					}
					// return the horizontal sum
					xmm_t0 = _mm_hadd_pd(xmm_t0, xmm_t1);
					xmm_c0 = _mm_add_pd(xmm_c0, xmm_t0);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						xmm_a0 = _mm_set1_pd(a[k]);
						xmm_b0 = _mm_set_pd(ptr_b1[k], ptr_b0[k]);
						// return the weighted sum
						xmm_c0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_c0);
					}
				}
				// store data into memory
				_mm_storeu_pd(c, xmm_c0);
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gemmt_double<cpu_avx>
	{
		// C(mx4) += A(mxp) * B(4xp)^T
		void operator()(size_t m, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_c0;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_c0 = _mm256_loadu_pd(c);
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = _mm256_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_pd(a + k);
						ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
						ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
						ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
						ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
						// return the weighted sum
						ymm_t0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_t0);
						ymm_t1 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b1), ymm_t1);
						ymm_t2 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b2), ymm_t2);
						ymm_t3 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b3), ymm_t3);
					}
					// return the horizontal sum
					ymm_t0 = _mm256_hadd_pd(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_pd(ymm_t2, ymm_t3);
					ymm_t1 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t3 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_pd(ymm_t1, ymm_t3);
					ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t0);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						ymm_a0 = _mm256_set1_pd(a[k]);
						ymm_b0 = _mm256_set_pd(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
						// return the weighted sum
						ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_c0);
					}
				}
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				a += rsa;
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gemmt_double<cpu_avx | cpu_fma>
	{
		// C(mx4) += A(mxp) * B(4xp)^T
		void operator()(size_t m, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const double *ptr_b0 = b;
			const double *ptr_b1 = b + rsb;
			const double *ptr_b2 = ptr_b1 + rsb;
			const double *ptr_b3 = ptr_b2 + rsb;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_c0;

			for (size_t i = 0; i < m; ++i)
			{
				ymm_c0 = _mm256_loadu_pd(c);
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = _mm256_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_pd(a + k);
						ymm_b0 = _mm256_loadu_pd(ptr_b0 + k);
						ymm_b1 = _mm256_loadu_pd(ptr_b1 + k);
						ymm_b2 = _mm256_loadu_pd(ptr_b2 + k);
						ymm_b3 = _mm256_loadu_pd(ptr_b3 + k);
						// return the weighted sum
						ymm_t0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_t0);
						ymm_t1 = _mm256_fmadd_pd(ymm_a0, ymm_b1, ymm_t1);
						ymm_t2 = _mm256_fmadd_pd(ymm_a0, ymm_b2, ymm_t2);
						ymm_t3 = _mm256_fmadd_pd(ymm_a0, ymm_b3, ymm_t3);
					}
					// return the horizontal sum
					ymm_t0 = _mm256_hadd_pd(ymm_t0, ymm_t1);
					ymm_t2 = _mm256_hadd_pd(ymm_t2, ymm_t3);
					ymm_t1 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 2, 0, 0));
					ymm_t3 = _mm256_permute2f128_pd(ymm_t0, ymm_t2, _MM_SHUFFLE(0, 3, 0, 1));
					ymm_t0 = _mm256_add_pd(ymm_t1, ymm_t3);
					ymm_c0 = _mm256_add_pd(ymm_c0, ymm_t0);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						ymm_a0 = _mm256_set1_pd(a[k]);
						ymm_b0 = _mm256_set_pd(ptr_b3[k], ptr_b2[k], ptr_b1[k], ptr_b0[k]);
						// return the weighted sum
						ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_c0);
					}
				}
				// store data into memory
				_mm256_storeu_pd(c, ymm_c0);
				a += rsa;
				c += rsc;
			}
		}
	};

} // namespace core

#endif
