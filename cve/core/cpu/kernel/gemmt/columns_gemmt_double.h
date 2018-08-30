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

#ifndef __CORE_CPU_KERNEL_COLUMNS_GEMMT_DOUBLE_H__
#define __CORE_CPU_KERNEL_COLUMNS_GEMMT_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template columns_gemmt_double
	
	template<cpu_inst_type inst>
	struct columns_gemmt_double
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			double val_b0;
			double val_t0, val_t1, val_t2, val_t3;

			for (size_t j = 0; j < n; ++j)
			{
				val_t0 = val_t1 = val_t2 = val_t3 = 0;
				for (size_t k = 0; k < p; ++k)
				{
					val_b0 = b[k];
					val_t0 += ptr_a0[k] * val_b0;
					val_t1 += ptr_a1[k] * val_b0;
					val_t2 += ptr_a2[k] * val_b0;
					val_t3 += ptr_a3[k] * val_b0;
				}
				ptr_c0[j] += val_t0;
				ptr_c1[j] += val_t1;
				ptr_c2[j] += val_t2;
				ptr_c3[j] += val_t3;
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_double<cpu_sse3>
	{
		// C(2xn) += A(2xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_c0;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = _mm_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 2)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
						xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
						xmm_b0 = _mm_loadu_pd(b + k);
						// return the weighted sum
						xmm_t0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_t0);
						xmm_t1 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b0), xmm_t1);
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
						xmm_a0 = _mm_set_pd(ptr_a1[k], ptr_a0[k]);
						xmm_b0 = _mm_set1_pd(b[k]);
						// return the weighted sum
						xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_c0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<double*>(&xmm_c0)[0];
				ptr_c1[j] += reinterpret_cast<double*>(&xmm_c0)[1];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_double<cpu_sse3 | cpu_fma>
	{
		// C(2xn) += A(2xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0;
			__m128d xmm_t0, xmm_t1;
			__m128d xmm_c0;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = _mm_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 4)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_pd(ptr_a0 + k);
						xmm_a1 = _mm_loadu_pd(ptr_a1 + k);
						xmm_b0 = _mm_loadu_pd(b + k);
						// return the weighted sum
						xmm_t0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_t0);
						xmm_t1 = _mm_fmadd_pd(xmm_a1, xmm_b0, xmm_t1);
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
						xmm_a0 = _mm_set_pd(ptr_a1[k], ptr_a0[k]);
						xmm_b0 = _mm_set1_pd(b[k]);
						// return the weighted sum
						xmm_c0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_c0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<double*>(&xmm_c0)[0];
				ptr_c1[j] += reinterpret_cast<double*>(&xmm_c0)[1];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_double<cpu_avx>
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_c0;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = _mm256_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
						ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
						ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
						ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
						ymm_b0 = _mm256_loadu_pd(b + k);
						// return the weighted sum
						ymm_t0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_t0);
						ymm_t1 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b0), ymm_t1);
						ymm_t2 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b0), ymm_t2);
						ymm_t3 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b0), ymm_t3);
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
						ymm_a0 = _mm256_set_pd(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						ymm_b0 = _mm256_set1_pd(b[k]);
						// return the weighted sum
						ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_c0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<double*>(&ymm_c0)[0];
				ptr_c1[j] += reinterpret_cast<double*>(&ymm_c0)[1];
				ptr_c2[j] += reinterpret_cast<double*>(&ymm_c0)[2];
				ptr_c3[j] += reinterpret_cast<double*>(&ymm_c0)[3];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_double<cpu_avx | cpu_fma>
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0;
			__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;
			__m256d ymm_c0;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = _mm256_setzero_pd();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_pd(ptr_a0 + k);
						ymm_a1 = _mm256_loadu_pd(ptr_a1 + k);
						ymm_a2 = _mm256_loadu_pd(ptr_a2 + k);
						ymm_a3 = _mm256_loadu_pd(ptr_a3 + k);
						ymm_b0 = _mm256_loadu_pd(b + k);
						// return the weighted sum
						ymm_t0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_t0);
						ymm_t1 = _mm256_fmadd_pd(ymm_a1, ymm_b0, ymm_t1);
						ymm_t2 = _mm256_fmadd_pd(ymm_a2, ymm_b0, ymm_t2);
						ymm_t3 = _mm256_fmadd_pd(ymm_a3, ymm_b0, ymm_t3);
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
						ymm_a0 = _mm256_set_pd(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						ymm_b0 = _mm256_set1_pd(b[k]);
						// return the weighted sum
						ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_c0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<double*>(&ymm_c0)[0];
				ptr_c1[j] += reinterpret_cast<double*>(&ymm_c0)[1];
				ptr_c2[j] += reinterpret_cast<double*>(&ymm_c0)[2];
				ptr_c3[j] += reinterpret_cast<double*>(&ymm_c0)[3];
				b += rsb;
			}
		}
	};

} // namespace core

#endif
