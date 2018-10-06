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

#ifndef __CORE_CPU_KERNEL_COLUMNS_GEMM_DOUBLE_H__
#define __CORE_CPU_KERNEL_COLUMNS_GEMM_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template columns_gemm_double

	template<cpu_inst_type inst>
	struct columns_gemm_double
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t /*aligned_n*/, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			double val_a0, val_a1, val_a2, val_a3;
			double val_b0;

			for (size_t k = 0; k < p; ++k)
			{
				val_a0 = ptr_a0[k];
				val_a1 = ptr_a1[k];
				val_a2 = ptr_a2[k];
				val_a3 = ptr_a3[k];
				for (size_t j = 0; j < n; ++j)
				{
					val_b0 = b[j];
					ptr_c0[j] += val_a0 * val_b0;
					ptr_c1[j] += val_a1 * val_b0;
					ptr_c2[j] += val_a2 * val_b0;
					ptr_c3[j] += val_a3 * val_b0;
				}
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemm_double<cpu_sse3>
	{
		// C(2xn) += A(2xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_b;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set1_pd(ptr_a0[k]);
					xmm_a1 = _mm_set1_pd(ptr_a1[k]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b + j);
						xmm_c0 = _mm_loadu_pd(ptr_c0 + j);
						xmm_c1 = _mm_loadu_pd(ptr_c1 + j);
						// return the weighted sum
						xmm_c0 = _mm_add_pd(_mm_mul_pd(xmm_a0, xmm_b0), xmm_c0);
						xmm_c1 = _mm_add_pd(_mm_mul_pd(xmm_a1, xmm_b0), xmm_c1);
						// store data into memory
						_mm_storeu_pd(ptr_c0 + j, xmm_c0);
						_mm_storeu_pd(ptr_c1 + j, xmm_c1);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set_pd(ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_pd(ptr_b[j]);
						// return the product
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<double*>(&xmm_c0)[0];
						ptr_c1[j] += reinterpret_cast<double*>(&xmm_c0)[1];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_double<cpu_sse3 | cpu_fma>
	{
		// C(2xn) += A(2xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_b;
			__m128d xmm_a0, xmm_a1;
			__m128d xmm_b0;
			__m128d xmm_c0, xmm_c1;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set1_pd(ptr_a0[k]);
					xmm_a1 = _mm_set1_pd(ptr_a1[k]);
					for (size_t j = 0; j < aligned_n; j += 2)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_pd(ptr_b + j);
						xmm_c0 = _mm_loadu_pd(ptr_c0 + j);
						xmm_c1 = _mm_loadu_pd(ptr_c1 + j);
						// return the weighted sum
						xmm_c0 = _mm_fmadd_pd(xmm_a0, xmm_b0, xmm_c0);
						xmm_c1 = _mm_fmadd_pd(xmm_a1, xmm_b0, xmm_c1);
						// store data into memory
						_mm_storeu_pd(ptr_c0 + j, xmm_c0);
						_mm_storeu_pd(ptr_c1 + j, xmm_c1);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set_pd(ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_pd(ptr_b[j]);
						// return the product
						xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<double*>(&xmm_c0)[0];
						ptr_c1[j] += reinterpret_cast<double*>(&xmm_c0)[1];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_double<cpu_avx>
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_b;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_pd(ptr_a0[k]);
					ymm_a1 = _mm256_set1_pd(ptr_a1[k]);
					ymm_a2 = _mm256_set1_pd(ptr_a2[k]);
					ymm_a3 = _mm256_set1_pd(ptr_a3[k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b + j);
						ymm_c0 = _mm256_loadu_pd(ptr_c0 + j);
						ymm_c1 = _mm256_loadu_pd(ptr_c1 + j);
						ymm_c2 = _mm256_loadu_pd(ptr_c2 + j);
						ymm_c3 = _mm256_loadu_pd(ptr_c3 + j);
						// return the weighted sum
						ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_c0);
						ymm_c1 = _mm256_add_pd(_mm256_mul_pd(ymm_a1, ymm_b0), ymm_c1);
						ymm_c2 = _mm256_add_pd(_mm256_mul_pd(ymm_a2, ymm_b0), ymm_c2);
						ymm_c3 = _mm256_add_pd(_mm256_mul_pd(ymm_a3, ymm_b0), ymm_c3);
						// store data into memory
						_mm256_storeu_pd(ptr_c0 + j, ymm_c0);
						_mm256_storeu_pd(ptr_c1 + j, ymm_c1);
						_mm256_storeu_pd(ptr_c2 + j, ymm_c2);
						_mm256_storeu_pd(ptr_c3 + j, ymm_c3);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set_pd(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_pd(ptr_b[j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c1[j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c2[j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c3[j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_double<cpu_avx | cpu_fma>
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			double *ptr_c0 = c;
			double *ptr_c1 = c + rsc;
			double *ptr_c2 = ptr_c1 + rsc;
			double *ptr_c3 = ptr_c2 + rsc;
			const double *ptr_a0 = a;
			const double *ptr_a1 = a + rsa;
			const double *ptr_a2 = ptr_a1 + rsa;
			const double *ptr_a3 = ptr_a2 + rsa;
			const double *ptr_b;
			__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
			__m256d ymm_b0;
			__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_pd(ptr_a0[k]);
					ymm_a1 = _mm256_set1_pd(ptr_a1[k]);
					ymm_a2 = _mm256_set1_pd(ptr_a2[k]);
					ymm_a3 = _mm256_set1_pd(ptr_a3[k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_pd(ptr_b + j);
						ymm_c0 = _mm256_loadu_pd(ptr_c0 + j);
						ymm_c1 = _mm256_loadu_pd(ptr_c1 + j);
						ymm_c2 = _mm256_loadu_pd(ptr_c2 + j);
						ymm_c3 = _mm256_loadu_pd(ptr_c3 + j);
						// return the weighted sum
						ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_c0);
						ymm_c1 = _mm256_fmadd_pd(ymm_a1, ymm_b0, ymm_c1);
						ymm_c2 = _mm256_fmadd_pd(ymm_a2, ymm_b0, ymm_c2);
						ymm_c3 = _mm256_fmadd_pd(ymm_a3, ymm_b0, ymm_c3);
						// store data into memory
						_mm256_storeu_pd(ptr_c0 + j, ymm_c0);
						_mm256_storeu_pd(ptr_c1 + j, ymm_c1);
						_mm256_storeu_pd(ptr_c2 + j, ymm_c2);
						_mm256_storeu_pd(ptr_c3 + j, ymm_c3);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set_pd(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_pd(ptr_b[j]);
						// return the product
						ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<double*>(&ymm_c0)[0];
						ptr_c1[j] += reinterpret_cast<double*>(&ymm_c0)[1];
						ptr_c2[j] += reinterpret_cast<double*>(&ymm_c0)[2];
						ptr_c3[j] += reinterpret_cast<double*>(&ymm_c0)[3];
					}
					ptr_b += rsb;
				}
			}
		}
	};

} // namespace core

#endif
