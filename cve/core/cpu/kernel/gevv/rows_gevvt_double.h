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

#ifndef __CORE_CPU_KERNEL_ROWS_GEVVT_DOUBLE_H__
#define __CORE_CPU_KERNEL_ROWS_GEVVT_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_gevvt_double
	template<cpu_inst_type inst>
	struct rows_gevvt_double
	{
		// C(1x1) += A(1xn) * B(1xn)^T
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, double *c) const
		{
			const double *ptr_a = a;
			const double *ptr_b = b;
			double val_c0, val_c1, val_c2, val_c3;

			val_c0 = 0.0;
			if (aligned_n > 0)
			{
				val_c1 = val_c2 = val_c3 = val_c0;
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					val_c0 += ptr_a[0] * ptr_b[0];
					val_c1 += ptr_a[1] * ptr_b[1];
					val_c2 += ptr_a[2] * ptr_b[2];
					val_c3 += ptr_a[3] * ptr_b[3];
					ptr_a += 4;
					ptr_b += 4;
				}
				val_c0 += val_c1;
				val_c2 += val_c3;
				val_c0 += val_c2;
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
					val_c0 += a[j] * b[j];
			}
			*c += val_c0;
		}
	};

	template<>
	struct rows_gevvt_double<cpu_sse3>
	{
		// C(1x1) += A(1xn) * B(1xn)^T
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, double *c) const
		{
			double val_c = 0.0;
			__m128d xmm_a, xmm_b, xmm_c;

			if (aligned_n > 0)
			{
				xmm_c = _mm_setzero_pd();
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_a = _mm_loadu_pd(a + j);
					xmm_b = _mm_loadu_pd(b + j);
					// return the weighted sum
					xmm_c = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b), xmm_c);
				}
				// return the horizontal summation
				xmm_c = _mm_hadd_pd(xmm_c, xmm_c);
				val_c += reinterpret_cast<double*>(&xmm_c)[0];
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
					val_c += a[j] * b[j];
			}
			// store data into memory
			*c += val_c;
		}
	};

	template<>
	struct rows_gevvt_double<cpu_sse3 | cpu_fma>
	{
		// C(1x1) += A(1xn) * B(1xn)^T
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, double *c) const
		{
			double val_c = 0.0;
			__m128d xmm_a, xmm_b, xmm_c;

			if (aligned_n > 0)
			{
				xmm_c = _mm_setzero_pd();
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_a = _mm_loadu_pd(a + j);
					xmm_b = _mm_loadu_pd(b + j);
					// return the weighted sum
					xmm_c = _mm_fmadd_pd(xmm_a, xmm_b, xmm_c);
				}
				// return the horizontal summation
				xmm_c = _mm_hadd_pd(xmm_c, xmm_c);
				val_c += reinterpret_cast<double*>(&xmm_c)[0];
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
					val_c += a[j] * b[j];
			}
			// store data into memory
			*c += val_c;
		}
	};

	template<>
	struct rows_gevvt_double<cpu_avx>
	{
		// C(1x1) += A(1xn) * B(1xn)^T
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, double *c) const
		{
			double val_c = 0.0;
			__m256d ymm_a0, ymm_b0;
			__m256d ymm_c0, ymm_c1;

			if (aligned_n > 0)
			{
				ymm_c0 = _mm256_setzero_pd();
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(a + j);
					ymm_b0 = _mm256_loadu_pd(b + j);
					// return the weighted sum
					ymm_c0 = _mm256_add_pd(_mm256_mul_pd(ymm_a0, ymm_b0), ymm_c0);
				}
				// return the horizontal summation
				ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
				ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
				val_c += reinterpret_cast<double*>(&ymm_c0)[0];
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
					val_c += a[j] * b[j];
			}
			// store data into memory
			*c += val_c;
		}
	};

	template<>
	struct rows_gevvt_double<cpu_avx | cpu_fma>
	{
		// C(1x1) += A(1xn) * B(1xn)^T
		void operator()(size_t aligned_n, size_t n, const double *a, const double *b, double *c) const
		{
			double val_c = 0.0;
			__m256d ymm_a0, ymm_b0;
			__m256d ymm_c0, ymm_c1;

			if (aligned_n > 0)
			{
				ymm_c0 = _mm256_setzero_pd();
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_a0 = _mm256_loadu_pd(a + j);
					ymm_b0 = _mm256_loadu_pd(b + j);
					// return the weighted sum
					ymm_c0 = _mm256_fmadd_pd(ymm_a0, ymm_b0, ymm_c0);
				}
				// return the horizontal summation
				ymm_c0 = _mm256_hadd_pd(ymm_c0, ymm_c0);
				ymm_c1 = _mm256_permute2f128_pd(ymm_c0, ymm_c0, _MM_SHUFFLE(0, 2, 0, 1));
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
				val_c += reinterpret_cast<double*>(&ymm_c0)[0];
			}
			if (aligned_n < n)
			{
				for (size_t j = aligned_n; j < n; ++j)
					val_c += a[j] * b[j];
			}
			// store data into memory
			*c += val_c;
		}
	};

} // namespace core

#endif
