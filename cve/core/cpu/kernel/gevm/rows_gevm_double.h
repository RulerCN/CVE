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

#ifndef __CORE_CPU_KERNEL_ROWS_GEVM_DOUBLE_H__
#define __CORE_CPU_KERNEL_ROWS_GEVM_DOUBLE_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_gevm_double
	template<cpu_inst_type inst>
	struct rows_gevm_double
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			double val_a;

			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				for (size_t j = 0; j < aligned_n;)
				{
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevm_double<cpu_sse3>
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			double val_a;
			__m128d xmm_a, xmm_b, xmm_c;

			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				xmm_a = _mm_set1_pd(val_a);
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_b = _mm_loadu_pd(b + j);
					xmm_c = _mm_loadu_pd(c + j);
					// return the weighted sum
					xmm_c = _mm_add_pd(_mm_mul_pd(xmm_a, xmm_b), xmm_c);
					// store data into memory
					_mm_storeu_pd(c + j, xmm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevm_double<cpu_sse3 | cpu_fma>
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			double val_a;
			__m128d xmm_a, xmm_b, xmm_c;

			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				xmm_a = _mm_set1_pd(val_a);
				for (size_t j = 0; j < aligned_n; j += 2)
				{
					// load data from memory
					xmm_b = _mm_loadu_pd(b + j);
					xmm_c = _mm_loadu_pd(c + j);
					// return the weighted sum
					xmm_c = _mm_fmadd_pd(xmm_a, xmm_b, xmm_c);
					// store data into memory
					_mm_storeu_pd(c + j, xmm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevm_double<cpu_avx>
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			double val_a;
			__m256d ymm_a, ymm_b, ymm_c;

			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				ymm_a = _mm256_set1_pd(val_a);
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_b = _mm256_loadu_pd(b + j);
					ymm_c = _mm256_loadu_pd(c + j);
					// return the weighted sum
					ymm_c = _mm256_add_pd(_mm256_mul_pd(ymm_a, ymm_b), ymm_c);
					// store data into memory
					_mm256_storeu_pd(c + j, ymm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevm_double<cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const double *a, const double *b, size_t rsb, double *c) const
		{
			double val_a;
			__m256d ymm_a, ymm_b, ymm_c;

			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				ymm_a = _mm256_set1_pd(val_a);
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					ymm_b = _mm256_loadu_pd(b + j);
					ymm_c = _mm256_loadu_pd(c + j);
					// return the weighted sum
					ymm_c = _mm256_fmadd_pd(ymm_a, ymm_b, _mm256_loadu_pd(c + j));
					// store data into memory
					_mm256_storeu_pd(c + j, ymm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

} // namespace core

#endif
