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

#ifndef __CORE_CPU_KERNEL_ROWS_GEVV_FLOAT_H__
#define __CORE_CPU_KERNEL_ROWS_GEVV_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_gevv_float
	template<cpu_inst_type inst>
	struct rows_gevv_float
	{
		// C(mxn) += A(mx1) * B(1xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, const float *b, float *c, size_t rsc) const
		{
			float val_a;

			for (size_t i = 0; i < m; ++i)
			{
				val_a = a[i];
				for (size_t j = 0; j < aligned_n;)
				{
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
					c[j] += val_a * b[j]; ++j;
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gevv_float<cpu_sse>
	{
		// C(mxn) += A(mx1) * B(1xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, const float *b, float *c, size_t rsc) const
		{
			float val_a;
			__m128 xmm_a, xmm_b, xmm_c;

			for (size_t i = 0; i < m; ++i)
			{
				val_a = a[i];
				xmm_a = _mm_set1_ps(val_a);
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_b = _mm_loadu_ps(b + j);
					xmm_c = _mm_loadu_ps(c + j);
					// return the weighted sum
					xmm_c = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b), xmm_c);
					// store data into memory
					_mm_storeu_ps(c + j, xmm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gevv_float<cpu_sse | cpu_fma>
	{
		// C(mxn) += A(mx1) * B(1xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, const float *b, float *c, size_t rsc) const
		{
			float val_a;
			__m128 xmm_a, xmm_b, xmm_c;

			for (size_t i = 0; i < m; ++i)
			{
				val_a = a[i];
				xmm_a = _mm_set1_ps(val_a);
				for (size_t j = 0; j < aligned_n; j += 4)
				{
					// load data from memory
					xmm_b = _mm_loadu_ps(b + j);
					xmm_c = _mm_loadu_ps(c + j);
					// return the weighted sum
					xmm_c = _mm_fmadd_ps(xmm_a, xmm_b, xmm_c);
					// store data into memory
					_mm_storeu_ps(c + j, xmm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gevv_float<cpu_avx>
	{
		// C(mxn) += A(mx1) * B(1xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, const float *b, float *c, size_t rsc) const
		{
			float val_a;
			__m256 ymm_a, ymm_b, ymm_c;

			for (size_t i = 0; i < m; ++i)
			{
				val_a = a[i];
				ymm_a = _mm256_set1_ps(val_a);
				for (size_t j = 0; j < aligned_n; j += 8)
				{
					// load data from memory
					ymm_b = _mm256_loadu_ps(b + j);
					ymm_c = _mm256_loadu_ps(c + j);
					// return the weighted sum
					ymm_c = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b), ymm_c);
					// store data into memory
					_mm256_storeu_ps(c + j, ymm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				c += rsc;
			}
		}
	};

	template<>
	struct rows_gevv_float<cpu_avx | cpu_fma>
	{
		// C(mxn) += A(mx1) * B(1xn)
		void operator()(size_t m, size_t aligned_n, size_t n, const float *a, const float *b, float *c, size_t rsc) const
		{
			float val_a;
			__m256 ymm_a, ymm_b, ymm_c;

			for (size_t i = 0; i < m; ++i)
			{
				val_a = a[i];
				ymm_a = _mm256_set1_ps(val_a);
				for (size_t j = 0; j < aligned_n; j += 8)
				{
					// load data from memory
					ymm_b = _mm256_loadu_ps(b + j);
					ymm_c = _mm256_loadu_ps(c + j);
					// return the weighted sum
					ymm_c = _mm256_fmadd_ps(ymm_a, ymm_b, ymm_c);
					// store data into memory
					_mm256_storeu_ps(c + j, ymm_c);
				}
				for (size_t j = aligned_n; j < n; ++j)
					c[j] += val_a * b[j];
				c += rsc;
			}
		}
	};

} // namespace core

#endif
