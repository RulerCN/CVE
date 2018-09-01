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

#ifndef __CORE_CPU_KERNEL_ROWS_GEVMT_FLOAT_H__
#define __CORE_CPU_KERNEL_ROWS_GEVMT_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template rows_gevmt_float
	template<cpu_inst_type inst>
	struct rows_gevmt_float
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			float val_c;

			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				for (size_t k = 0; k < aligned_p;)
				{
					val_c += a[k] * b[k]; ++k;
					val_c += a[k] * b[k]; ++k;
					val_c += a[k] * b[k]; ++k;
					val_c += a[k] * b[k]; ++k;
				}
				for (size_t k = aligned_p; k < p; ++k)
					val_c += a[k] * b[k];
				c[j] += val_c;
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevmt_float<cpu_sse3>
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			float val_c;
			__m128 xmm_a, xmm_b, xmm_c;

			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				if (aligned_p > 0)
				{
					xmm_c = _mm_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_ps(a + k);
						xmm_b = _mm_loadu_ps(b + k);
						// return the weighted sum
						xmm_c = _mm_add_ps(_mm_mul_ps(xmm_a, xmm_b), xmm_c);
					}
					// return the horizontal sum
					xmm_c = _mm_hadd_ps(xmm_c, xmm_c);
					xmm_c = _mm_hadd_ps(xmm_c, xmm_c);
					val_c = reinterpret_cast<float*>(&xmm_c)[0];
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
						val_c += a[k] * b[k];
				}
				// store data into memory
				c[j] += val_c;
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevmt_float<cpu_sse3 | cpu_fma>
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			float val_c;
			__m128 xmm_a, xmm_b, xmm_c;

			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				if (aligned_p > 0)
				{
					xmm_c = _mm_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 4)
					{
						// load data from memory
						xmm_a = _mm_loadu_ps(a + k);
						xmm_b = _mm_loadu_ps(b + k);
						// return the weighted sum
						xmm_c = _mm_fmadd_ps(xmm_a, xmm_b, xmm_c);
					}
					// return the horizontal sum
					xmm_c = _mm_hadd_ps(xmm_c, xmm_c);
					xmm_c = _mm_hadd_ps(xmm_c, xmm_c);
					val_c = reinterpret_cast<float*>(&xmm_c)[0];
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
						val_c += a[k] * b[k];
				}
				// store data into memory
				c[j] += val_c;
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevmt_float<cpu_avx>
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			float val_c;
			__m256 ymm_a, ymm_b, ymm_c, ymm_t;

			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				if (aligned_p > 0)
				{
					ymm_c = _mm256_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						// load data from memory
						ymm_a = _mm256_loadu_ps(a + k);
						ymm_b = _mm256_loadu_ps(b + k);
						// return the weighted sum
						ymm_c = _mm256_add_ps(_mm256_mul_ps(ymm_a, ymm_b), ymm_c);
					}
					// return the horizontal summation
					ymm_c = _mm256_hadd_ps(ymm_c, ymm_c);
					ymm_c = _mm256_hadd_ps(ymm_c, ymm_c);
					ymm_t = _mm256_permute2f128_ps(ymm_c, ymm_c, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_c = _mm256_add_ps(ymm_c, ymm_t);
					val_c = reinterpret_cast<float*>(&ymm_c)[0];
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
						val_c += a[k] * b[k];
				}
				// store data into memory
				c[j] += val_c;
				b += rsb;
			}
		}
	};

	template<>
	struct rows_gevmt_float<cpu_avx | cpu_fma>
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, const float *b, size_t rsb, float *c) const
		{
			float val_c;
			__m256 ymm_a, ymm_b, ymm_c, ymm_t;

			for (size_t j = 0; j < n; ++j)
			{
				val_c = 0;
				if (aligned_p > 0)
				{
					ymm_c = _mm256_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						// load data from memory
						ymm_a = _mm256_loadu_ps(a + k);
						ymm_b = _mm256_loadu_ps(b + k);
						// return the weighted sum
						ymm_c = _mm256_fmadd_ps(ymm_a, ymm_b, ymm_c);
					}
					// return the horizontal summation
					ymm_c = _mm256_hadd_ps(ymm_c, ymm_c);
					ymm_c = _mm256_hadd_ps(ymm_c, ymm_c);
					ymm_t = _mm256_permute2f128_ps(ymm_c, ymm_c, _MM_SHUFFLE(0, 2, 0, 1));
					ymm_c = _mm256_add_ps(ymm_c, ymm_t);
					val_c = reinterpret_cast<float*>(&ymm_c)[0];
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
						val_c += a[k] * b[k];
				}
				// store data into memory
				c[j] += val_c;
				b += rsb;
			}
		}
	};

} // namespace core

#endif
