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

#ifndef __CORE_CPU_KERNEL_COLUMNS_GEMM_FLOAT_H__
#define __CORE_CPU_KERNEL_COLUMNS_GEMM_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template columns_gemm_float

	template<cpu_inst_type inst>
	struct columns_gemm_float
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			float val_a0, val_a1, val_a2, val_a3;
			float val_b0;

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
	struct columns_gemm_float<cpu_sse3>
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_b;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set1_ps(ptr_a0[k]);
					xmm_a1 = _mm_set1_ps(ptr_a1[k]);
					xmm_a2 = _mm_set1_ps(ptr_a2[k]);
					xmm_a3 = _mm_set1_ps(ptr_a3[k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b + j);
						xmm_c0 = _mm_loadu_ps(ptr_c0 + j);
						xmm_c1 = _mm_loadu_ps(ptr_c1 + j);
						xmm_c2 = _mm_loadu_ps(ptr_c2 + j);
						xmm_c3 = _mm_loadu_ps(ptr_c3 + j);
						// return the weighted sum
						xmm_c0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_c0);
						xmm_c1 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b0), xmm_c1);
						xmm_c2 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b0), xmm_c2);
						xmm_c3 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b0), xmm_c3);
						// store data into memory
						_mm_storeu_ps(ptr_c0 + j, xmm_c0);
						_mm_storeu_ps(ptr_c1 + j, xmm_c1);
						_mm_storeu_ps(ptr_c2 + j, xmm_c2);
						_mm_storeu_ps(ptr_c3 + j, xmm_c3);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set_ps(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_ps(ptr_b[j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c1[j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c2[j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c3[j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_float<cpu_sse3 | cpu_fma>
	{
		// C(4xn) += A(4xp) * B(pxn)
		void operator()(size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_b;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0;
			__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set1_ps(ptr_a0[k]);
					xmm_a1 = _mm_set1_ps(ptr_a1[k]);
					xmm_a2 = _mm_set1_ps(ptr_a2[k]);
					xmm_a3 = _mm_set1_ps(ptr_a3[k]);
					for (size_t j = 0; j < aligned_n; j += 4)
					{
						// load data from memory
						xmm_b0 = _mm_loadu_ps(ptr_b + j);
						xmm_c0 = _mm_loadu_ps(ptr_c0 + j);
						xmm_c1 = _mm_loadu_ps(ptr_c1 + j);
						xmm_c2 = _mm_loadu_ps(ptr_c2 + j);
						xmm_c3 = _mm_loadu_ps(ptr_c3 + j);
						// return the weighted sum
						xmm_c0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_c0);
						xmm_c1 = _mm_fmadd_ps(xmm_a1, xmm_b0, xmm_c1);
						xmm_c2 = _mm_fmadd_ps(xmm_a2, xmm_b0, xmm_c2);
						xmm_c3 = _mm_fmadd_ps(xmm_a3, xmm_b0, xmm_c3);
						// store data into memory
						_mm_storeu_ps(ptr_c0 + j, xmm_c0);
						_mm_storeu_ps(ptr_c1 + j, xmm_c1);
						_mm_storeu_ps(ptr_c2 + j, xmm_c2);
						_mm_storeu_ps(ptr_c3 + j, xmm_c3);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					xmm_a0 = _mm_set_ps(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						xmm_b0 = _mm_set1_ps(ptr_b[j]);
						// return the product
						xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<float*>(&xmm_c0)[0];
						ptr_c1[j] += reinterpret_cast<float*>(&xmm_c0)[1];
						ptr_c2[j] += reinterpret_cast<float*>(&xmm_c0)[2];
						ptr_c3[j] += reinterpret_cast<float*>(&xmm_c0)[3];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_float<cpu_avx>
	{
		// C(8xn) += A(8xp) * B(8xn)
		void operator()(size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			float *ptr_c4 = ptr_c3 + rsc;
			float *ptr_c5 = ptr_c4 + rsc;
			float *ptr_c6 = ptr_c5 + rsc;
			float *ptr_c7 = ptr_c6 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			const float *ptr_b;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_ps(ptr_a0[k]);
					ymm_a1 = _mm256_set1_ps(ptr_a1[k]);
					ymm_a2 = _mm256_set1_ps(ptr_a2[k]);
					ymm_a3 = _mm256_set1_ps(ptr_a3[k]);
					ymm_a4 = _mm256_set1_ps(ptr_a4[k]);
					ymm_a5 = _mm256_set1_ps(ptr_a5[k]);
					ymm_a6 = _mm256_set1_ps(ptr_a6[k]);
					ymm_a7 = _mm256_set1_ps(ptr_a7[k]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b + j);
						ymm_c0 = _mm256_loadu_ps(ptr_c0 + j);
						ymm_c1 = _mm256_loadu_ps(ptr_c1 + j);
						ymm_c2 = _mm256_loadu_ps(ptr_c2 + j);
						ymm_c3 = _mm256_loadu_ps(ptr_c3 + j);
						ymm_c4 = _mm256_loadu_ps(ptr_c4 + j);
						ymm_c5 = _mm256_loadu_ps(ptr_c5 + j);
						ymm_c6 = _mm256_loadu_ps(ptr_c6 + j);
						ymm_c7 = _mm256_loadu_ps(ptr_c7 + j);
						// return the weighted sum
						ymm_c0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_c0);
						ymm_c1 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b0), ymm_c1);
						ymm_c2 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b0), ymm_c2);
						ymm_c3 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b0), ymm_c3);
						ymm_c4 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b0), ymm_c4);
						ymm_c5 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b0), ymm_c5);
						ymm_c6 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b0), ymm_c6);
						ymm_c7 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b0), ymm_c7);
						// store data into memory
						_mm256_storeu_ps(ptr_c0 + j, ymm_c0);
						_mm256_storeu_ps(ptr_c1 + j, ymm_c1);
						_mm256_storeu_ps(ptr_c2 + j, ymm_c2);
						_mm256_storeu_ps(ptr_c3 + j, ymm_c3);
						_mm256_storeu_ps(ptr_c4 + j, ymm_c4);
						_mm256_storeu_ps(ptr_c5 + j, ymm_c5);
						_mm256_storeu_ps(ptr_c6 + j, ymm_c6);
						_mm256_storeu_ps(ptr_c7 + j, ymm_c7);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set_ps(ptr_a7[k], ptr_a6[k], ptr_a5[k], ptr_a4[k], ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_ps(ptr_b[j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c1[j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c2[j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c3[j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c4[j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c5[j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c6[j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c7[j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
					ptr_b += rsb;
				}
			}
		}
	};

	template<>
	struct columns_gemm_float<cpu_avx | cpu_fma>
	{
		// C(8xn) += A(8xp) * B(8xn)
		void operator()(size_t p, size_t aligned_n, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			float *ptr_c4 = ptr_c3 + rsc;
			float *ptr_c5 = ptr_c4 + rsc;
			float *ptr_c6 = ptr_c5 + rsc;
			float *ptr_c7 = ptr_c6 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			const float *ptr_a4 = ptr_a3 + rsa;
			const float *ptr_a5 = ptr_a4 + rsa;
			const float *ptr_a6 = ptr_a5 + rsa;
			const float *ptr_a7 = ptr_a6 + rsa;
			const float *ptr_b;
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;
			__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

			if (aligned_n > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set1_ps(ptr_a0[k]);
					ymm_a1 = _mm256_set1_ps(ptr_a1[k]);
					ymm_a2 = _mm256_set1_ps(ptr_a2[k]);
					ymm_a3 = _mm256_set1_ps(ptr_a3[k]);
					ymm_a4 = _mm256_set1_ps(ptr_a4[k]);
					ymm_a5 = _mm256_set1_ps(ptr_a5[k]);
					ymm_a6 = _mm256_set1_ps(ptr_a6[k]);
					ymm_a7 = _mm256_set1_ps(ptr_a7[k]);
					for (size_t j = 0; j < aligned_n; j += 8)
					{
						// load data from memory
						ymm_b0 = _mm256_loadu_ps(ptr_b + j);
						ymm_c0 = _mm256_loadu_ps(ptr_c0 + j);
						ymm_c1 = _mm256_loadu_ps(ptr_c1 + j);
						ymm_c2 = _mm256_loadu_ps(ptr_c2 + j);
						ymm_c3 = _mm256_loadu_ps(ptr_c3 + j);
						ymm_c4 = _mm256_loadu_ps(ptr_c4 + j);
						ymm_c5 = _mm256_loadu_ps(ptr_c5 + j);
						ymm_c6 = _mm256_loadu_ps(ptr_c6 + j);
						ymm_c7 = _mm256_loadu_ps(ptr_c7 + j);
						// return the weighted sum
						ymm_c0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_c0);
						ymm_c1 = _mm256_fmadd_ps(ymm_a1, ymm_b0, ymm_c1);
						ymm_c2 = _mm256_fmadd_ps(ymm_a2, ymm_b0, ymm_c2);
						ymm_c3 = _mm256_fmadd_ps(ymm_a3, ymm_b0, ymm_c3);
						ymm_c4 = _mm256_fmadd_ps(ymm_a4, ymm_b0, ymm_c4);
						ymm_c5 = _mm256_fmadd_ps(ymm_a5, ymm_b0, ymm_c5);
						ymm_c6 = _mm256_fmadd_ps(ymm_a6, ymm_b0, ymm_c6);
						ymm_c7 = _mm256_fmadd_ps(ymm_a7, ymm_b0, ymm_c7);
						// store data into memory
						_mm256_storeu_ps(ptr_c0 + j, ymm_c0);
						_mm256_storeu_ps(ptr_c1 + j, ymm_c1);
						_mm256_storeu_ps(ptr_c2 + j, ymm_c2);
						_mm256_storeu_ps(ptr_c3 + j, ymm_c3);
						_mm256_storeu_ps(ptr_c4 + j, ymm_c4);
						_mm256_storeu_ps(ptr_c5 + j, ymm_c5);
						_mm256_storeu_ps(ptr_c6 + j, ymm_c6);
						_mm256_storeu_ps(ptr_c7 + j, ymm_c7);
					}
					ptr_b += rsb;
				}
			}
			if (aligned_n < n)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					ymm_a0 = _mm256_set_ps(ptr_a7[k], ptr_a6[k], ptr_a5[k], ptr_a4[k], ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
					for (size_t j = aligned_n; j < n; ++j)
					{
						// load data from memory
						ymm_b0 = _mm256_set1_ps(ptr_b[j]);
						// return the product
						ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
						// store data into memory
						ptr_c0[j] += reinterpret_cast<float*>(&ymm_c0)[0];
						ptr_c1[j] += reinterpret_cast<float*>(&ymm_c0)[1];
						ptr_c2[j] += reinterpret_cast<float*>(&ymm_c0)[2];
						ptr_c3[j] += reinterpret_cast<float*>(&ymm_c0)[3];
						ptr_c4[j] += reinterpret_cast<float*>(&ymm_c0)[4];
						ptr_c5[j] += reinterpret_cast<float*>(&ymm_c0)[5];
						ptr_c6[j] += reinterpret_cast<float*>(&ymm_c0)[6];
						ptr_c7[j] += reinterpret_cast<float*>(&ymm_c0)[7];
					}
					ptr_b += rsb;
				}
			}
		}
	};

} // namespace core

#endif
