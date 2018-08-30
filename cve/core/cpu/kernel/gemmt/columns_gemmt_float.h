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

#ifndef __CORE_CPU_KERNEL_COLUMNS_GEMMT_FLOAT_H__
#define __CORE_CPU_KERNEL_COLUMNS_GEMMT_FLOAT_H__

#include "../../cpu_inst.h"

namespace core
{
	// Class template columns_gemmt_float
	
	template<cpu_inst_type inst>
	struct columns_gemmt_float
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			float val_b0;
			float val_t0, val_t1, val_t2, val_t3;

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
	struct columns_gemmt_float<cpu_sse3>
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = xmm_t2 = xmm_t3 = _mm_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 4)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
						xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
						xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
						xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
						xmm_b0 = _mm_loadu_ps(b + k);
						// return the weighted sum
						xmm_t0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_t0);
						xmm_t1 = _mm_add_ps(_mm_mul_ps(xmm_a1, xmm_b0), xmm_t1);
						xmm_t2 = _mm_add_ps(_mm_mul_ps(xmm_a2, xmm_b0), xmm_t2);
						xmm_t3 = _mm_add_ps(_mm_mul_ps(xmm_a3, xmm_b0), xmm_t3);
					}
					// return the horizontal sum
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_ps(xmm_t2, xmm_t3);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t2);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						xmm_a0 = _mm_set_ps(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						xmm_b0 = _mm_set1_ps(b[k]);
						// return the weighted sum
						xmm_t0 = _mm_add_ps(_mm_mul_ps(xmm_a0, xmm_b0), xmm_t0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				ptr_c1[j] += reinterpret_cast<float*>(&xmm_t0)[1];
				ptr_c2[j] += reinterpret_cast<float*>(&xmm_t0)[2];
				ptr_c3[j] += reinterpret_cast<float*>(&xmm_t0)[3];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_float<cpu_sse3 | cpu_fma>
	{
		// C(4xn) += A(4xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			float *ptr_c0 = c;
			float *ptr_c1 = c + rsc;
			float *ptr_c2 = ptr_c1 + rsc;
			float *ptr_c3 = ptr_c2 + rsc;
			const float *ptr_a0 = a;
			const float *ptr_a1 = a + rsa;
			const float *ptr_a2 = ptr_a1 + rsa;
			const float *ptr_a3 = ptr_a2 + rsa;
			__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
			__m128 xmm_b0;
			__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					xmm_t0 = xmm_t1 = xmm_t2 = xmm_t3 = _mm_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 4)
					{
						// load data from memory
						xmm_a0 = _mm_loadu_ps(ptr_a0 + k);
						xmm_a1 = _mm_loadu_ps(ptr_a1 + k);
						xmm_a2 = _mm_loadu_ps(ptr_a2 + k);
						xmm_a3 = _mm_loadu_ps(ptr_a3 + k);
						xmm_b0 = _mm_loadu_ps(b + k);
						// return the weighted sum
						xmm_t0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_t0);
						xmm_t1 = _mm_fmadd_ps(xmm_a1, xmm_b0, xmm_t1);
						xmm_t2 = _mm_fmadd_ps(xmm_a2, xmm_b0, xmm_t2);
						xmm_t3 = _mm_fmadd_ps(xmm_a3, xmm_b0, xmm_t3);
					}
					// return the horizontal sum
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t1);
					xmm_t2 = _mm_hadd_ps(xmm_t2, xmm_t3);
					xmm_t0 = _mm_hadd_ps(xmm_t0, xmm_t2);
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						xmm_a0 = _mm_set_ps(ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						xmm_b0 = _mm_set1_ps(b[k]);
						// return the weighted sum
						xmm_t0 = _mm_fmadd_ps(xmm_a0, xmm_b0, xmm_t0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<float*>(&xmm_t0)[0];
				ptr_c1[j] += reinterpret_cast<float*>(&xmm_t0)[1];
				ptr_c2[j] += reinterpret_cast<float*>(&xmm_t0)[2];
				ptr_c3[j] += reinterpret_cast<float*>(&xmm_t0)[3];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_float<cpu_avx>
	{
		// C(8xn) += A(8xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
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
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = ymm_t4 = ymm_t5 = ymm_t6 = ymm_t7 = _mm256_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
						ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
						ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
						ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
						ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
						ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
						ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
						ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
						ymm_b0 = _mm256_loadu_ps(b + k);
						// return the weighted sum
						ymm_t0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_t0);
						ymm_t1 = _mm256_add_ps(_mm256_mul_ps(ymm_a1, ymm_b0), ymm_t1);
						ymm_t2 = _mm256_add_ps(_mm256_mul_ps(ymm_a2, ymm_b0), ymm_t2);
						ymm_t3 = _mm256_add_ps(_mm256_mul_ps(ymm_a3, ymm_b0), ymm_t3);
						ymm_t4 = _mm256_add_ps(_mm256_mul_ps(ymm_a4, ymm_b0), ymm_t4);
						ymm_t5 = _mm256_add_ps(_mm256_mul_ps(ymm_a5, ymm_b0), ymm_t5);
						ymm_t6 = _mm256_add_ps(_mm256_mul_ps(ymm_a6, ymm_b0), ymm_t6);
						ymm_t7 = _mm256_add_ps(_mm256_mul_ps(ymm_a7, ymm_b0), ymm_t7);
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
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						ymm_a0 = _mm256_set_ps(ptr_a7[k], ptr_a6[k], ptr_a5[k], ptr_a4[k], ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						ymm_b0 = _mm256_set1_ps(b[k]);
						// return the weighted sum
						ymm_t0 = _mm256_add_ps(_mm256_mul_ps(ymm_a0, ymm_b0), ymm_t0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<float*>(&ymm_t0)[0];
				ptr_c1[j] += reinterpret_cast<float*>(&ymm_t0)[1];
				ptr_c2[j] += reinterpret_cast<float*>(&ymm_t0)[2];
				ptr_c3[j] += reinterpret_cast<float*>(&ymm_t0)[3];
				ptr_c4[j] += reinterpret_cast<float*>(&ymm_t0)[4];
				ptr_c5[j] += reinterpret_cast<float*>(&ymm_t0)[5];
				ptr_c6[j] += reinterpret_cast<float*>(&ymm_t0)[6];
				ptr_c7[j] += reinterpret_cast<float*>(&ymm_t0)[7];
				b += rsb;
			}
		}
	};

	template<>
	struct columns_gemmt_float<cpu_avx | cpu_fma>
	{
		// C(8xn) += A(8xp) * B(nxp)^T
		void operator()(size_t n, size_t aligned_p, size_t p, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
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
			__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
			__m256 ymm_b0;
			__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

			for (size_t j = 0; j < n; ++j)
			{
				if (aligned_p > 0)
				{
					ymm_t0 = ymm_t1 = ymm_t2 = ymm_t3 = ymm_t4 = ymm_t5 = ymm_t6 = ymm_t7 = _mm256_setzero_ps();
					for (size_t k = 0; k < aligned_p; k += 8)
					{
						ymm_a0 = _mm256_loadu_ps(ptr_a0 + k);
						ymm_a1 = _mm256_loadu_ps(ptr_a1 + k);
						ymm_a2 = _mm256_loadu_ps(ptr_a2 + k);
						ymm_a3 = _mm256_loadu_ps(ptr_a3 + k);
						ymm_a4 = _mm256_loadu_ps(ptr_a4 + k);
						ymm_a5 = _mm256_loadu_ps(ptr_a5 + k);
						ymm_a6 = _mm256_loadu_ps(ptr_a6 + k);
						ymm_a7 = _mm256_loadu_ps(ptr_a7 + k);
						ymm_b0 = _mm256_loadu_ps(b + k);
						// return the weighted sum
						ymm_t0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_t0);
						ymm_t1 = _mm256_fmadd_ps(ymm_a1, ymm_b0, ymm_t1);
						ymm_t2 = _mm256_fmadd_ps(ymm_a2, ymm_b0, ymm_t2);
						ymm_t3 = _mm256_fmadd_ps(ymm_a3, ymm_b0, ymm_t3);
						ymm_t4 = _mm256_fmadd_ps(ymm_a4, ymm_b0, ymm_t4);
						ymm_t5 = _mm256_fmadd_ps(ymm_a5, ymm_b0, ymm_t5);
						ymm_t6 = _mm256_fmadd_ps(ymm_a6, ymm_b0, ymm_t6);
						ymm_t7 = _mm256_fmadd_ps(ymm_a7, ymm_b0, ymm_t7);
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
				}
				if (aligned_p < p)
				{
					for (size_t k = aligned_p; k < p; ++k)
					{
						// load data from memory
						ymm_a0 = _mm256_set_ps(ptr_a7[k], ptr_a6[k], ptr_a5[k], ptr_a4[k], ptr_a3[k], ptr_a2[k], ptr_a1[k], ptr_a0[k]);
						ymm_b0 = _mm256_set1_ps(b[k]);
						// return the weighted sum
						ymm_t0 = _mm256_fmadd_ps(ymm_a0, ymm_b0, ymm_t0);
					}
				}
				// store data into memory
				ptr_c0[j] += reinterpret_cast<float*>(&ymm_t0)[0];
				ptr_c1[j] += reinterpret_cast<float*>(&ymm_t0)[1];
				ptr_c2[j] += reinterpret_cast<float*>(&ymm_t0)[2];
				ptr_c3[j] += reinterpret_cast<float*>(&ymm_t0)[3];
				ptr_c4[j] += reinterpret_cast<float*>(&ymm_t0)[4];
				ptr_c5[j] += reinterpret_cast<float*>(&ymm_t0)[5];
				ptr_c6[j] += reinterpret_cast<float*>(&ymm_t0)[6];
				ptr_c7[j] += reinterpret_cast<float*>(&ymm_t0)[7];
				b += rsb;
			}
		}
	};

} // namespace core

#endif
