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

#ifndef __CORE_CPU_KERNEL_ARITHMETIC_EXP_H__
#define __CORE_CPU_KERNEL_ARITHMETIC_EXP_H__

#include "../../math/cpu_exp.h"

namespace core
{
	// Class template kernel_exp

	template<class T, cpu_inst_type inst>
	struct kernel_exp
	{
		void operator()(size_t n, const T *a, T *b) const
		{
			for (size_t i = 0; i < n; ++i)
				b[i] = exp<inst>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse2>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128 xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expf4<cpu_sse2>(xmm_a);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse2 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128 xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expf4<cpu_sse2 | cpu_fma>(xmm_a);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse41>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128 xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expf4<cpu_sse41>(xmm_a);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_sse41 | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m128 xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_ps(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expf4<cpu_sse41 | cpu_fma>(xmm_a);
				// store data into memory
				_mm_storeu_ps(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse2>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128d xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expd2<cpu_sse2>(xmm_a);
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse2 | cpu_fma>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128d xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expd2<cpu_sse2 | cpu_fma>(xmm_a);
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse41>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128d xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expd2<cpu_sse41>(xmm_a);
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_sse41 | cpu_fma>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 2;
			const size_t aligned = n & ~(block - 1);
			__m128d xmm_a, xmm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				xmm_a = _mm_loadu_pd(a + i);
				// xmm_b = exp(xmm_a);
				xmm_b = expd2<cpu_sse41 | cpu_fma>(xmm_a);
				// store data into memory
				_mm_storeu_pd(b + i, xmm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_avx>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256 ymm_a, ymm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// ymm_b = exp(ymm_a);
				ymm_b = expf8<cpu_avx>(ymm_a);
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<float, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, const float *a, float *b) const
		{
			constexpr size_t block = 8;
			const size_t aligned = n & ~(block - 1);
			__m256 ymm_a, ymm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_ps(a + i);
				// ymm_b = exp(ymm_a);
				ymm_b = expf8<cpu_avx | cpu_fma>(ymm_a);
				// store data into memory
				_mm256_storeu_ps(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_avx>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m256d ymm_a, ymm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_pd(a + i);
				// ymm_b = exp(ymm_a);
				ymm_b = expd4<cpu_avx>(ymm_a);
				// store data into memory
				_mm256_storeu_pd(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

	template<>
	struct kernel_exp<double, cpu_avx | cpu_fma>
	{
		void operator()(size_t n, const double *a, double *b) const
		{
			constexpr size_t block = 4;
			const size_t aligned = n & ~(block - 1);
			__m256d ymm_a, ymm_b;

			for (size_t i = 0; i < aligned; i += block)
			{
				// load data from memory
				ymm_a = _mm256_loadu_pd(a + i);
				// ymm_b = exp(ymm_a);
				ymm_b = expd4<cpu_avx | cpu_fma>(ymm_a);
				// store data into memory
				_mm256_storeu_pd(b + i, ymm_b);
			}
			for (size_t i = aligned; i < n; ++i)
				b[i] = exp<cpu_none>(a[i]);
		}
	};

} // namespace core

#endif
