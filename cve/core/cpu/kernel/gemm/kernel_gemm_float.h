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

#ifndef __CORE_CPU_KERNEL_GEMM_FLOAT_H__
#define __CORE_CPU_KERNEL_GEMM_FLOAT_H__

#include "block_gemm_float.h"
#include "columns_gemm_float.h"
#include "rows_gemm_float.h"
#include "rect_gemm_float.h"

namespace core
{
	// Class template kernel_gemm_float
	template<size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_gemm_float
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b;
			const size_t block_rsa = block_m * rsa;
			const size_t block_rsb = block_p * rsb;
			const size_t block_rsc = block_m * rsc;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const struct block_gemm_float<inst> block_functor;
			const struct columns_gemm_float<inst> columns_functor;
			const struct rows_gemm_float<inst> rows_functor;
			const struct rect_gemm_float<inst> rect_functor;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					block_functor(aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					columns_functor(surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
				a += block_rsa;
				c += block_rsc;
			}
			if (surplus_m > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					rows_functor(surplus_m, aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					rect_functor(surplus_m, surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
			}
		}
	};

	// Class template kernel_getm_float
	template<size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_getm_float
	{
		// C(lxmxn) += A(lxmxp) * B(pxn)
		void operator()(size_t l, size_t m, size_t p, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b;
			const size_t block_rsa = block_m * rsa;
			const size_t block_rsb = block_p * rsb;
			const size_t block_rsc = block_m * rsc;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_rsa = surplus_m * rsa;
			const size_t surplus_rsc = surplus_m * rsc;
			const struct block_gemm_float<inst> block_functor;
			const struct columns_gemm_float<inst> columns_functor;
			const struct rows_gemm_float<inst> rows_functor;
			const struct rect_gemm_float<inst> rect_functor;

			for (size_t r = 0; r < l; ++r)
			{
				for (size_t i = 0; i < aligned_m; i += block_m)
				{
					ptr_b = b;
					for (size_t k = 0; k < aligned_p; k += block_p)
					{
						block_functor(aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
						ptr_b += block_rsb;
					}
					if (surplus_p > 0)
						columns_functor(surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
					a += block_rsa;
					c += block_rsc;
				}
				if (surplus_m > 0)
				{
					ptr_b = b;
					for (size_t k = 0; k < aligned_p; k += block_p)
					{
						rows_functor(surplus_m, aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
						ptr_b += block_rsb;
					}
					if (surplus_p > 0)
						rect_functor(surplus_m, surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
					a += surplus_rsa;
					c += surplus_rsc;
				}
			}
		}
	};

	// Class template kernel_gett_float
	template<size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_gett_float
	{
		// C(lxmxn) += A(lxmxp) * B(lxpxn)
		void operator()(size_t l, size_t m, size_t p, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const float *ptr_b;
			const size_t msb = n * rsb;
			const size_t block_rsa = block_m * rsa;
			const size_t block_rsb = block_p * rsb;
			const size_t block_rsc = block_m * rsc;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_rsa = surplus_m * rsa;
			const size_t surplus_rsc = surplus_m * rsc;
			const struct block_gemm_float<inst> block_functor;
			const struct columns_gemm_float<inst> columns_functor;
			const struct rows_gemm_float<inst> rows_functor;
			const struct rect_gemm_float<inst> rect_functor;

			for (size_t r = 0; r < l; ++r)
			{
				for (size_t i = 0; i < aligned_m; i += block_m)
				{
					ptr_b = b;
					for (size_t k = 0; k < aligned_p; k += block_p)
					{
						block_functor(aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
						ptr_b += block_rsb;
					}
					if (surplus_p > 0)
						columns_functor(surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
					a += block_rsa;
					c += block_rsc;
				}
				if (surplus_m > 0)
				{
					ptr_b = b;
					for (size_t k = 0; k < aligned_p; k += block_p)
					{
						rows_functor(surplus_m, aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
						ptr_b += block_rsb;
					}
					if (surplus_p > 0)
						rect_functor(surplus_m, surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
					a += surplus_rsa;
					c += surplus_rsc;
				}
				b += msb;
			}
		}
	};

} // namespace core

#endif