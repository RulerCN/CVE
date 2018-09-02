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

#ifndef __CORE_CPU_KERNEL_GEVM_FLOAT_H__
#define __CORE_CPU_KERNEL_GEVM_FLOAT_H__

#include "block_gevm_float.h"
#include "rows_gevm_float.h"

namespace core
{
	// Class template kernel_gevm_float
	template<size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_gevm_float
	{
		// C(1xn) += A(1xp) * B(pxn)
		void operator()(size_t p, size_t n, const float *a, const float *b, size_t rsb, float *c) const
		{
			const size_t block_rsb = block_p * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const struct block_gevm_float<inst> block_functor;
			const struct rows_gevm_float<inst> rows_functor;

			for (size_t k = 0; k < aligned_p; k += block_p)
			{
				block_functor(aligned_n, n, a + k, b, rsb, c);
				b += block_rsb;
			}
			if (surplus_p > 0)
				rows_functor(surplus_p, aligned_n, n, a + aligned_p, b, rsb, c);
		}
	};

	// Class template kernel_gemt_float
	template<size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_gemt_float
	{
		// C(lxn) += A(lxp) * B(lxpxn)
		void operator()(size_t l, size_t p, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc) const
		{
			const size_t msb = p * rsb;
			const size_t block_rsb = block_p * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const struct block_gevm_float<inst> block_functor;
			const struct rows_gevm_float<inst> rows_functor;

			for (size_t r = 0; r < l; ++r)
			{
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					block_functor(aligned_n, n, a + k, b, rsb, c);
					b += block_rsb;
				}
				if (surplus_p > 0)
					rows_functor(surplus_p, aligned_n, n, a + aligned_p, b, rsb, c);
				a += rsa;
				b += msb;
				c += rsc;
			}
		}
	};

} // namespace core

#endif
