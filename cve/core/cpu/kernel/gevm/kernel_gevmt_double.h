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

#ifndef __CORE_CPU_KERNEL_GEVMT_DOUBLE_H__
#define __CORE_CPU_KERNEL_GEVMT_DOUBLE_H__

#include "block_gevmt_double.h"
#include "rows_gevmt_double.h"

namespace core
{
	// Class template kernel_gevmt_double
	template<size_t block_n, size_t block_p, cpu_inst_type inst>
	struct kernel_gevmt_double
	{
		// C(1xn) += A(1xp) * B(nxp)^T
		void operator()(size_t n, size_t p, const double *a, const double *b, size_t rsb, double *c) const
		{
			const size_t block_rsb = block_n * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const struct block_gevmt_double<inst> block_functor;
			const struct rows_gevmt_double<inst> rows_functor;

			for (size_t j = 0; j < aligned_n; j += block_n)
			{
				block_functor(aligned_p, p, a, b, rsb, c + j);
				b += block_rsb;
			}
			if (surplus_n > 0)
				rows_functor(surplus_n, aligned_p, p, a, b, rsb, c + aligned_n);
		}
	};

	// Class template kernel_gemtt_double
	template<size_t block_n, size_t block_p, cpu_inst_type inst>
	struct kernel_gemtt_double
	{
		// C(lxn) += A(lxp) * B(lxnxp)^T
		void operator()(size_t l, size_t n, size_t p, const double *a, size_t rsa, const double *b, size_t rsb, double *c, size_t rsc) const
		{
			const size_t block_rsb = block_n * rsb;
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_p = p - aligned_p;
			const size_t surplus_rsb = surplus_p * rsb;
			const struct block_gevmt_double<inst> block_functor;
			const struct rows_gevmt_double<inst> rows_functor;

			for (size_t r = 0; r < l; ++r)
			{
				for (size_t j = 0; j < aligned_n; j += block_n)
				{
					block_functor(aligned_p, p, a, b, rsb, c + j);
					b += block_rsb;
				}
				if (surplus_n > 0)
				{
					rows_functor(surplus_n, aligned_p, p, a, b, rsb, c + aligned_n);
					b += surplus_rsb;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

} // namespace core

#endif
