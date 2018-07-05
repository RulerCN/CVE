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

#ifndef __CORE_CPU_KERNEL_MATMUL_RMCM_H__
#define __CORE_CPU_KERNEL_MATMUL_RMCM_H__

#include "kernel_matmul_rvcm.h"

namespace core
{
	// Class template kernel_matmul_rmcm
	template<class T, size_t block_n, size_t block_p, cpu_inst_type inst>
	struct kernel_matmul_rmcm
	{
		// C(mxn) += A(mxp) * B(nxp)^T
		void operator()(size_t m, size_t n, size_t p, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			const size_t block_rsb = block_n * rsb;
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t surplus_n = n - aligned_n;
			const size_t surplus_p = p - aligned_p;
			const struct common_matmul_rvcm<T> functor;
			const struct block_matmul_rvcm<T, inst> special_functor;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t j = 0; j < aligned_n; j += block_n)
				{
					if (aligned_p > 0)
						special_functor(aligned_p, a, ptr_b, rsb, c + j);
					if (surplus_p > 0)
						functor(block_n, surplus_p, a + aligned_p, ptr_b + aligned_p, rsb, c + j);
					ptr_b += block_rsb;
				}
				if (surplus_n > 0)
					functor(surplus_n, p, a, b, rsb, c + aligned_n);
				a += rsa;
				c += rsc;
			}
		}
	};

} // namespace core

#endif
