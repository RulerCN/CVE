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

#ifndef __CORE_CPU_KERNEL_MATMUL_RMRM_H__
#define __CORE_CPU_KERNEL_MATMUL_RMRM_H__

#include "kernel_matmul_rmrm00.h"
#include "kernel_matmul_rmrm01.h"
#include "kernel_matmul_rmrm10.h"
#include "kernel_matmul_rmrm11.h"

namespace core
{
	// Class template common_matmul_rmrm
	template<class T>
	struct common_matmul_rmrm
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			T val_a;

			for (size_t i = 0; i < m; ++i)
			{
				ptr_b = b;
				for (size_t k = 0; k < p; ++k)
				{
					val_a = a[k];
					for (size_t j = 0; j < n; ++j)
						c[j] += val_a * ptr_b[j];
					ptr_b += rsb;
				}
				a += rsa;
				c += rsc;
			}
		}
	};

	// Class template kernel_matmul_rmrm
	template<class T, size_t block_m, size_t block_p, size_t block_n, cpu_inst_type inst>
	struct kernel_matmul_rmrm
	{
		// C(mxn) += A(mxp) * B(pxn)
		void operator()(size_t m, size_t p, size_t n, const T *a, size_t rsa, const T *b, size_t rsb, T *c, size_t rsc) const
		{
			const T *ptr_b;
			const size_t block_rsa = block_m * rsa;
			const size_t block_rsb = block_p * rsb;
			const size_t block_rsc = block_m * rsc;
			const size_t aligned_m = m & ~(block_m - 1);
			const size_t aligned_p = p & ~(block_p - 1);
			const size_t aligned_n = n & ~(block_n - 1);
			const size_t surplus_m = m - aligned_m;
			const size_t surplus_p = p - aligned_p;
			const struct block_matmul_rmrm00<T, inst> functor00;
			const struct block_matmul_rmrm01<T, inst> functor01;
			const struct block_matmul_rmrm10<T, inst> functor10;
			const struct block_matmul_rmrm11<T, inst> functor11;

			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					functor00(aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					functor01(surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
				a += block_rsa;
				c += block_rsc;
			}
			if (surplus_m > 0)
			{
				ptr_b = b;
				for (size_t k = 0; k < aligned_p; k += block_p)
				{
					functor10(surplus_m, aligned_n, n, a + k, rsa, ptr_b, rsb, c, rsc);
					ptr_b += block_rsb;
				}
				if (surplus_p > 0)
					functor11(surplus_m, surplus_p, aligned_n, n, a + aligned_p, rsa, ptr_b, rsb, c, rsc);
			}
		}
	};

} // namespace core

#endif
