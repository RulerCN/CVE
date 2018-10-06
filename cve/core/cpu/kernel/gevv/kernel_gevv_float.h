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

#ifndef __CORE_CPU_KERNEL_GEVV_FLOAT_H__
#define __CORE_CPU_KERNEL_GEVV_FLOAT_H__

#include "rows_gevv_float.h"

namespace core
{
	// Function template kernel_gevv_float

	template<size_t block_n, cpu_inst_type inst>
	void kernel_gevv_float(size_t m, size_t n, const float *a, const float *b, float *c, size_t rsc)
	{
		const size_t aligned_n = n & ~(block_n - 1);
		const struct rows_gevv_float<inst> functor;

		functor(m, aligned_n, n, a, b, c, rsc);
	}

	template<size_t block_n, cpu_inst_type inst>
	void kernel_gevv_float(size_t l, size_t m, size_t n, const float *a, size_t rsa, const float *b, size_t rsb, float *c, size_t rsc)
	{
		const size_t block_rsc = m * rsc;
		const size_t aligned_n = n & ~(block_n - 1);
		const struct rows_gevv_float<inst> functor;

		for (size_t r = 0; r < l; ++r)
		{
			functor(m, aligned_n, n, a, b, c, rsc);
			a += rsa;
			b += rsb;
			c += block_rsc;
		}
	}

} // namespace core

#endif
