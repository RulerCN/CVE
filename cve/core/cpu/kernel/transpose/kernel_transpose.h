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

#ifndef __CORE_CPU_KERNEL_TRANSPOSE_H__
#define __CORE_CPU_KERNEL_TRANSPOSE_H__

#include "block_transpose.h"
#include "common_transpose.h"

namespace core
{
	// Function template kernel_transpose

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_transpose(size_t m, size_t n, const T *a, size_t rsa, T *b, size_t rsb)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t aligned_rsb = aligned_n * rsb;
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const struct common_transpose<T> functor;
		const struct block_transpose<T, inst> special_functor;

		for (size_t i = 0; i < aligned_m; i += block_m)
		{
			if (aligned_n > 0)
				special_functor(aligned_n, a, rsa, b, rsb);
			if (surplus_n > 0)
				functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_rsb, rsb);
			a += block_rsa;
			b += block_m;
		}
		if (surplus_m > 0)
			functor(surplus_m, n, a, rsa, b, rsb);
	}

	//template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	//void kernel_transpose(size_t l, size_t m, size_t n, const T *a, size_t rsa, T *b, size_t rsb)
	//{
	//	const size_t block_rsa = block_m * rsa;
	//	const size_t aligned_m = m & ~(block_m - 1);
	//	const size_t aligned_n = n & ~(block_n - 1);
	//	const size_t aligned_rsb = aligned_n * rsb;
	//	const size_t surplus_m = m - aligned_m;
	//	const size_t surplus_n = n - aligned_n;
	//	const struct common_transpose<T> functor;
	//	const struct block_transpose<T, inst> special_functor;

	//	for (size_t i = 0; i < aligned_m; i += block_m)
	//	{
	//		if (aligned_n > 0)
	//			special_functor(aligned_n, a, rsa, b, rsb);
	//		if (surplus_n > 0)
	//			functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_rsb, rsb);
	//		a += block_rsa;
	//		b += block_m;
	//	}
	//	if (surplus_m > 0)
	//		functor(surplus_m, n, a, rsa, b, rsb);
	//}

} // namespace core

#endif
