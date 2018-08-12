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

#ifndef __CORE_CPU_KERNEL_REDUCE_SUM_DOUBLE_H__
#define __CORE_CPU_KERNEL_REDUCE_SUM_DOUBLE_H__

#include "common_reduce_sum_double.h"
#include "rows_reduce_sum_double.h"
#include "block_reduce_sum_double.h"

namespace core
{
	// Function template kernel_reduce_sum_double

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sum_double(size_t m, size_t n, const T *a, size_t rsa, double *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const struct common_reduce_sum_int32<T> functor;
		const struct rows_reduce_sum_int32<T, inst> rows_functor;
		const struct block_reduce_sum_int32<T, inst> block_functor;

		for (size_t i = 0; i < aligned_m; i += block_m)
		{
			if (aligned_n > 0)
				block_functor(aligned_n, a, rsa, b);
			if (surplus_n > 0)
				functor(block_m, surplus_n, a + aligned_n, rsa, b);
			a += block_rsa;
			b += block_m;
		}
		if (surplus_m > 0)
		{
			if (aligned_n > 0)
				rows_functor(surplus_m, aligned_n, a, rsa, b);
			if (surplus_n > 0)
				functor(surplus_m, surplus_n, a + aligned_n, rsa, b);
		}
	}

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sum_double(size_t l, size_t m, size_t n, const T *a, size_t rsa, double *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const size_t surplus_rsa = surplus_m * rsa;
		const struct common_reduce_sum_int32<T> functor;
		const struct rows_reduce_sum_int32<T, inst> rows_functor;
		const struct block_reduce_sum_int32<T, inst> block_functor;

		for (size_t j = 0; j < l; j++)
		{
			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_n > 0)
					block_functor(aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(block_m, surplus_n, a + aligned_n, rsa, b);
				a += block_rsa;
				b += block_m;
			}
			if (surplus_m > 0)
			{
				if (aligned_n > 0)
					rows_functor(surplus_m, aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(surplus_m, surplus_n, a + aligned_n, rsa, b);
				a += surplus_rsa;
				b += surplus_m;
			}
		}
	}

	// Function template kernel_reduce_sumt_double

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sumt_double(size_t m, size_t n, const T *a, size_t rsa, double *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const struct common_reduce_sumt_int32<T> functor;
		const struct rows_reduce_sumt_int32<T, inst> rows_functor;
		const struct block_reduce_sumt_int32<T, inst> block_functor;

		for (size_t i = 0; i < aligned_m; i += block_m)
		{
			if (aligned_n > 0)
				block_functor(aligned_n, a, rsa, b);
			if (surplus_n > 0)
				functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_n);
			a += block_rsa;
		}
		if (surplus_m > 0)
		{
			if (aligned_n > 0)
				rows_functor(surplus_m, aligned_n, a, rsa, b);
			if (surplus_n > 0)
				functor(surplus_m, surplus_n, a + aligned_n, rsa, b + aligned_n);
		}
	}

	template<class T, size_t block_m, size_t block_n, cpu_inst_type inst>
	void kernel_reduce_sumt_double(size_t l, size_t m, size_t n, const T *a, size_t rsa, double *b)
	{
		const size_t block_rsa = block_m * rsa;
		const size_t aligned_m = m & ~(block_m - 1);
		const size_t aligned_n = n & ~(block_n - 1);
		const size_t surplus_m = m - aligned_m;
		const size_t surplus_n = n - aligned_n;
		const size_t surplus_rsa = surplus_m * rsa;
		const struct common_reduce_sumt_int32<T> functor;
		const struct rows_reduce_sumt_int32<T, inst> rows_functor;
		const struct block_reduce_sumt_int32<T, inst> block_functor;

		for (size_t j = 0; j < l; j++)
		{
			for (size_t i = 0; i < aligned_m; i += block_m)
			{
				if (aligned_n > 0)
					block_functor(aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(block_m, surplus_n, a + aligned_n, rsa, b + aligned_n);
				a += block_rsa;
			}
			if (surplus_m > 0)
			{
				if (aligned_n > 0)
					rows_functor(surplus_m, aligned_n, a, rsa, b);
				if (surplus_n > 0)
					functor(surplus_m, surplus_n, a + aligned_n, rsa, b + aligned_n);
				a += surplus_rsa;
			}
		}
	}

} // namespace core

#endif
