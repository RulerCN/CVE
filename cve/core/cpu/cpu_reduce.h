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

#ifndef __CORE_CPU_REDUCE_H__
#define __CORE_CPU_REDUCE_H__

#include "reduce/cpu_reduce_col_min.h"
#include "reduce/cpu_reduce_col_max.h"
#include "reduce/cpu_reduce_col_sum.h"
#include "reduce/cpu_reduce_row_min.h"
#include "reduce/cpu_reduce_row_max.h"
#include "reduce/cpu_reduce_row_sum.h"
#include "arithmetic/cpu_arithmetic_mul.h"

namespace core
{
	// Reducing the dimension of a matrix by a specified operation
	// Parameters:
	// 1. b - output vector.
	// 2. a - input matrix.
	// 3. mode - reduction operation that could be one of the following:
	//     reduce_row_min: return the minimum of each row of matrix.
	//     reduce_row_max: return the maximum of each row of matrix.
	//     reduce_row_avg: return the mean of each row of matrix.
	//     reduce_col_min: return the minimum of each column of matrix.
	//     reduce_col_max: return the maximum of each column of matrix.
	//     reduce_col_sum: return the sum of each column of matrix.
	//     reduce_col_avg: return the mean of each column of matrix.

	template <class A1, class A2>
	vector<signed char, A1>& cpu_reduce(vector<signed char, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type mode)
	{
		switch (mode)
		{
		case reduce_col_min:
			b.fill(int8_max);
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			b.fill(int8_min);
			return cpu_reduce_col_max(b, a);
		case reduce_row_min:
			b.fill(int8_max);
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			b.fill(int8_min);
			return cpu_reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce(vector<signed int, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			b.fill(int32_zero);
			return cpu_reduce_col_sum(b, a);
		case reduce_row_sum:
			b.fill(int32_zero);
			return cpu_reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			b.fill(0.0F);
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			b.fill(0.0F);
			cpu_reduce_col_sum(b, a);
			return cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
		case reduce_row_sum:
			b.fill(0.0F);
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			b.fill(0.0F);
			cpu_reduce_row_sum(b, a);
			return cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_reduce(vector<unsigned char, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			b.fill(uint8_max);
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			b.fill(uint8_min);
			return cpu_reduce_col_max(b, a);
		case reduce_row_min:
			b.fill(uint8_max);
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			b.fill(uint8_min);
			return cpu_reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce(vector<signed int, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			b.fill(int32_zero);
			return cpu_reduce_col_sum(b, a);
		case reduce_row_sum:
			b.fill(int32_zero);
			return cpu_reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			b.fill(0.0F);
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_reduce(vector<signed short, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce(vector<signed int, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_reduce(vector<unsigned short, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce(vector<signed int, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_reduce(vector<signed int, A1> &b, const matrix<signed int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<signed int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_reduce(vector<unsigned int, A1> &b, const matrix<unsigned int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<unsigned int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_reduce(vector<float, A1> &b, const matrix<float, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0F / static_cast<float>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce(vector<double, A1> &b, const matrix<double, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case reduce_col_min:
			return cpu_reduce_col_min(b, a);
		case reduce_col_max:
			return cpu_reduce_col_max(b, a);
		case reduce_col_sum:
			return cpu_reduce_col_sum(b, a);
		case reduce_col_avg:
			cpu_reduce_col_sum(b, a);
			cpu_mul(b, 1.0 / static_cast<double>(a.rows()), b);
			return b;
		case reduce_row_min:
			return cpu_reduce_row_min(b, a);
		case reduce_row_max:
			return cpu_reduce_row_max(b, a);
		case reduce_row_sum:
			return cpu_reduce_row_sum(b, a);
		case reduce_row_avg:
			cpu_reduce_row_sum(b, a);
			cpu_mul(b, 1.0 / static_cast<double>(a.rows()), b);
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

} // namespace core

#endif
