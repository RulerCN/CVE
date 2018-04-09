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

#include "reduce_col_min.h"
#include "reduce_col_max.h"
#include "reduce_col_sum.h"
#include "reduce_row_min.h"
#include "reduce_row_max.h"
#include "reduce_row_sum.h"
#include "convert_scale.h"

namespace core
{
	typedef unsigned char reduce_mode_type;
	static constexpr reduce_mode_type rm_col_min = 0x01;
	static constexpr reduce_mode_type rm_col_max = 0x02;
	static constexpr reduce_mode_type rm_col_sum = 0x03;
	static constexpr reduce_mode_type rm_col_avg = 0x04;
	static constexpr reduce_mode_type rm_row_min = 0x11;
	static constexpr reduce_mode_type rm_row_max = 0x12;
	static constexpr reduce_mode_type rm_row_sum = 0x13;
	static constexpr reduce_mode_type rm_row_avg = 0x14;

	// Reducing the dimension of a matrix by a specified operation
	// Parameters:
	// 1. b - output vector.
	// 2. a - input matrix.
	// 3. reduce_mode - reduction operation that could be one of the following:
	//     rm_row_min: return the minimum of each row of matrix.
	//     rm_row_max: return the maximum of each row of matrix.
	//     rm_row_sum: return the sum of each row of matrix.
	//     rm_row_avg: return the mean of each row of matrix.
	//     rm_col_min: return the minimum of each column of matrix.
	//     rm_col_max: return the maximum of each column of matrix.
	//     rm_col_sum: return the sum of each column of matrix.
	//     rm_col_avg: return the mean of each column of matrix.

	template <class A1, class A2>
	vector<signed char, A1>& reduce(vector<signed char, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			b.fill(int8_max);
			return reduce_col_min(b, a);
		case rm_col_max:
			b.fill(int8_min);
			return reduce_col_max(b, a);
		case rm_row_min:
			b.fill(int8_max);
			return reduce_row_min(b, a);
		case rm_row_max:
			b.fill(int8_min);
			return reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& reduce(vector<signed int, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			b.fill(int32_zero);
			return reduce_col_sum(b, a);
		case rm_row_sum:
			b.fill(int32_zero);
			return reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<signed char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			b.fill(0.0F);
			return reduce_col_sum(b, a);
		case rm_col_avg:
			b.fill(0.0F);
			reduce_col_sum(b, a);
			return convert_scale(b, b, 1.0F / static_cast<float>(a.rows()));
		case rm_row_sum:
			b.fill(0.0F);
			return reduce_row_sum(b, a);
		case rm_row_avg:
			b.fill(0.0F);
			reduce_row_sum(b, a);
			return convert_scale(b, b, 1.0F / static_cast<float>(a.rows()));
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& reduce(vector<unsigned char, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			b.fill(uint8_max);
			return reduce_col_min(b, a);
		case rm_col_max:
			b.fill(uint8_min);
			return reduce_col_max(b, a);
		case rm_row_min:
			b.fill(uint8_max);
			return reduce_row_min(b, a);
		case rm_row_max:
			b.fill(uint8_min);
			return reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& reduce(vector<signed int, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			b.fill(int32_zero);
			return reduce_col_sum(b, a);
		case rm_row_sum:
			b.fill(int32_zero);
			return reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<unsigned char, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			b.fill(0.0F);
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed short, A1>& reduce(vector<signed short, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& reduce(vector<signed int, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_row_sum:
			return reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<signed short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& reduce(vector<unsigned short, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& reduce(vector<signed int, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_row_sum:
			return reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<unsigned short, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<signed int, A1>& reduce(vector<signed int, A1> &b, const matrix<signed int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		case rm_row_sum:
			return reduce_row_sum(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<signed int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& reduce(vector<unsigned int, A1> &b, const matrix<unsigned int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<unsigned int, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<float, A1>& reduce(vector<float, A1> &b, const matrix<float, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0f / static_cast<float>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	template <class A1, class A2>
	vector<double, A1>& reduce(vector<double, A1> &b, const matrix<double, A2> &a, reduce_mode_type reduce_mode)
	{
		switch (reduce_mode)
		{
		case rm_col_min:
			return reduce_col_min(b, a);
		case rm_col_max:
			return reduce_col_max(b, a);
		case rm_col_sum:
			return reduce_col_sum(b, a);
		case rm_col_avg:
			reduce_col_sum(b, a);
			convert_scale(b, b, 1.0 / static_cast<double>(a.rows()));
			return b;
		case rm_row_min:
			return reduce_row_min(b, a);
		case rm_row_max:
			return reduce_row_max(b, a);
		case rm_row_sum:
			return reduce_row_sum(b, a);
		case rm_row_avg:
			reduce_row_sum(b, a);
			convert_scale(b, b, 1.0 / static_cast<double>(a.rows()));
			return b;
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

} // namespace core

#endif
