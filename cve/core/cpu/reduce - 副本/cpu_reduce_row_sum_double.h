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

#ifndef __CORE_CPU_REDUCE_ROW_SUM_DOUBLE_H__
#define __CORE_CPU_REDUCE_ROW_SUM_DOUBLE_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/reduce/kernel_reduce_row_sum_double.h"

namespace core
{
	// Get the summation of each row of matrix

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			block_reduce_row_sum_double<16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			block_reduce_row_sum_double<8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			block_reduce_row_sum_double<16, 16, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			block_reduce_row_sum_double<8, 16, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			block_reduce_row_sum_double<8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			block_reduce_row_sum_double<4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			block_reduce_row_sum_double<8, 8, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			block_reduce_row_sum_double<4, 8, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			block_reduce_row_sum_double<8, 4, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			block_reduce_row_sum_double<4, 4, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			block_reduce_row_sum_double<8, 4, cpu_avx2>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			block_reduce_row_sum_double<4, 4, cpu_sse41>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			block_reduce_row_sum_double<8, 8, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			block_reduce_row_sum_double<4, 4, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_row_sum(vector<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			block_reduce_row_sum_double<8, 4, cpu_avx>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			block_reduce_row_sum_double<4, 4, cpu_sse3>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			block_reduce_row_sum_double<4, 4, cpu_none>()(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
