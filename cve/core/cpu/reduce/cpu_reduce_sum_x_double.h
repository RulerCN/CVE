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

#ifndef __CORE_CPU_REDUCE_ROW_SUM_X_DOUBLE_H__
#define __CORE_CPU_REDUCE_ROW_SUM_X_DOUBLE_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/reduce/kernel_reduce_sum_double.h"

namespace core
{
	// Computes the sum of elements of a vector

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<signed char, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<unsigned char, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<signed short, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<unsigned short, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<signed int, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<unsigned int, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 4, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<float, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A1, class A2>
	double& cpu_reduce_sum_x(double &b, const vector<double, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	// Computes the sum of elements across the x axis of a matrix

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 4, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 4, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_x(vector<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the x axis of a tensor

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<8, 4, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<4, 4, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_x(matrix<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<8, 4, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<4, 4, cpu_sse3>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_reduce_sum_double<4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
