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

#ifndef __CORE_CPU_REDUCE_ROW_SUM_XY_DOUBLE_H__
#define __CORE_CPU_REDUCE_ROW_SUM_XY_DOUBLE_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/reduce/kernel_reduce_sum_double.h"

namespace core
{
	// Computes the sum of elements of a matrix

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<signed char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed char, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed char, 8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<unsigned char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned char, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned char, 8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<signed short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed short, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed short, 4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<unsigned short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned short, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned short, 4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<signed int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<signed int, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<signed int, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<unsigned int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned int, 8, 4, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<float, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<float, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<float, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_reduce_sum_xy(double &b, const matrix<double, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = 0.0;
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<double, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<double, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_reduce_sum_double<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	// Computes the sum of elements of a matrix

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<signed char, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed char, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed char, 8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned char, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned char, 8, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<signed short, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed short, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed short, 4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned short, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned short, 4, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<signed int, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<signed int, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<signed int, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned int, 8, 4, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<float, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<float, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<float, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_reduce_sum_xy(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<double, 8, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<double, 4, 4, cpu_sse3>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_reduce_sum_double<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	// Computes the sum of elements across the x and y axis of a tensor

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed char, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed char, 8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned char, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned char, 8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed short, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed short, 4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned short, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned short, 4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<signed int, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<signed int, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned int, 8, 4, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<float, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<float, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<float, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_reduce_sum_xy(vector<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<double, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<double, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<double, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the x and y axis of a tensor

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed char, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed char, 8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned char, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned char, 8, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<signed short, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<signed short, 4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned short, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned short, 4, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<signed int, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<signed int, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<signed int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx2())
			kernel_reduce_sum_double<unsigned int, 8, 4, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<unsigned int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<float, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<float, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<float, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_reduce_sum_xy(tensor<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0.0);
		if (cpu_inst::is_support_avx())
			kernel_reduce_sum_double<double, 8, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse3())
			kernel_reduce_sum_double<double, 4, 4, cpu_sse3>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_reduce_sum_double<double, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

} // namespace core

#endif
