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

#ifndef __CORE_CPU_SUM_Y_INT32_H__
#define __CORE_CPU_SUM_Y_INT32_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/sum/kernel_sum_int32.h"

namespace core
{
	// Computes the sum of elements across the y axis of a matrix

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed char, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed char, 8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned char, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned char, 8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed short, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed short, 4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned short, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned short, 4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<signed int, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<unsigned int, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<float, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<float, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<float, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sum_y(vector<signed int, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<double, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<double, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<double, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the y axis of a matrix

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed char, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed char, 8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned char, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned char, 8, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed short, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed short, 4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned short, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned short, 4, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<signed int, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<unsigned int, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<float, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<float, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<float, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<double, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<double, 4, 4, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<double, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the y axis of a tensor

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed char, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed char, 8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned char, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned char, 8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed short, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed short, 4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned short, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned short, 4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<signed int, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<unsigned int, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<float, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<float, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<float, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sum_y(matrix<signed int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<double, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<double, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<double, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the sum of elements across the y axis of a tensor

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed char, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed char, 8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned char, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned char, 8, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed short, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<signed short, 4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned short, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_sumt_int32<unsigned short, 4, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<signed int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<signed int, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<signed int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<unsigned int, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<unsigned int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<float, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<float, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<float, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sum_y(tensor<signed int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.row_size())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(0);
		if (cpu_inst::is_support_avx2())
			kernel_sumt_int32<double, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sumt_int32<double, 4, 4, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_sumt_int32<double, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
