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

#ifndef __CORE_CPU_CONVERT_SCALE_FLOAT_H__
#define __CORE_CPU_CONVERT_SCALE_FLOAT_H__

#include "../scalar.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "kernel/kernel_convert_scale_float.h"

namespace core
{
	// Scalar data-type conversion and scaling

	template <class T, class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<T, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_scale_float<T, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<signed char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<unsigned char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<signed short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<unsigned short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<signed int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<signed int, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<signed int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<unsigned int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<float, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<float, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse())
			kernel_convert_scale_float<float, inst_sse>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<float, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& convert_scale_float(scalar<float, A1> &b, const scalar<double, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<double, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<double, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<double, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	// Vector data-type conversion and scaling

	template <class T, class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<T, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_scale_float<T, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<signed char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<unsigned char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<signed short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<unsigned short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<signed int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<signed int, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<signed int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<unsigned int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<float, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<float, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse())
			kernel_convert_scale_float<float, inst_sse>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<float, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& convert_scale_float(vector<float, A1> &b, const vector<double, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<double, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<double, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<double, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	// Matrix data-type conversion and scaling

	template <class T, class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<T, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_scale_float<T, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<signed char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<unsigned char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<signed short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<unsigned short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<signed int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<signed int, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<signed int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<unsigned int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<float, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<float, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse())
			kernel_convert_scale_float<float, inst_sse>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<float, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& convert_scale_float(matrix<float, A1> &b, const matrix<double, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<double, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<double, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<double, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	// Tensor data-type conversion and scaling

	template <class T, class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<T, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_scale_float<T, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<signed char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<unsigned char, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned char, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned char, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<signed short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<signed short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<signed short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<unsigned short, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned short, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse41())
			kernel_convert_scale_float<unsigned short, inst_sse41>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned short, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<signed int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<signed int, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<signed int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<signed int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<unsigned int, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_scale_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<unsigned int, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<float, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<float, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse())
			kernel_convert_scale_float<float, inst_sse>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<float, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& convert_scale_float(tensor<float, A1> &b, const tensor<double, A2> &a, float scale)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_scale_float<double, inst_avx>()(a.size(), a.data(), b.data(), scale);
		else if (global::is_support_sse2())
			kernel_convert_scale_float<double, inst_sse2>()(a.size(), a.data(), b.data(), scale);
		else
			kernel_convert_scale_float<double, inst_none>()(a.size(), a.data(), b.data(), scale);
		return b;
	}

} // namespace core

#endif
