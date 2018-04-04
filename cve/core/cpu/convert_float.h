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

#ifndef __CORE_CPU_CONVERT_FLOAT_H__
#define __CORE_CPU_CONVERT_FLOAT_H__

#include "../scalar.h"
#include "../vector.h"
#include "../matrix.h"
#include "../tensor.h"
#include "kernel/kernel_convert_float.h"

namespace core
{
	// Scalar data-type conversion

	template <class T, class A1, class A2>
	scalar<T, A1>& convert_float(scalar<T, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_float<T, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed char, A1>& convert_float(scalar<signed char, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& convert_float(scalar<unsigned char, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& convert_float(scalar<signed short, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& convert_float(scalar<unsigned short, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& convert_float(scalar<signed int, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<signed int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& convert_float(scalar<unsigned int, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<unsigned int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<double, A1>& convert_float(scalar<double, A1> &b, const scalar<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<double, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<double, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<double, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	// Vector data-type conversion

	template <class T, class A1, class A2>
	vector<T, A1>& convert_float(vector<T, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_float<T, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed char, A1>& convert_float(vector<signed char, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& convert_float(vector<unsigned char, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed short, A1>& convert_float(vector<signed short, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& convert_float(vector<unsigned short, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& convert_float(vector<signed int, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<signed int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& convert_float(vector<unsigned int, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<unsigned int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& convert_float(vector<double, A1> &b, const vector<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<double, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<double, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<double, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	// Matrix data-type conversion

	template <class T, class A1, class A2>
	matrix<T, A1>& convert_float(matrix<T, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_float<T, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed char, A1>& convert_float(matrix<signed char, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& convert_float(matrix<unsigned char, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& convert_float(matrix<signed short, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& convert_float(matrix<unsigned short, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& convert_float(matrix<signed int, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<signed int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& convert_float(matrix<unsigned int, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<unsigned int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& convert_float(matrix<double, A1> &b, const matrix<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<double, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<double, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<double, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	// Tensor data-type conversion

	template <class T, class A1, class A2>
	tensor<T, A1>& convert_float(tensor<T, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_float<T, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed char, A1>& convert_float(tensor<signed char, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& convert_float(tensor<unsigned char, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned char, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned char, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned char, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& convert_float(tensor<signed short, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<signed int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& convert_float(tensor<unsigned short, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx2())
			kernel_convert_float<unsigned int, inst_avx2>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& convert_float(tensor<signed int, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<signed int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<signed int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<signed int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& convert_float(tensor<unsigned int, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<unsigned int, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<unsigned int, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<unsigned int, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& convert_float(tensor<double, A1> &b, const tensor<float A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (global::is_support_avx())
			kernel_convert_float<double, inst_avx>()(a.size(), a.data(), b.data());
		else if (global::is_support_sse2())
			kernel_convert_float<double, inst_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_float<double, inst_none>()(a.size(), a.data(), b.data());
		return b;
	}

} // namespace core

#endif
