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

#ifndef __CORE_CPU_CONVERT_DOUBLE_H__
#define __CORE_CPU_CONVERT_DOUBLE_H__

#include <cstring>
#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/convert/kernel_convert_double.h"

namespace core
{
	// Scalar data-type conversion

	template <class T, class A1, class A2>
	scalar<T, A1>& cpu_convert_double(scalar<T, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_double<T, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed char, A1>& cpu_convert_double(scalar<signed char, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& cpu_convert_double(scalar<unsigned char, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& cpu_convert_double(scalar<signed short, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& cpu_convert_double(scalar<unsigned short, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& cpu_convert_double(scalar<signed int, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<signed int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& cpu_convert_double(scalar<unsigned int, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<unsigned int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_convert_double(scalar<float, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<float, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<float, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<float, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<double, A1>& cpu_convert_double(scalar<double, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		::std::memcpy(b.data(), a.data(), a.size());
		return b;
	}

	// Vector data-type conversion

	template <class T, class A1, class A2>
	vector<T, A1>& cpu_convert_double(vector<T, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_double<T, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed char, A1>& cpu_convert_double(vector<signed char, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_convert_double(vector<unsigned char, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_convert_double(vector<signed short, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_convert_double(vector<unsigned short, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_convert_double(vector<signed int, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<signed int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_convert_double(vector<unsigned int, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<unsigned int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_convert_double(vector<float, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<float, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<float, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<float, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_convert_double(vector<double, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		::std::memcpy(b.data(), a.data(), a.size());
		return b;
	}

	// Matrix data-type conversion

	template <class T, class A1, class A2>
	matrix<T, A1>& cpu_convert_double(matrix<T, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_double<T, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_convert_double(matrix<signed char, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_convert_double(matrix<unsigned char, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_convert_double(matrix<signed short, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_convert_double(matrix<unsigned short, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_convert_double(matrix<signed int, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<signed int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_convert_double(matrix<unsigned int, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<unsigned int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_convert_double(matrix<float, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<float, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<float, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<float, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_convert_double(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		::std::memcpy(b.data(), a.data(), a.size());
		return b;
	}

	// Tensor data-type conversion

	template <class T, class A1, class A2>
	tensor<T, A1>& cpu_convert_double(tensor<T, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		kernel_convert_double<T, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_convert_double(tensor<signed char, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_convert_double(tensor<unsigned char, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned char, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned char, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned char, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_convert_double(tensor<signed short, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<signed short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_convert_double(tensor<unsigned short, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_convert_double<unsigned short, cpu_avx2>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned short, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned short, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_convert_double(tensor<signed int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<signed int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<signed int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<signed int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_convert_double(tensor<unsigned int, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<unsigned int, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<unsigned int, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<unsigned int, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_convert_double(tensor<float, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_convert_double<float, cpu_avx>()(a.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_convert_double<float, cpu_sse2>()(a.size(), a.data(), b.data());
		else
			kernel_convert_double<float, cpu_none>()(a.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_convert_double(tensor<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		::std::memcpy(b.data(), a.data(), a.size());
		return b;
	}

} // namespace core

#endif
