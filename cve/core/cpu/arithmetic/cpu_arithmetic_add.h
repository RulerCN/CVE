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

#ifndef __CORE_CPU_ARITHMETIC_ADD_H__
#define __CORE_CPU_ARITHMETIC_ADD_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_add.h"
#include "../kernel/arithmetic/kernel_arithmetic_add_value.h"

namespace core
{
	// Addition of scalar and value

	template <class A1, class A2>
	scalar<signed char, A1>& cpu_add(scalar<signed char, A1> &c, const signed char a, const scalar<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& cpu_add(scalar<unsigned char, A1> &c, const unsigned char a, const scalar<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& cpu_add(scalar<signed short, A1> &c, const signed short a, const scalar<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& cpu_add(scalar<unsigned short, A1> &c, const unsigned short a, const scalar<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& cpu_add(scalar<signed int, A1> &c, const signed int a, const scalar<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& cpu_add(scalar<unsigned int, A1> &c, const unsigned int a, const scalar<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_add(scalar<float, A1> &c, const float a, const scalar<float> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<double, A1>& cpu_add(scalar<double, A1> &c, const double a, const scalar<double> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of vector and value

	template <class A1, class A2>
	vector<signed char, A1>& cpu_add(vector<signed char, A1> &c, const signed char a, const vector<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_add(vector<unsigned char, A1> &c, const unsigned char a, const vector<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_add(vector<signed short, A1> &c, const signed short a, const vector<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_add(vector<unsigned short, A1> &c, const unsigned short a, const vector<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_add(vector<signed int, A1> &c, const signed int a, const vector<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_add(vector<unsigned int, A1> &c, const unsigned int a, const vector<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_add(vector<float, A1> &c, const float a, const vector<float> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_add(vector<double, A1> &c, const double a, const vector<double> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of matrix and value

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_add(matrix<signed char, A1> &c, const signed char a, const matrix<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_add(matrix<unsigned char, A1> &c, const unsigned char a, const matrix<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_add(matrix<signed short, A1> &c, const signed short a, const matrix<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_add(matrix<unsigned short, A1> &c, const unsigned short a, const matrix<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_add(matrix<signed int, A1> &c, const signed int a, const matrix<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_add(matrix<unsigned int, A1> &c, const unsigned int a, const matrix<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_add(matrix<float, A1> &c, const float a, const matrix<float> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_add(matrix<double, A1> &c, const double a, const matrix<double> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of tensor and value

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_add(tensor<signed char, A1> &c, const signed char a, const tensor<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_add(tensor<unsigned char, A1> &c, const unsigned char a, const tensor<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_add(tensor<signed short, A1> &c, const signed short a, const tensor<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_add(tensor<unsigned short, A1> &c, const unsigned short a, const tensor<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_add(tensor<signed int, A1> &c, const signed int a, const tensor<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_add(tensor<unsigned int, A1> &c, const unsigned int a, const tensor<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_add(tensor<float, A1> &c, const float a, const tensor<float> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_add(tensor<double, A1> &c, const double a, const tensor<double> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of scalar and scalar

	template <class A1, class A2, class A3>
	scalar<signed char, A1>& cpu_add(scalar<signed char, A1> &c, const scalar<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned char, A1>& cpu_add(scalar<unsigned char, A1> &c, const scalar<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed short, A1>& cpu_add(scalar<signed short, A1> &c, const scalar<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned short, A1>& cpu_add(scalar<unsigned short, A1> &c, const scalar<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed int, A1>& cpu_add(scalar<signed int, A1> &c, const scalar<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned int, A1>& cpu_add(scalar<unsigned int, A1> &c, const scalar<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<float, A1>& cpu_add(scalar<float, A1> &c, const scalar<float, A2> &a, const scalar<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<double, A1>& cpu_add(scalar<double, A1> &c, const scalar<double, A2> &a, const scalar<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of vector and vector

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_add(vector<signed char, A1> &c, const vector<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_add(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_add(vector<signed short, A1> &c, const vector<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_add(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_add(vector<signed int, A1> &c, const vector<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_add(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<float, A1>& cpu_add(vector<float, A1> &c, const vector<float, A2> &a, const vector<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<double, A1>& cpu_add(vector<double, A1> &c, const vector<double, A2> &a, const vector<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of matrix and matrix

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_add(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_add(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_add(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_add(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_add(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_add(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<float, A1>& cpu_add(matrix<float, A1> &c, const matrix<float, A2> &a, const matrix<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<double, A1>& cpu_add(matrix<double, A1> &c, const matrix<double, A2> &a, const matrix<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of tensor and tensor

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_add(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const tensor<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_add(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const tensor<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_add(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const tensor<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_add(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const tensor<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_add(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const tensor<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_add(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const tensor<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<float, A1>& cpu_add(tensor<float, A1> &c, const tensor<float, A2> &a, const tensor<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<double, A1>& cpu_add(tensor<double, A1> &c, const tensor<double, A2> &a, const tensor<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_add<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of vector and scalar

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_add(vector<signed char, A1> &c, const vector<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_add(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_add(vector<signed short, A1> &c, const vector<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_add(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_add(vector<signed int, A1> &c, const vector<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_add(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<float, A1>& cpu_add(vector<float, A1> &c, const vector<float, A2> &a, const scalar<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<float, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_element<float, cpu_sse>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<float, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<double, A1>& cpu_add(vector<double, A1> &c, const vector<double, A2> &a, const scalar<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<double, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<double, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_add_element<double, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of matrix and vector

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_add(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_add(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_add(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_add(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_add(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_add(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<float, A1>& cpu_add(matrix<float, A1> &c, const matrix<float, A2> &a, const vector<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<float, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_element<float, cpu_sse>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<float, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<double, A1>& cpu_add(matrix<double, A1> &c, const matrix<double, A2> &a, const vector<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<double, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<double, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<double, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	// Addition of tensor and matrix

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_add(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_add(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_add(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_add(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_add(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<signed int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<signed int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<signed int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_add(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_add_element<unsigned int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<unsigned int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<unsigned int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<float, A1>& cpu_add(tensor<float, A1> &c, const tensor<float, A2> &a, const matrix<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<float, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_element<float, cpu_sse>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<float, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<double, A1>& cpu_add(tensor<double, A1> &c, const tensor<double, A2> &a, const matrix<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_add_element<double, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_element<double, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_add_element<double, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

} // namespace core

#endif
