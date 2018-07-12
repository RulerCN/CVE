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

#ifndef __CORE_CPU_ARITHMETIC_SUB_H__
#define __CORE_CPU_ARITHMETIC_SUB_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_sub.h"
#include "../kernel/arithmetic/kernel_arithmetic_sub_value.h"

namespace core
{
	// Subtraction of scalar and value

	template <class A1, class A2>
	scalar<signed char, A1>& cpu_sub(scalar<signed char, A1> &c, const scalar<signed char, A2> &a, const signed char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& cpu_sub(scalar<unsigned char, A1> &c, const scalar<unsigned char, A2> &a, const unsigned char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& cpu_sub(scalar<signed short, A1> &c, const scalar<signed short, A2> &a, const signed short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& cpu_sub(scalar<unsigned short, A1> &c, const scalar<unsigned short, A2> &a, const unsigned short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& cpu_sub(scalar<signed int, A1> &c, const scalar<signed int, A2> &a, const signed int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& cpu_sub(scalar<unsigned int, A1> &c, const scalar<unsigned int, A2> &a, const unsigned int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_sub(scalar<float, A1> &c, const scalar<float, A2> &a, const float b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<float, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_value<float, cpu_sse>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<float, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<double, A1>& cpu_sub(scalar<double, A1> &c, const scalar<double, A2> &a, const double b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<double, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<double, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<double, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	// Subtraction of vector and value

	template <class A1, class A2>
	vector<signed char, A1>& cpu_sub(vector<signed char, A1> &c, const vector<signed char, A2> &a, const signed char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_sub(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const unsigned char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_sub(vector<signed short, A1> &c, const vector<signed short, A2> &a, const signed short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_sub(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const unsigned short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_sub(vector<signed int, A1> &c, const vector<signed int, A2> &a, const signed int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_sub(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const unsigned int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_sub(vector<float, A1> &c, const vector<float, A2> &a, const float b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<float, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_value<float, cpu_sse>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<float, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_sub(vector<double, A1> &c, const vector<double, A2> &a, const double b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<double, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<double, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<double, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	// Subtraction of matrix and value

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_sub(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const signed char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_sub(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const unsigned char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_sub(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const signed short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_sub(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const unsigned short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_sub(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const signed int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_sub(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const unsigned int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_sub(matrix<float, A1> &c, const matrix<float, A2> &a, const float b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<float, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_value<float, cpu_sse>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<float, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_sub(matrix<double, A1> &c, const matrix<double, A2> &a, const double b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<double, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<double, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<double, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	// Subtraction of tensor and value

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_sub(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const signed char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_sub(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const unsigned char b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned char, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned char, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned char, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_sub(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const signed short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_sub(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const unsigned short b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned short, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned short, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned short, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_sub(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const signed int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<signed int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<signed int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<signed int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_sub(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const unsigned int b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub_value<unsigned int, cpu_avx2>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<unsigned int, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<unsigned int, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_sub(tensor<float, A1> &c, const tensor<float, A2> &a, const float b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<float, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_value<float, cpu_sse>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<float, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_sub(tensor<double, A1> &c, const tensor<double, A2> &a, const double b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub_value<double, cpu_avx>()(c.size(), a.data(), b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_value<double, cpu_sse2>()(c.size(), a.data(), b, c.data());
		else
			kernel_sub_value<double, cpu_none>()(c.size(), a.data(), b, c.data());
		return c;
	}

	// Subtraction of scalar and scalar

	template <class A1, class A2, class A3>
	scalar<signed char, A1>& cpu_sub(scalar<signed char, A1> &c, const scalar<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned char, A1>& cpu_sub(scalar<unsigned char, A1> &c, const scalar<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed short, A1>& cpu_sub(scalar<signed short, A1> &c, const scalar<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned short, A1>& cpu_sub(scalar<unsigned short, A1> &c, const scalar<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed int, A1>& cpu_sub(scalar<signed int, A1> &c, const scalar<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned int, A1>& cpu_sub(scalar<unsigned int, A1> &c, const scalar<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<float, A1>& cpu_sub(scalar<float, A1> &c, const scalar<float, A2> &a, const scalar<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<double, A1>& cpu_sub(scalar<double, A1> &c, const scalar<double, A2> &a, const scalar<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of vector and vector

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_sub(vector<signed char, A1> &c, const vector<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_sub(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_sub(vector<signed short, A1> &c, const vector<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_sub(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_sub(vector<signed int, A1> &c, const vector<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_sub(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<float, A1>& cpu_sub(vector<float, A1> &c, const vector<float, A2> &a, const vector<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<double, A1>& cpu_sub(vector<double, A1> &c, const vector<double, A2> &a, const vector<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of matrix and matrix

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_sub(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_sub(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_sub(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_sub(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_sub(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_sub(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<float, A1>& cpu_sub(matrix<float, A1> &c, const matrix<float, A2> &a, const matrix<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<double, A1>& cpu_sub(matrix<double, A1> &c, const matrix<double, A2> &a, const matrix<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of tensor and tensor

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_sub(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const tensor<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_sub(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const tensor<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_sub(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const tensor<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_sub(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const tensor<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_sub(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const tensor<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_sub(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const tensor<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sub<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<float, A1>& cpu_sub(tensor<float, A1> &c, const tensor<float, A2> &a, const tensor<float, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<float, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub<float, cpu_sse>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<float, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<double, A1>& cpu_sub(tensor<double, A1> &c, const tensor<double, A2> &a, const tensor<double, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_sub<double, cpu_avx>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub<double, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_sub<double, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of vector and scalar

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_sub(vector<signed char, A1> &c, const vector<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_sub(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_sub(vector<signed short, A1> &c, const vector<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_sub(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_sub(vector<signed int, A1> &c, const vector<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_sub(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<float, A1>& cpu_sub(vector<float, A1> &c, const vector<float, A2> &a, const scalar<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<float, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_element<float, cpu_sse>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<float, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<double, A1>& cpu_sub(vector<double, A1> &c, const vector<double, A2> &a, const scalar<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<double, cpu_avx>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<double, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<double, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of matrix and vector

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_sub(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_sub(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_sub(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_sub(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_sub(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_sub(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<float, A1>& cpu_sub(matrix<float, A1> &c, const matrix<float, A2> &a, const vector<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<float, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_element<float, cpu_sse>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<float, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<double, A1>& cpu_sub(matrix<double, A1> &c, const matrix<double, A2> &a, const vector<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<double, cpu_avx>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<double, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<double, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	// Subtraction of tensor and matrix

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_sub(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_sub(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_sub(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_sub(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_sub(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<signed int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<signed int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<signed int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_sub(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_sub_element<unsigned int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<unsigned int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<unsigned int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<float, A1>& cpu_sub(tensor<float, A1> &c, const tensor<float, A2> &a, const matrix<float, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<float, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_sub_element<float, cpu_sse>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<float, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<double, A1>& cpu_sub(tensor<double, A1> &c, const tensor<double, A2> &a, const matrix<double, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx())
			kernel_sub_element<double, cpu_avx>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_sub_element<double, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_sub_element<double, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

} // namespace core

#endif
