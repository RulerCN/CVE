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

#ifndef __CORE_CPU_ARITHMETIC_ADDS_H__
#define __CORE_CPU_ARITHMETIC_ADDS_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/arithmetic/kernel_arithmetic_adds.h"
#include "../kernel/arithmetic/kernel_arithmetic_adds_value.h"

namespace core
{
	// Saturation addition of value and scalar

	template <class A1, class A2>
	scalar<signed char, A1>& cpu_adds(scalar<signed char, A1> &c, const signed char a, const scalar<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& cpu_adds(scalar<unsigned char, A1> &c, const unsigned char a, const scalar<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& cpu_adds(scalar<signed short, A1> &c, const signed short a, const scalar<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& cpu_adds(scalar<unsigned short, A1> &c, const unsigned short a, const scalar<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& cpu_adds(scalar<signed int, A1> &c, const signed int a, const scalar<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& cpu_adds(scalar<unsigned int, A1> &c, const unsigned int a, const scalar<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Saturation addition of value and vector

	template <class A1, class A2>
	vector<signed char, A1>& cpu_adds(vector<signed char, A1> &c, const signed char a, const vector<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_adds(vector<unsigned char, A1> &c, const unsigned char a, const vector<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_adds(vector<signed short, A1> &c, const signed short a, const vector<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_adds(vector<unsigned short, A1> &c, const unsigned short a, const vector<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_adds(vector<signed int, A1> &c, const signed int a, const vector<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_adds(vector<unsigned int, A1> &c, const unsigned int a, const vector<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Saturation addition of value and matrix

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_adds(matrix<signed char, A1> &c, const signed char a, const matrix<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_adds(matrix<unsigned char, A1> &c, const unsigned char a, const matrix<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_adds(matrix<signed short, A1> &c, const signed short a, const matrix<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_adds(matrix<unsigned short, A1> &c, const unsigned short a, const matrix<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_adds(matrix<signed int, A1> &c, const signed int a, const matrix<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_adds(matrix<unsigned int, A1> &c, const unsigned int a, const matrix<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Saturation addition of value and tensor

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_adds(tensor<signed char, A1> &c, const signed char a, const tensor<signed char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_adds(tensor<unsigned char, A1> &c, const unsigned char a, const tensor<unsigned char> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_adds(tensor<signed short, A1> &c, const signed short a, const tensor<signed short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_adds(tensor<unsigned short, A1> &c, const unsigned short a, const tensor<unsigned short> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_adds(tensor<signed int, A1> &c, const signed int a, const tensor<signed int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_adds(tensor<unsigned int, A1> &c, const unsigned int a, const tensor<unsigned int> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_adds_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Saturation addition of scalar and scalar

	template <class A1, class A2, class A3>
	scalar<signed char, A1>& cpu_adds(scalar<signed char, A1> &c, const scalar<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned char, A1>& cpu_adds(scalar<unsigned char, A1> &c, const scalar<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed short, A1>& cpu_adds(scalar<signed short, A1> &c, const scalar<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned short, A1>& cpu_adds(scalar<unsigned short, A1> &c, const scalar<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<signed int, A1>& cpu_adds(scalar<signed int, A1> &c, const scalar<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	scalar<unsigned int, A1>& cpu_adds(scalar<unsigned int, A1> &c, const scalar<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of vector and vector

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_adds(vector<signed char, A1> &c, const vector<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_adds(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_adds(vector<signed short, A1> &c, const vector<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_adds(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_adds(vector<signed int, A1> &c, const vector<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_adds(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of matrix and matrix

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_adds(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_adds(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_adds(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_adds(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_adds(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_adds(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of tensor and tensor

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_adds(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const tensor<signed char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_adds(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const tensor<unsigned char, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned char, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned char, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned char, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_adds(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const tensor<signed short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_adds(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const tensor<unsigned short, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned short, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned short, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned short, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_adds(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const tensor<signed int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<signed int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<signed int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<signed int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_adds(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const tensor<unsigned int, A3> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != a.size() || c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_adds<unsigned int, cpu_avx2>()(c.size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds<unsigned int, cpu_sse2>()(c.size(), a.data(), b.data(), c.data());
		else
			kernel_adds<unsigned int, cpu_none>()(c.size(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of vector and scalar

	template <class A1, class A2, class A3>
	vector<signed char, A1>& cpu_adds(vector<signed char, A1> &c, const vector<signed char, A2> &a, const scalar<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned char, A1>& cpu_adds(vector<unsigned char, A1> &c, const vector<unsigned char, A2> &a, const scalar<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned char, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned char, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned char, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed short, A1>& cpu_adds(vector<signed short, A1> &c, const vector<signed short, A2> &a, const scalar<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned short, A1>& cpu_adds(vector<unsigned short, A1> &c, const vector<unsigned short, A2> &a, const scalar<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned short, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned short, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned short, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<signed int, A1>& cpu_adds(vector<signed int, A1> &c, const vector<signed int, A2> &a, const scalar<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	vector<unsigned int, A1>& cpu_adds(vector<unsigned int, A1> &c, const vector<unsigned int, A2> &a, const scalar<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != a.size() || c.dimension() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned int, cpu_avx2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned int, cpu_sse2>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned int, cpu_none>()(c.length(), c.dimension(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of matrix and vector

	template <class A1, class A2, class A3>
	matrix<signed char, A1>& cpu_adds(matrix<signed char, A1> &c, const matrix<signed char, A2> &a, const vector<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned char, A1>& cpu_adds(matrix<unsigned char, A1> &c, const matrix<unsigned char, A2> &a, const vector<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned char, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned char, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned char, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed short, A1>& cpu_adds(matrix<signed short, A1> &c, const matrix<signed short, A2> &a, const vector<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned short, A1>& cpu_adds(matrix<unsigned short, A1> &c, const matrix<unsigned short, A2> &a, const vector<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned short, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned short, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned short, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<signed int, A1>& cpu_adds(matrix<signed int, A1> &c, const matrix<signed int, A2> &a, const vector<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	matrix<unsigned int, A1>& cpu_adds(matrix<unsigned int, A1> &c, const matrix<unsigned int, A2> &a, const vector<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != a.size() || c.row_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned int, cpu_avx2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned int, cpu_sse2>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned int, cpu_none>()(c.rows(), c.row_size(), a.data(), b.data(), c.data());
		return c;
	}

	// Saturation addition of tensor and matrix

	template <class A1, class A2, class A3>
	tensor<signed char, A1>& cpu_adds(tensor<signed char, A1> &c, const tensor<signed char, A2> &a, const matrix<signed char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned char, A1>& cpu_adds(tensor<unsigned char, A1> &c, const tensor<unsigned char, A2> &a, const matrix<unsigned char, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned char, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned char, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned char, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed short, A1>& cpu_adds(tensor<signed short, A1> &c, const tensor<signed short, A2> &a, const matrix<signed short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned short, A1>& cpu_adds(tensor<unsigned short, A1> &c, const tensor<unsigned short, A2> &a, const matrix<unsigned short, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned short, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned short, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned short, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<signed int, A1>& cpu_adds(tensor<signed int, A1> &c, const tensor<signed int, A2> &a, const matrix<signed int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<signed int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<signed int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<signed int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

	template <class A1, class A2, class A3>
	tensor<unsigned int, A1>& cpu_adds(tensor<unsigned int, A1> &c, const tensor<unsigned int, A2> &a, const matrix<unsigned int, A3> &b)
	{
		if (c.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != a.size() || c.matrix_size() != b.size())
			throw ::std::invalid_argument(invalid_shape);

		if (cpu_inst::is_support_avx2())
			kernel_adds_element<unsigned int, cpu_avx2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_adds_element<unsigned int, cpu_sse2>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		else
			kernel_adds_element<unsigned int, cpu_none>()(c.batch(), c.matrix_size(), a.data(), b.data(), c.data());
		return c;
	}

} // namespace core

#endif
