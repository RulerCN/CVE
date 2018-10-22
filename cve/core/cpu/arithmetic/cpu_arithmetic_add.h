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
	// Addition of value and scalar

	template <class A>
	scalar<signed char, A>& cpu_add(scalar<signed char, A> &c, signed char a, const scalar<signed char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<unsigned char, A>& cpu_add(scalar<unsigned char, A> &c, unsigned char a, const scalar<unsigned char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<signed short, A>& cpu_add(scalar<signed short, A> &c, signed short a, const scalar<signed short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<unsigned short, A>& cpu_add(scalar<unsigned short, A> &c, unsigned short a, const scalar<unsigned short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<signed int, A>& cpu_add(scalar<signed int, A> &c, signed int a, const scalar<signed int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<unsigned int, A>& cpu_add(scalar<unsigned int, A> &c, unsigned int a, const scalar<unsigned int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<float, A>& cpu_add(scalar<float, A> &c, float a, const scalar<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	scalar<double, A>& cpu_add(scalar<double, A> &c, double a, const scalar<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of value and vector

	template <class A>
	vector<signed char, A>& cpu_add(vector<signed char, A> &c, signed char a, const vector<signed char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<unsigned char, A>& cpu_add(vector<unsigned char, A> &c, unsigned char a, const vector<unsigned char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<signed short, A>& cpu_add(vector<signed short, A> &c, signed short a, const vector<signed short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<unsigned short, A>& cpu_add(vector<unsigned short, A> &c, unsigned short a, const vector<unsigned short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<signed int, A>& cpu_add(vector<signed int, A> &c, signed int a, const vector<signed int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<unsigned int, A>& cpu_add(vector<unsigned int, A> &c, unsigned int a, const vector<unsigned int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<float, A>& cpu_add(vector<float, A> &c, float a, const vector<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_add(vector<double, A> &c, double a, const vector<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of value and matrix

	template <class A>
	matrix<signed char, A>& cpu_add(matrix<signed char, A> &c, signed char a, const matrix<signed char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_add(matrix<unsigned char, A> &c, unsigned char a, const matrix<unsigned char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<signed short, A>& cpu_add(matrix<signed short, A> &c, signed short a, const matrix<signed short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_add(matrix<unsigned short, A> &c, unsigned short a, const matrix<unsigned short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<signed int, A>& cpu_add(matrix<signed int, A> &c, signed int a, const matrix<signed int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_add(matrix<unsigned int, A> &c, unsigned int a, const matrix<unsigned int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<float, A>& cpu_add(matrix<float, A> &c, float a, const matrix<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	matrix<double, A>& cpu_add(matrix<double, A> &c, double a, const matrix<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Addition of value and tensor

	template <class A>
	tensor<signed char, A>& cpu_add(tensor<signed char, A> &c, signed char a, const tensor<signed char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_add(tensor<unsigned char, A> &c, unsigned char a, const tensor<unsigned char, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned char, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<signed short, A>& cpu_add(tensor<signed short, A> &c, signed short a, const tensor<signed short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_add(tensor<unsigned short, A> &c, unsigned short a, const tensor<unsigned short, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned short, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<signed int, A>& cpu_add(tensor<signed int, A> &c, signed int a, const tensor<signed int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<signed int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_add(tensor<unsigned int, A> &c, unsigned int a, const tensor<unsigned int, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_add_value<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_add_value<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<float, A>& cpu_add(tensor<float, A> &c, float a, const tensor<float, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_add_value<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_add_value<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_add_value<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A>
	tensor<double, A>& cpu_add(tensor<double, A> &c, double a, const tensor<double, A> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
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

	template <class A>
	scalar<signed char, A>& cpu_add(scalar<signed char, A> &c, const scalar<signed char, A> &a, const scalar<signed char, A> &b)
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

	template <class A>
	scalar<unsigned char, A>& cpu_add(scalar<unsigned char, A> &c, const scalar<unsigned char, A> &a, const scalar<unsigned char, A> &b)
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

	template <class A>
	scalar<signed short, A>& cpu_add(scalar<signed short, A> &c, const scalar<signed short, A> &a, const scalar<signed short, A> &b)
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

	template <class A>
	scalar<unsigned short, A>& cpu_add(scalar<unsigned short, A> &c, const scalar<unsigned short, A> &a, const scalar<unsigned short, A> &b)
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

	template <class A>
	scalar<signed int, A>& cpu_add(scalar<signed int, A> &c, const scalar<signed int, A> &a, const scalar<signed int, A> &b)
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

	template <class A>
	scalar<unsigned int, A>& cpu_add(scalar<unsigned int, A> &c, const scalar<unsigned int, A> &a, const scalar<unsigned int, A> &b)
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

	template <class A>
	scalar<float, A>& cpu_add(scalar<float, A> &c, const scalar<float, A> &a, const scalar<float, A> &b)
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

	template <class A>
	scalar<double, A>& cpu_add(scalar<double, A> &c, const scalar<double, A> &a, const scalar<double, A> &b)
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

	template <class A>
	vector<signed char, A>& cpu_add(vector<signed char, A> &c, const vector<signed char, A> &a, const vector<signed char, A> &b)
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

	template <class A>
	vector<unsigned char, A>& cpu_add(vector<unsigned char, A> &c, const vector<unsigned char, A> &a, const vector<unsigned char, A> &b)
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

	template <class A>
	vector<signed short, A>& cpu_add(vector<signed short, A> &c, const vector<signed short, A> &a, const vector<signed short, A> &b)
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

	template <class A>
	vector<unsigned short, A>& cpu_add(vector<unsigned short, A> &c, const vector<unsigned short, A> &a, const vector<unsigned short, A> &b)
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

	template <class A>
	vector<signed int, A>& cpu_add(vector<signed int, A> &c, const vector<signed int, A> &a, const vector<signed int, A> &b)
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

	template <class A>
	vector<unsigned int, A>& cpu_add(vector<unsigned int, A> &c, const vector<unsigned int, A> &a, const vector<unsigned int, A> &b)
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

	template <class A>
	vector<float, A>& cpu_add(vector<float, A> &c, const vector<float, A> &a, const vector<float, A> &b)
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

	template <class A>
	vector<double, A>& cpu_add(vector<double, A> &c, const vector<double, A> &a, const vector<double, A> &b)
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

	template <class A>
	matrix<signed char, A>& cpu_add(matrix<signed char, A> &c, const matrix<signed char, A> &a, const matrix<signed char, A> &b)
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

	template <class A>
	matrix<unsigned char, A>& cpu_add(matrix<unsigned char, A> &c, const matrix<unsigned char, A> &a, const matrix<unsigned char, A> &b)
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

	template <class A>
	matrix<signed short, A>& cpu_add(matrix<signed short, A> &c, const matrix<signed short, A> &a, const matrix<signed short, A> &b)
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

	template <class A>
	matrix<unsigned short, A>& cpu_add(matrix<unsigned short, A> &c, const matrix<unsigned short, A> &a, const matrix<unsigned short, A> &b)
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

	template <class A>
	matrix<signed int, A>& cpu_add(matrix<signed int, A> &c, const matrix<signed int, A> &a, const matrix<signed int, A> &b)
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

	template <class A>
	matrix<unsigned int, A>& cpu_add(matrix<unsigned int, A> &c, const matrix<unsigned int, A> &a, const matrix<unsigned int, A> &b)
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

	template <class A>
	matrix<float, A>& cpu_add(matrix<float, A> &c, const matrix<float, A> &a, const matrix<float, A> &b)
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

	template <class A>
	matrix<double, A>& cpu_add(matrix<double, A> &c, const matrix<double, A> &a, const matrix<double, A> &b)
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

	template <class A>
	tensor<signed char, A>& cpu_add(tensor<signed char, A> &c, const tensor<signed char, A> &a, const tensor<signed char, A> &b)
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

	template <class A>
	tensor<unsigned char, A>& cpu_add(tensor<unsigned char, A> &c, const tensor<unsigned char, A> &a, const tensor<unsigned char, A> &b)
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

	template <class A>
	tensor<signed short, A>& cpu_add(tensor<signed short, A> &c, const tensor<signed short, A> &a, const tensor<signed short, A> &b)
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

	template <class A>
	tensor<unsigned short, A>& cpu_add(tensor<unsigned short, A> &c, const tensor<unsigned short, A> &a, const tensor<unsigned short, A> &b)
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

	template <class A>
	tensor<signed int, A>& cpu_add(tensor<signed int, A> &c, const tensor<signed int, A> &a, const tensor<signed int, A> &b)
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

	template <class A>
	tensor<unsigned int, A>& cpu_add(tensor<unsigned int, A> &c, const tensor<unsigned int, A> &a, const tensor<unsigned int, A> &b)
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

	template <class A>
	tensor<float, A>& cpu_add(tensor<float, A> &c, const tensor<float, A> &a, const tensor<float, A> &b)
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

	template <class A>
	tensor<double, A>& cpu_add(tensor<double, A> &c, const tensor<double, A> &a, const tensor<double, A> &b)
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

	template <class A>
	vector<signed char, A>& cpu_add(vector<signed char, A> &c, const vector<signed char, A> &a, const scalar<signed char, A> &b)
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

	template <class A>
	vector<unsigned char, A>& cpu_add(vector<unsigned char, A> &c, const vector<unsigned char, A> &a, const scalar<unsigned char, A> &b)
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

	template <class A>
	vector<signed short, A>& cpu_add(vector<signed short, A> &c, const vector<signed short, A> &a, const scalar<signed short, A> &b)
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

	template <class A>
	vector<unsigned short, A>& cpu_add(vector<unsigned short, A> &c, const vector<unsigned short, A> &a, const scalar<unsigned short, A> &b)
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

	template <class A>
	vector<signed int, A>& cpu_add(vector<signed int, A> &c, const vector<signed int, A> &a, const scalar<signed int, A> &b)
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

	template <class A>
	vector<unsigned int, A>& cpu_add(vector<unsigned int, A> &c, const vector<unsigned int, A> &a, const scalar<unsigned int, A> &b)
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

	template <class A>
	vector<float, A>& cpu_add(vector<float, A> &c, const vector<float, A> &a, const scalar<float, A> &b)
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

	template <class A>
	vector<double, A>& cpu_add(vector<double, A> &c, const vector<double, A> &a, const scalar<double, A> &b)
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

	template <class A>
	matrix<signed char, A>& cpu_add(matrix<signed char, A> &c, const matrix<signed char, A> &a, const vector<signed char, A> &b)
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

	template <class A>
	matrix<unsigned char, A>& cpu_add(matrix<unsigned char, A> &c, const matrix<unsigned char, A> &a, const vector<unsigned char, A> &b)
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

	template <class A>
	matrix<signed short, A>& cpu_add(matrix<signed short, A> &c, const matrix<signed short, A> &a, const vector<signed short, A> &b)
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

	template <class A>
	matrix<unsigned short, A>& cpu_add(matrix<unsigned short, A> &c, const matrix<unsigned short, A> &a, const vector<unsigned short, A> &b)
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

	template <class A>
	matrix<signed int, A>& cpu_add(matrix<signed int, A> &c, const matrix<signed int, A> &a, const vector<signed int, A> &b)
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

	template <class A>
	matrix<unsigned int, A>& cpu_add(matrix<unsigned int, A> &c, const matrix<unsigned int, A> &a, const vector<unsigned int, A> &b)
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

	template <class A>
	matrix<float, A>& cpu_add(matrix<float, A> &c, const matrix<float, A> &a, const vector<float, A> &b)
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

	template <class A>
	matrix<double, A>& cpu_add(matrix<double, A> &c, const matrix<double, A> &a, const vector<double, A> &b)
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

	template <class A>
	tensor<signed char, A>& cpu_add(tensor<signed char, A> &c, const tensor<signed char, A> &a, const matrix<signed char, A> &b)
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

	template <class A>
	tensor<unsigned char, A>& cpu_add(tensor<unsigned char, A> &c, const tensor<unsigned char, A> &a, const matrix<unsigned char, A> &b)
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

	template <class A>
	tensor<signed short, A>& cpu_add(tensor<signed short, A> &c, const tensor<signed short, A> &a, const matrix<signed short, A> &b)
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

	template <class A>
	tensor<unsigned short, A>& cpu_add(tensor<unsigned short, A> &c, const tensor<unsigned short, A> &a, const matrix<unsigned short, A> &b)
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

	template <class A>
	tensor<signed int, A>& cpu_add(tensor<signed int, A> &c, const tensor<signed int, A> &a, const matrix<signed int, A> &b)
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

	template <class A>
	tensor<unsigned int, A>& cpu_add(tensor<unsigned int, A> &c, const tensor<unsigned int, A> &a, const matrix<unsigned int, A> &b)
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

	template <class A>
	tensor<float, A>& cpu_add(tensor<float, A> &c, const tensor<float, A> &a, const matrix<float, A> &b)
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

	template <class A>
	tensor<double, A>& cpu_add(tensor<double, A> &c, const tensor<double, A> &a, const matrix<double, A> &b)
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
