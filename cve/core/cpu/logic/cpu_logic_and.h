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

#ifndef __CORE_CPU_LOGIC_AND_H__
#define __CORE_CPU_LOGIC_AND_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/logic/kernel_logic_and.h"

namespace core
{
	// The and function for scalar

	template <class A1, class A2>
	scalar<signed char, A1>& cpu_and(scalar<signed char, A1> &b, const scalar<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned char, A1>& cpu_and(scalar<unsigned char, A1> &b, const scalar<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed short, A1>& cpu_and(scalar<signed short, A1> &b, const scalar<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned short, A1>& cpu_and(scalar<unsigned short, A1> &b, const scalar<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<signed int, A1>& cpu_and(scalar<signed int, A1> &b, const scalar<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<unsigned int, A1>& cpu_and(scalar<unsigned int, A1> &b, const scalar<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_and(scalar<float, A1> &b, const scalar<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_and<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_and<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	scalar<double, A1>& cpu_and(scalar<double, A1> &b, const scalar<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The and function for vector

	template <class A1, class A2>
	vector<signed char, A1>& cpu_and(vector<signed char, A1> &b, const vector<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_and(vector<unsigned char, A1> &b, const vector<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_and(vector<signed short, A1> &b, const vector<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_and(vector<unsigned short, A1> &b, const vector<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_and(vector<signed int, A1> &b, const vector<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_and(vector<unsigned int, A1> &b, const vector<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_and(vector<float, A1> &b, const vector<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_and<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_and<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_and(vector<double, A1> &b, const vector<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The and function for matrix

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_and(matrix<signed char, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_and(matrix<unsigned char, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_and(matrix<signed short, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_and(matrix<unsigned short, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_and(matrix<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_and(matrix<unsigned int, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_and(matrix<float, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_and<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_and<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_and(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	// The and function for tensor

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_and(tensor<signed char, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_and(tensor<unsigned char, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned char, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned char, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned char, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_and(tensor<signed short, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_and(tensor<unsigned short, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned short, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned short, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned short, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_and(tensor<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<signed int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<signed int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<signed int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_and(tensor<unsigned int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_and<unsigned int, cpu_avx2>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<unsigned int, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<unsigned int, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_and(tensor<float, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<float, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_and<float, cpu_sse>()(b.size(), a.data(), b.data());
		else
			kernel_and<float, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_and(tensor<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_and<double, cpu_avx>()(b.size(), a.data(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_and<double, cpu_sse2>()(b.size(), a.data(), b.data());
		else
			kernel_and<double, cpu_none>()(b.size(), a.data(), b.data());
		return b;
	}

} // namespace core

#endif
