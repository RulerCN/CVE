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

#ifndef __CORE_CPU_CVTMUL_FLOAT_H__
#define __CORE_CPU_CVTMUL_FLOAT_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/convert/kernel_cvtmul_float.h"

namespace core
{
	// Scalar data-type conversion and scaling

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<signed char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<unsigned char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<signed short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<unsigned short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<signed int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<signed int, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<unsigned int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<float, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_cvtmul_float<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	scalar<float, A1>& cpu_cvtmul_float(scalar<float, A1> &c, float a, const scalar<double, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(scalar_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Vector data-type conversion and scaling

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<signed char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<unsigned char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<signed short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<unsigned short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<signed int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<signed int, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<unsigned int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<float, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_cvtmul_float<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_cvtmul_float(vector<float, A1> &c, float a, const vector<double, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Matrix data-type conversion and scaling

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<signed char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<unsigned char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<signed short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<unsigned short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<signed int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<signed int, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<unsigned int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<float, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_cvtmul_float<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_cvtmul_float(matrix<float, A1> &c, float a, const matrix<double, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	// Tensor data-type conversion and scaling

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<signed char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<unsigned char, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned char, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned char, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned char, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<signed short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<signed short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<signed short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<unsigned short, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned short, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse41())
			kernel_cvtmul_float<unsigned short, cpu_sse41>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned short, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<signed int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<signed int, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<signed int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<signed int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<unsigned int, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_cvtmul_float<unsigned int, cpu_avx2>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<unsigned int, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<unsigned int, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<float, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<float, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse())
			kernel_cvtmul_float<float, cpu_sse>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<float, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_cvtmul_float(tensor<float, A1> &c, float a, const tensor<double, A2> &b)
	{
		if (c.empty() || b.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (c.size() != b.size())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx())
			kernel_cvtmul_float<double, cpu_avx>()(c.size(), a, b.data(), c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_cvtmul_float<double, cpu_sse2>()(c.size(), a, b.data(), c.data());
		else
			kernel_cvtmul_float<double, cpu_none>()(c.size(), a, b.data(), c.data());
		return c;
	}

} // namespace core

#endif
