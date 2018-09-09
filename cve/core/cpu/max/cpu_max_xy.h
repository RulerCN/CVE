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

#ifndef __CORE_CPU_MAX_XY_H__
#define __CORE_CPU_MAX_XY_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/max/kernel_max.h"

namespace core
{
	// Computes the max of elements of a matrix

	template <class A>
	signed char& cpu_max_xy(signed char &b, const matrix<signed char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = int8_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned char& cpu_max_xy(unsigned char &b, const matrix<unsigned char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = uint8_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed short& cpu_max_xy(signed short &b, const matrix<signed short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = int16_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned short& cpu_max_xy(unsigned short &b, const matrix<unsigned short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = uint16_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed int& cpu_max_xy(signed int &b, const matrix<signed int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = int32_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned int& cpu_max_xy(unsigned int &b, const matrix<unsigned int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = uint32_max;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	float& cpu_max_xy(float &b, const matrix<float, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = flt_max;
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_max_xy(double &b, const matrix<double, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		b = dbl_max;
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	// Computes the max of elements of a matrix

	template <class A1, class A2>
	matrix<signed char, A1>& cpu_max_xy(matrix<signed char, A1> &b, const matrix<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned char, A1>& cpu_max_xy(matrix<unsigned char, A1> &b, const matrix<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed short, A1>& cpu_max_xy(matrix<signed short, A1> &b, const matrix<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned short, A1>& cpu_max_xy(matrix<unsigned short, A1> &b, const matrix<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<signed int, A1>& cpu_max_xy(matrix<signed int, A1> &b, const matrix<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<unsigned int, A1>& cpu_max_xy(matrix<unsigned int, A1> &b, const matrix<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<float, A1>& cpu_max_xy(matrix<float, A1> &b, const matrix<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A1, class A2>
	matrix<double, A1>& cpu_max_xy(matrix<double, A1> &b, const matrix<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	// Computes the max of elements across the x and y axis of a tensor

	template <class A1, class A2>
	vector<signed char, A1>& cpu_max_xy(vector<signed char, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned char, A1>& cpu_max_xy(vector<unsigned char, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed short, A1>& cpu_max_xy(vector<signed short, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned short, A1>& cpu_max_xy(vector<unsigned short, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<signed int, A1>& cpu_max_xy(vector<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<unsigned int, A1>& cpu_max_xy(vector<unsigned int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<float, A1>& cpu_max_xy(vector<float, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	vector<double, A1>& cpu_max_xy(vector<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	// Computes the max of elements across the x and y axis of a tensor

	template <class A1, class A2>
	tensor<signed char, A1>& cpu_max_xy(tensor<signed char, A1> &b, const tensor<signed char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned char, A1>& cpu_max_xy(tensor<unsigned char, A1> &b, const tensor<unsigned char, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed short, A1>& cpu_max_xy(tensor<signed short, A1> &b, const tensor<signed short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned short, A1>& cpu_max_xy(tensor<unsigned short, A1> &b, const tensor<unsigned short, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<signed int, A1>& cpu_max_xy(tensor<signed int, A1> &b, const tensor<signed int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<unsigned int, A1>& cpu_max_xy(tensor<unsigned int, A1> &b, const tensor<unsigned int, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_max);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<float, A1>& cpu_max_xy(tensor<float, A1> &b, const tensor<float, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_max);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

	template <class A1, class A2>
	tensor<double, A1>& cpu_max_xy(tensor<double, A1> &b, const tensor<double, A2> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.size() != a.batch())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_max);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.batch(), a.matrix_size(), a.data(), a.matrix_size(), b.data());
		return b;
	}

} // namespace core

#endif
