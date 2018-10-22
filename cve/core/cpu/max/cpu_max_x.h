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

#ifndef __CORE_CPU_MAX_X_H__
#define __CORE_CPU_MAX_X_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/max/kernel_max.h"

namespace core
{
	// Computes the max of elements of a vector

	template <class A>
	signed char& cpu_max_x(signed char &b, const vector<signed char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = int8_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned char& cpu_max_x(unsigned char &b, const vector<unsigned char, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = uint8_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed short& cpu_max_x(signed short &b, const vector<signed short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = int16_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned short& cpu_max_x(unsigned short &b, const vector<unsigned short, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = uint16_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	signed int& cpu_max_x(signed int &b, const vector<signed int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = int32_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	unsigned int& cpu_max_x(unsigned int &b, const vector<unsigned int, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = uint32_min;
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	float& cpu_max_x(float &b, const vector<float, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = flt_min;
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	template <class A>
	double& cpu_max_x(double &b, const vector<double, A> &a)
	{
		if (a.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		b = dbl_min;
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), &b);
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), &b);
		else
			kernel_max<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), &b);
		return b;
	}

	// Computes the max of elements of a vector

	template <class A>
	vector<signed char, A>& cpu_max_x(vector<signed char, A> &b, const vector<signed char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned char, A>& cpu_max_x(vector<unsigned char, A> &b, const vector<unsigned char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<signed short, A>& cpu_max_x(vector<signed short, A> &b, const vector<signed short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned short, A>& cpu_max_x(vector<unsigned short, A> &b, const vector<unsigned short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<signed int, A>& cpu_max_x(vector<signed int, A> &b, const vector<signed int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned int, A>& cpu_max_x(vector<unsigned int, A> &b, const vector<unsigned int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<float, A>& cpu_max_x(vector<float, A> &b, const vector<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_min);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	template <class A>
	vector<double, A>& cpu_max_x(vector<double, A> &b, const vector<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (b.size() != size_t(1))
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_min);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(size_t(1), a.size(), a.data(), a.size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(size_t(1), a.size(), a.data(), a.size(), b.data());
		return b;
	}

	// Computes the max of elements across the x axis of a matrix

	template <class A>
	vector<signed char, A>& cpu_max_x(vector<signed char, A> &b, const matrix<signed char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned char, A>& cpu_max_x(vector<unsigned char, A> &b, const matrix<unsigned char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<signed short, A>& cpu_max_x(vector<signed short, A> &b, const matrix<signed short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned short, A>& cpu_max_x(vector<unsigned short, A> &b, const matrix<unsigned short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<signed int, A>& cpu_max_x(vector<signed int, A> &b, const matrix<signed int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<unsigned int, A>& cpu_max_x(vector<unsigned int, A> &b, const matrix<unsigned int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<float, A>& cpu_max_x(vector<float, A> &b, const matrix<float, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_min);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	vector<double, A>& cpu_max_x(vector<double, A> &b, const matrix<double, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(vector_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_min);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the max of elements across the x axis of a matrix

	template <class A>
	matrix<signed char, A>& cpu_max_x(matrix<signed char, A> &b, const matrix<signed char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_max_x(matrix<unsigned char, A> &b, const matrix<unsigned char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed short, A>& cpu_max_x(matrix<signed short, A> &b, const matrix<signed short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_max_x(matrix<unsigned short, A> &b, const matrix<unsigned short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed int, A>& cpu_max_x(matrix<signed int, A> &b, const matrix<signed int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_max_x(matrix<unsigned int, A> &b, const matrix<unsigned int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<float, A>& cpu_max_x(matrix<float, A> &b, const matrix<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_min);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<double, A>& cpu_max_x(matrix<double, A> &b, const matrix<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (b.size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_min);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the max of elements across the x axis of a tensor

	template <class A>
	matrix<signed char, A>& cpu_max_x(matrix<signed char, A> &b, const tensor<signed char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_max_x(matrix<unsigned char, A> &b, const tensor<unsigned char, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed short, A>& cpu_max_x(matrix<signed short, A> &b, const tensor<signed short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_max_x(matrix<unsigned short, A> &b, const tensor<unsigned short, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<signed int, A>& cpu_max_x(matrix<signed int, A> &b, const tensor<signed int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_max_x(matrix<unsigned int, A> &b, const tensor<unsigned int, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<float, A>& cpu_max_x(matrix<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_min);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	matrix<double, A>& cpu_max_x(matrix<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.rows() != a.batch() || b.row_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_min);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	// Computes the max of elements across the x axis of a tensor

	template <class A>
	tensor<signed char, A>& cpu_max_x(tensor<signed char, A> &b, const tensor<signed char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed char, 32, 32, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed char, 16, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_max_x(tensor<unsigned char, A> &b, const tensor<unsigned char, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint8_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned char, 32, 32, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned char, 16, 16, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned char, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed short, A>& cpu_max_x(tensor<signed short, A> &b, const tensor<signed short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed short, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed short, 8, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_max_x(tensor<unsigned short, A> &b, const tensor<unsigned short, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint16_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned short, 16, 16, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned short, 8, 8, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned short, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<signed int, A>& cpu_max_x(tensor<signed int, A> &b, const tensor<signed int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(int32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<signed int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<signed int, 4, 4, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<signed int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_max_x(tensor<unsigned int, A> &b, const tensor<unsigned int, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(uint32_min);
		if (cpu_inst::is_support_avx2())
			kernel_max<unsigned int, 8, 8, cpu_avx2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse41())
			kernel_max<unsigned int, 4, 4, cpu_sse41>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<unsigned int, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<float, A>& cpu_max_x(tensor<float, A> &b, const tensor<float, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(flt_min);
		if (cpu_inst::is_support_avx())
			kernel_max<float, 8, 8, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse())
			kernel_max<float, 4, 4, cpu_sse>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<float, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

	template <class A>
	tensor<double, A>& cpu_max_x(tensor<double, A> &b, const tensor<double, A> &a)
	{
		if (b.empty() || a.empty())
			throw ::std::invalid_argument(tensor_not_initialized);
		if (b.batch() != a.batch() || b.matrix_size() != a.rows())
			throw ::std::invalid_argument(invalid_shape);

		b.fill(dbl_min);
		if (cpu_inst::is_support_avx())
			kernel_max<double, 4, 4, cpu_avx>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else if (cpu_inst::is_support_sse2())
			kernel_max<double, 2, 2, cpu_sse2>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		else
			kernel_max<double, 4, 4, cpu_none>(a.batch(), a.rows(), a.row_size(), a.data(), a.row_size(), b.data());
		return b;
	}

} // namespace core

#endif
