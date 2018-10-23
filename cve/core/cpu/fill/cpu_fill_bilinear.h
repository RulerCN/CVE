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

#ifndef __CORE_CPU_FILL_BILINEAR_H__
#define __CORE_CPU_FILL_BILINEAR_H__

#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/fill/kernel_fill_bilinear.h"

namespace core
{
	// Bilinear fill a vector

	template <class A>
	vector<signed char, A>& cpu_bilinear(vector<signed char, A> &d, signed char a, signed char b, signed char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed char, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed char, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed char, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<unsigned char, A>& cpu_bilinear(vector<unsigned char, A> &d, unsigned char a, unsigned char b, unsigned char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned char, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned char, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned char, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<signed short, A>& cpu_bilinear(vector<signed short, A> &d, signed short a, signed short b, signed short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed short, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed short, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed short, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<unsigned short, A>& cpu_bilinear(vector<unsigned short, A> &d, unsigned short a, unsigned short b, unsigned short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned short, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned short, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned short, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<signed int, A>& cpu_bilinear(vector<signed int, A> &d, signed int a, signed int b, signed int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed int, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed int, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed int, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<unsigned int, A>& cpu_bilinear(vector<unsigned int, A> &d, unsigned int a, unsigned int b, unsigned int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned int, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned int, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned int, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<signed long long, A>& cpu_bilinear(vector<signed long long, A> &d, signed long long a, signed long long b, signed long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed long long, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed long long, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed long long, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<unsigned long long, A>& cpu_bilinear(vector<unsigned long long, A> &d, unsigned long long a, unsigned long long b, unsigned long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned long long, cpu_avx2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned long long, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned long long, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<float, A>& cpu_bilinear(vector<float, A> &d, float a, float b, float c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<float, cpu_avx>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse())
			kernel_fill_bilinear<float, cpu_sse>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<float, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	vector<double, A>& cpu_bilinear(vector<double, A> &d, double a, double b, double c)
	{
		if (d.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<double, cpu_avx>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<double, cpu_sse2>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<double, cpu_none>(d.length(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	// Bilinear fill a matrix

	template <class A>
	matrix<signed char, A>& cpu_bilinear(matrix<signed char, A> &d, signed char a, signed char b, signed char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed char, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed char, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed char, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_bilinear(matrix<unsigned char, A> &d, unsigned char a, unsigned char b, unsigned char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned char, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned char, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned char, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<signed short, A>& cpu_bilinear(matrix<signed short, A> &d, signed short a, signed short b, signed short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed short, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed short, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed short, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_bilinear(matrix<unsigned short, A> &d, unsigned short a, unsigned short b, unsigned short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned short, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned short, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned short, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<signed int, A>& cpu_bilinear(matrix<signed int, A> &d, signed int a, signed int b, signed int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed int, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed int, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed int, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_bilinear(matrix<unsigned int, A> &d, unsigned int a, unsigned int b, unsigned int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned int, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned int, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned int, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<signed long long, A>& cpu_bilinear(matrix<signed long long, A> &d, signed long long a, signed long long b, signed long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed long long, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed long long, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed long long, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<unsigned long long, A>& cpu_bilinear(matrix<unsigned long long, A> &d, unsigned long long a, unsigned long long b, unsigned long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned long long, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned long long, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned long long, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<float, A>& cpu_bilinear(matrix<float, A> &d, float a, float b, float c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<float, cpu_avx>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse())
			kernel_fill_bilinear<float, cpu_sse>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<float, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	matrix<double, A>& cpu_bilinear(matrix<double, A> &d, double a, double b, double c)
	{
		if (d.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<double, cpu_avx>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<double, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<double, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	// Bilinear fill a tensor

	template <class A>
	tensor<signed char, A>& cpu_bilinear(tensor<signed char, A> &d, signed char a, signed char b, signed char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed char, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed char, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed char, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_bilinear(tensor<unsigned char, A> &d, unsigned char a, unsigned char b, unsigned char c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned char, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned char, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned char, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<signed short, A>& cpu_bilinear(tensor<signed short, A> &d, signed short a, signed short b, signed short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed short, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed short, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed short, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_bilinear(tensor<unsigned short, A> &d, unsigned short a, unsigned short b, unsigned short c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned short, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned short, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned short, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<signed int, A>& cpu_bilinear(tensor<signed int, A> &d, signed int a, signed int b, signed int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed int, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed int, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed int, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_bilinear(tensor<unsigned int, A> &d, unsigned int a, unsigned int b, unsigned int c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned int, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned int, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned int, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<signed long long, A>& cpu_bilinear(tensor<signed long long, A> &d, signed long long a, signed long long b, signed long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<signed long long, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<signed long long, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<signed long long, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<unsigned long long, A>& cpu_bilinear(tensor<unsigned long long, A> &d, unsigned long long a, unsigned long long b, unsigned long long c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_bilinear<unsigned long long, cpu_avx2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<unsigned long long, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<unsigned long long, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<float, A>& cpu_bilinear(tensor<float, A> &d, float a, float b, float c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<float, cpu_avx>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse())
			kernel_fill_bilinear<float, cpu_sse>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<float, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

	template <class A>
	tensor<double, A>& cpu_bilinear(tensor<double, A> &d, double a, double b, double c)
	{
		if (d.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_bilinear<double, cpu_avx>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_bilinear<double, cpu_sse2>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		else
			kernel_fill_bilinear<double, cpu_none>(d.area(), d.dimension(), a, b, c, d.data(), d.dimension());
		return d;
	}

} // namespace core

#endif
