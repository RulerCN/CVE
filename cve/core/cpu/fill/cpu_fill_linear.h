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

#ifndef __CORE_CPU_FILL_LINEAR_H__
#define __CORE_CPU_FILL_LINEAR_H__

#include "../../scalar.h"
#include "../../vector.h"
#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/fill/kernel_fill_linear.h"

namespace core
{
	// Linear fill a scalar

	template <class A>
	scalar<signed char, A>& cpu_linear(scalar<signed char, A> &c, signed char a, signed char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<unsigned char, A>& cpu_linear(scalar<unsigned char, A> &c, unsigned char a, unsigned char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<signed short, A>& cpu_linear(scalar<signed short, A> &c, signed short a, signed short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<unsigned short, A>& cpu_linear(scalar<unsigned short, A> &c, unsigned short a, unsigned short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<signed int, A>& cpu_linear(scalar<signed int, A> &c, signed int a, signed int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<unsigned int, A>& cpu_linear(scalar<unsigned int, A> &c, unsigned int a, unsigned int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<signed long long, A>& cpu_linear(scalar<signed long long, A> &c, signed long long a, signed long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<unsigned long long, A>& cpu_linear(scalar<unsigned long long, A> &c, unsigned long long a, unsigned long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<float, A>& cpu_linear(scalar<float, A> &c, float a, float b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<float, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_fill_linear<float, cpu_sse>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<float, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	scalar<double, A>& cpu_linear(scalar<double, A> &c, double a, double b)
	{
		if (c.empty())
			throw ::std::invalid_argument(scalar_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<double, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<double, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<double, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	// Linear fill a vector

	template <class A>
	vector<signed char, A>& cpu_linear(vector<signed char, A> &c, signed char a, signed char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<unsigned char, A>& cpu_linear(vector<unsigned char, A> &c, unsigned char a, unsigned char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<signed short, A>& cpu_linear(vector<signed short, A> &c, signed short a, signed short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<unsigned short, A>& cpu_linear(vector<unsigned short, A> &c, unsigned short a, unsigned short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<signed int, A>& cpu_linear(vector<signed int, A> &c, signed int a, signed int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<unsigned int, A>& cpu_linear(vector<unsigned int, A> &c, unsigned int a, unsigned int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<signed long long, A>& cpu_linear(vector<signed long long, A> &c, signed long long a, signed long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<unsigned long long, A>& cpu_linear(vector<unsigned long long, A> &c, unsigned long long a, unsigned long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<float, A>& cpu_linear(vector<float, A> &c, float a, float b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<float, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_fill_linear<float, cpu_sse>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<float, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	vector<double, A>& cpu_linear(vector<double, A> &c, double a, double b)
	{
		if (c.empty())
			throw ::std::invalid_argument(vector_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<double, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<double, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<double, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	// Linear fill a matrix

	template <class A>
	matrix<signed char, A>& cpu_linear(matrix<signed char, A> &c, signed char a, signed char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<unsigned char, A>& cpu_linear(matrix<unsigned char, A> &c, unsigned char a, unsigned char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<signed short, A>& cpu_linear(matrix<signed short, A> &c, signed short a, signed short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<unsigned short, A>& cpu_linear(matrix<unsigned short, A> &c, unsigned short a, unsigned short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<signed int, A>& cpu_linear(matrix<signed int, A> &c, signed int a, signed int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<unsigned int, A>& cpu_linear(matrix<unsigned int, A> &c, unsigned int a, unsigned int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<signed long long, A>& cpu_linear(matrix<signed long long, A> &c, signed long long a, signed long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<unsigned long long, A>& cpu_linear(matrix<unsigned long long, A> &c, unsigned long long a, unsigned long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<float, A>& cpu_linear(matrix<float, A> &c, float a, float b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<float, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_fill_linear<float, cpu_sse>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<float, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	matrix<double, A>& cpu_linear(matrix<double, A> &c, double a, double b)
	{
		if (c.empty())
			throw ::std::invalid_argument(matrix_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<double, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<double, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<double, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	// Linear fill a tensor

	template <class A>
	tensor<signed char, A>& cpu_linear(tensor<signed char, A> &c, signed char a, signed char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<unsigned char, A>& cpu_linear(tensor<unsigned char, A> &c, unsigned char a, unsigned char b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned char, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned char, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned char, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<signed short, A>& cpu_linear(tensor<signed short, A> &c, signed short a, signed short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<unsigned short, A>& cpu_linear(tensor<unsigned short, A> &c, unsigned short a, unsigned short b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned short, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned short, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned short, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<signed int, A>& cpu_linear(tensor<signed int, A> &c, signed int a, signed int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<unsigned int, A>& cpu_linear(tensor<unsigned int, A> &c, unsigned int a, unsigned int b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned int, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned int, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned int, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<signed long long, A>& cpu_linear(tensor<signed long long, A> &c, signed long long a, signed long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<signed long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<signed long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<signed long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<unsigned long long, A>& cpu_linear(tensor<unsigned long long, A> &c, unsigned long long a, unsigned long long b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx2())
			kernel_fill_linear<unsigned long long, cpu_avx2>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<unsigned long long, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<unsigned long long, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<float, A>& cpu_linear(tensor<float, A> &c, float a, float b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<float, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse())
			kernel_fill_linear<float, cpu_sse>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<float, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

	template <class A>
	tensor<double, A>& cpu_linear(tensor<double, A> &c, double a, double b)
	{
		if (c.empty())
			throw ::std::invalid_argument(tensor_not_initialized);

		if (cpu_inst::is_support_avx())
			kernel_fill_linear<double, cpu_avx>(c.size(), a, b, c.data());
		else if (cpu_inst::is_support_sse2())
			kernel_fill_linear<double, cpu_sse2>(c.size(), a, b, c.data());
		else
			kernel_fill_linear<double, cpu_none>(c.size(), a, b, c.data());
		return c;
	}

} // namespace core

#endif
